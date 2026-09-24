using NTDLS.Determinet.ActivationFunctions;
using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using ProtoBuf;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet
{
    /// <summary>
    /// Represents a layer in a neural network, including its type, node count, activation function, and configuration
    /// parameters.
    /// </summary>
    /// <remarks>The <see cref="DniLayer"/> class is a fundamental building block for constructing neural
    /// networks. It encapsulates the properties and behavior of a single layer, including its activation function and
    /// the activations of its nodes. Layers can be configured with different types, activation functions, and
    /// parameters to suit various network architectures.</remarks>
    [ProtoContract]
    public class DniLayer
    {
        #region Training-time state (not serialized). Populated by the most recent training forward pass.

        /// <summary>
        /// Output of the layer (after the activation function) from the most recent training forward pass.
        /// </summary>
        public double[] Activations { get; internal set; }

        /// <summary>
        /// Input to the activation function (after the weighted sum and optional layer normalization) from the most
        /// recent training forward pass. Backpropagation evaluates activation derivatives at these values.
        /// </summary>
        public double[] PreActivations { get; internal set; }

        /// <summary>
        /// Layer-normalized weighted sums (x-hat) from the most recent training forward pass. Only used when
        /// <see cref="UsesLayerNorm"/> is <see langword="true"/>.
        /// </summary>
        internal double[] Normalized { get; set; } = [];

        /// <summary>
        /// 1 / sqrt(variance + epsilon) of the weighted sums from the most recent training forward pass.
        /// </summary>
        internal double InverseStdDev { get; set; }

        #endregion

        /// <summary>
        /// Object instance of the activation function for this layer, or <see langword="null"/> for none (identity).
        /// Set by InstantiateActivationFunction() method.
        /// </summary>
        public IDniActivationFunction? ActivationFunction { get; internal set; }

        /// <summary>
        /// Gets the type of the layer represented by this instance.
        /// </summary>
        [ProtoMember(1)] public DniLayerType LayerType { get; private set; }
        /// <summary>
        /// Gets the total number of nodes in the structure.
        /// </summary>
        [ProtoMember(2)] public int NodeCount { get; private set; }
        /// <summary>
        /// A collection of named parameters that configure the layer's behavior. These are also passed to the activation function.
        /// </summary>
        [ProtoMember(3)] public DniNamedParameterCollection Parameters { get; private set; }
        /// <summary>
        /// Gets the activation type associated with the current instance.
        /// </summary>
        [ProtoMember(4)] public DniActivationType ActivationType { get; private set; }

        #region Layer Normalization.

        //ProtoMember 5 and 6 were the (removed) batch-norm running mean and variance.

        /// <summary>
        /// Learned per-node scale applied after normalization, or <see langword="null"/> when layer normalization is disabled.
        /// </summary>
        [ProtoMember(7)] public double[]? Gamma { get; internal set; }
        /// <summary>
        /// Learned per-node shift applied after normalization, or <see langword="null"/> when layer normalization is disabled.
        /// </summary>
        [ProtoMember(8)] public double[]? Beta { get; internal set; }

        [ProtoMember(10)] internal double[] AdamMeanGamma { get; set; } = [];
        [ProtoMember(11)] internal double[] AdamVarianceGamma { get; set; } = [];
        [ProtoMember(12)] internal double[] AdamMeanBeta { get; set; } = [];
        [ProtoMember(13)] internal double[] AdamVarianceBeta { get; set; } = [];

        /// <summary>
        /// Whether this layer applies layer normalization before its activation function.
        /// </summary>
        public bool UsesLayerNorm => Gamma != null && Beta != null;

        #endregion

        /// <summary>
        /// Gets or sets an optional array of labels associated with the layer.
        /// These are only used if the layer is an input or output layer.
        /// </summary>
        [ProtoMember(9)] public string[]? Labels { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="DniLayer"/> class with the specified layer type, node count,
        /// activation type, and parameters.
        /// </summary>
        /// <param name="layerType">The type of the layer, which determines its role in the network.</param>
        /// <param name="nodeCount">The number of nodes (or neurons) in the layer. Must be a positive integer.</param>
        /// <param name="activationType">The activation function type to be used by the layer.</param>
        /// <param name="parameters">A collection of named parameters that configure the layer's behavior. These are also passed to the activation function.</param>
        /// <param name="labels">An optional array of labels associated with the layer. These are only used if the layer is an input or output layer.</param>
        public DniLayer(DniLayerType layerType, int nodeCount, DniActivationType activationType, DniNamedParameterCollection parameters, string[]? labels)
        {
            if (nodeCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(nodeCount), $"{layerType} layer must have at least one node.");

            if (labels != null && labels.Length != nodeCount)
                throw new ArgumentException($"{layerType} layer label count ({labels.Length}) does not match node count ({nodeCount}).", nameof(labels));

            Labels = labels;
            Parameters = parameters;
            LayerType = layerType;
            NodeCount = nodeCount;
            ActivationType = activationType;
            Activations = new double[nodeCount];
            PreActivations = new double[nodeCount];
            InstantiateActivationFunction();

            if (Parameters.Get(Layer.UseLayerNorm, false))
            {
                if (layerType != DniLayerType.Intermediate)
                    throw new ArgumentException("Layer normalization is only supported on intermediate layers.", nameof(parameters));

                if (nodeCount < 2)
                    throw new ArgumentException("Layer normalization requires at least two nodes.", nameof(parameters));

                Gamma = Enumerable.Repeat(1.0, NodeCount).ToArray();
                Beta = new double[NodeCount];
            }
        }

        /// <summary>
        /// Used only for deserialization.
        /// </summary>
        public DniLayer()
        {
            Activations = Array.Empty<double>();
            PreActivations = Array.Empty<double>();
            Parameters = new();
        }

        /// <summary>
        /// Applies this layer's activation function to <paramref name="preActivations"/>, returning a new array.
        /// With no activation function the values are copied through unchanged.
        /// </summary>
        public double[] Activate(double[] preActivations)
            => ActivationFunction?.Activation(preActivations) ?? (double[])preActivations.Clone();

        /// <summary>
        /// Computes the derivative of the activation function for the specified node, evaluated at that node's
        /// pre-activation value from the most recent training forward pass.
        /// </summary>
        public double ActivateDerivative(int nodeIndex)
            => ActivationFunction?.Derivative(PreActivations[nodeIndex]) ?? 1.0;

        internal void EnsureAdamBuffers()
        {
            if (Gamma == null || AdamMeanGamma.Length == Gamma.Length)
                return;

            AdamMeanGamma = new double[Gamma.Length];
            AdamVarianceGamma = new double[Gamma.Length];
            AdamMeanBeta = new double[Gamma.Length];
            AdamVarianceBeta = new double[Gamma.Length];
        }

        internal void AfterDeserialization()
        {
            Activations = new double[NodeCount];
            PreActivations = new double[NodeCount];
            InstantiateActivationFunction();
        }

        internal void InstantiateActivationFunction()
        {
            ActivationFunction = ActivationType switch
            {
                DniActivationType.None => null,
                DniActivationType.Identity => new DniIdentityFunction(Parameters),
                DniActivationType.ReLU => new DniReLUFunction(Parameters),
                DniActivationType.PiecewiseLinear => new DniPiecewiseLinearFunction(Parameters),
                DniActivationType.Linear => new DniLinearFunction(Parameters),
                DniActivationType.Sigmoid => new DniSigmoidFunction(Parameters),
                DniActivationType.Tanh => new DniTanhFunction(Parameters),
                DniActivationType.LeakyReLU => new DniLeakyReLUFunction(Parameters),
                DniActivationType.SoftMax => new DniSoftMaxFunction(Parameters),
                DniActivationType.SimpleSoftMax => new DniSimpleSoftMaxFunction(Parameters),
                DniActivationType.ELU => new DniELUFunction(Parameters),
                DniActivationType.Gaussian => new DniGaussianFunction(Parameters),
                DniActivationType.HardSigmoid => new DniHardSigmoidFunction(Parameters),
                DniActivationType.HardTanh => new DniHardTanhFunction(Parameters),
                DniActivationType.Mish => new DniMishFunction(Parameters),
                DniActivationType.SELU => new DniSELUFunction(Parameters),
                DniActivationType.SoftSign => new DniSoftSignFunction(Parameters),
                DniActivationType.Swish => new DniSwishFunction(Parameters),
                DniActivationType.SoftPlus => new DniSoftPlusFunction(Parameters),
                _ => throw new NotImplementedException($"Unknown activation type: [{ActivationType}].")
            };

            if (ActivationFunction is IDniSoftMaxFunction && LayerType != DniLayerType.Output)
                throw new ArgumentException($"{ActivationType} is only valid on the output layer (it has no element-wise derivative).");
        }
    }
}
