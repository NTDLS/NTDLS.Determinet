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
        /// <summary>
        /// Gets the activation values associated with the current instance.
        /// </summary>
        public double[] Activations { get; internal set; }

        /// <summary>
        /// Object instance of the activation function for this layer.
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

        #region Batch Normalization Properties.

        /// <summary>
        /// Gets or sets the running mean value used for batch normalization.
        /// See BatchNormalize() method.
        /// </summary>
        [ProtoMember(5)] public double RunningMean { get; set; }
        /// <summary>
        /// Gets or sets the running variance used for batch normalization.
        /// See BatchNormalize() method.
        /// </summary>
        [ProtoMember(6)] public double RunningVariance { get; set; }
        /// <summary>
        /// Gets or sets the gamma values used for batch normalization.
        /// See BatchNormalize() method.
        /// </summary>
        [ProtoMember(7)] public double[]? Gamma { get; set; }
        /// <summary>
        /// Gets or sets the beta values used for batch normalization.
        /// See BatchNormalize() method.
        /// </summary>
        [ProtoMember(8)] public double[]? Beta { get; set; }

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
        /// <remarks>This constructor initializes the layer's activation function and allocates memory for
        /// the activations based on the specified node count.</remarks>
        /// <param name="layerType">The type of the layer, which determines its role in the network.</param>
        /// <param name="nodeCount">The number of nodes (or neurons) in the layer. Must be a positive integer.</param>
        /// <param name="activationType">The activation function type to be used by the layer.</param>
        /// <param name="parameters">A collection of named parameters that configure the layer's behavior. These are also passed to the activation function.</param>
        /// <param name="labels">An optional array of labels associated with the layer. These are only used if the layer is an input or output layer.</param>
        public DniLayer(DniLayerType layerType, int nodeCount, DniActivationType activationType, DniNamedParameterCollection parameters, string[]? labels)
        {
            Labels = labels;
            Parameters = parameters;
            LayerType = layerType;
            NodeCount = nodeCount;
            ActivationType = activationType;
            Activations = new double[nodeCount];
            InstantiateActivationFunction();

            if (Parameters.Get(Layer.UseBatchNorm, false))
            {
                Gamma = Enumerable.Repeat(1.0, NodeCount).ToArray();
                Beta = new double[NodeCount];
                RunningMean = 0.0;
                RunningVariance = 1.0;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DniLayer"/> class.
        /// </summary>
        /// <remarks>This constructor is intended for use during deserialization. It initializes the <see
        /// cref="Activations"/>  property to an empty array and the <see cref="Parameters"/> property to a new instance
        /// of a collection.</remarks>
        public DniLayer()
        {
            //Only used for deserialization.
            Activations = Array.Empty<double>();
            Parameters = new();
        }

        /// <summary>
        /// Computes the output activations of the current layer.
        /// </summary>
        /// <remarks>If an activation function is defined, it is applied to the current activations. 
        /// Otherwise, the raw activations are returned.</remarks>
        /// <returns>An array of double values representing the output activations of the layer. If no activation function is
        /// defined, the raw activations are returned.</returns>
        public double[] Activate()
        {
            if (ActivationFunction != null)
            {
                return ActivationFunction.Activation(Activations);
            }
            return Activations;
        }

        /// <summary>
        /// Computes the derivative of the activation function for the specified node.
        /// </summary>
        /// <remarks>This method relies on the <see cref="ActivationFunction"/> to compute the derivative.
        /// If the <see cref="ActivationFunction"/> is <c>null</c>, the method directly returns the activation value of
        /// the specified node.</remarks>
        /// <param name="nodeIndex">The index of the node for which to compute the derivative. Must be a valid index within the
        /// <c>Activations</c> collection.</param>
        /// <returns>The derivative of the activation function for the specified node. If no activation function is set, returns
        /// the activation value of the node itself.</returns>
        public double ActivateDerivative(int nodeIndex)
        {
            if (ActivationFunction != null)
            {
                return ActivationFunction.Derivative(Activations[nodeIndex]);
            }
            return Activations[nodeIndex];
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
        }
    }
}
