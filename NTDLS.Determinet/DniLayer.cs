using NTDLS.Determinet.ActivationFunctions;
using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using ProtoBuf;

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
        public double[] Activations { get; internal set; }

        /// <summary>
        /// Object instance of the activation function for this layer.
        /// Set by InstantiateActivationFunction() method.
        /// </summary>
        public IDniActivationFunction? ActivationFunction { get; internal set; }

        [ProtoMember(1)] public DniLayerType LayerType { get; private set; }
        [ProtoMember(2)] public int NodeCount { get; private set; }
        [ProtoMember(3)] public DniNamedFunctionParameters Parameters { get; private set; }
        [ProtoMember(4)] public DniActivationType ActivationType { get; private set; }

        // [RunningMean], [RunningVariance], [Gamma], and [Beta] are used for Batch Normalization in BatchNormalize().
        [ProtoMember(5)] public double RunningMean { get; set; }
        [ProtoMember(6)] public double RunningVariance { get; set; }
        [ProtoMember(7)] public double[]? Gamma { get; set; }
        [ProtoMember(8)] public double[]? Beta { get; set; }

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
        public DniLayer(DniLayerType layerType, int nodeCount, DniActivationType activationType, DniNamedFunctionParameters parameters)
        {
            Parameters = parameters;
            LayerType = layerType;
            NodeCount = nodeCount;
            ActivationType = activationType;
            Activations = new double[nodeCount];
            InstantiateActivationFunction();

            if (Parameters.Get(DniParameters.LayerParameters.UseBatchNorm, false))
            {
                Gamma = Enumerable.Repeat(1.0, NodeCount).ToArray();
                Beta = new double[NodeCount];
                RunningMean = 0.0;
                RunningVariance = 1.0;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DniLayer"/> class with the specified layer type, node count,
        /// and parameters.
        /// </summary>
        /// <remarks>This constructor initializes the layer with the specified configuration, setting up
        /// the activation function and allocating space for activations. The <see cref="ActivationType"/> is set to
        /// <see cref="DniActivationType.None"/> by default.</remarks>
        /// <param name="layerType">The type of the layer, which determines its role in the network.</param>
        /// <param name="nodeCount">The number of nodes in the layer. Must be a positive integer.</param>
        /// <param name="parameters">A collection of named parameters that configure the layer's behavior. These are also passed to the activation function.</param>
        public DniLayer(DniLayerType layerType, int nodeCount, DniNamedFunctionParameters parameters)
        {
            Parameters = parameters;
            LayerType = layerType;
            NodeCount = nodeCount;
            ActivationType = DniActivationType.None;
            Activations = new double[nodeCount];
            InstantiateActivationFunction();

            if (Parameters.Get(DniParameters.LayerParameters.UseBatchNorm, false))
            {
                Gamma = Enumerable.Repeat(1.0, NodeCount).ToArray();
                Beta = new double[NodeCount];
                RunningMean = 0.0;
                RunningVariance = 1.0;
            }
        }

        public DniLayer()
        {
            //Only used for deserialization.
            Activations = Array.Empty<double>();
            Parameters = new();
        }

        public double[] Activate()
        {
            if (ActivationFunction != null)
            {
                return ActivationFunction.Activation(Activations);
            }
            return Activations;
        }

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
                _ => throw new NotImplementedException($"Unknown activation type: [{ActivationType}].")
            };
        }
    }
}
