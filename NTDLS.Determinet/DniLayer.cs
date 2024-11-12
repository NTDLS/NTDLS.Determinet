using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using NTDLS.Determinet.ActivationFunctions;
using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    internal class DniLayer
    {
        public DniLayerType LayerType { get; set; }
        [JsonIgnore]
        public double[] Activations { get; set; }
        public int NodeCount { get; set; }
        public DniNamedFunctionParameters ActivationParameters { get; set; }

        [JsonConverter(typeof(StringEnumConverter))]
        public DniActivationType ActivationType { get; set; }

        [JsonIgnore]
        public IDniActivationFunction? ActivationFunction { get; internal set; }

        public DniLayer(DniLayerType layerType, int nodeCount, DniActivationType activationType, DniNamedFunctionParameters activationParameters)
        {
            ActivationParameters = activationParameters;
            LayerType = layerType;
            NodeCount = nodeCount;
            ActivationType = activationType;
            Activations = new double[nodeCount];
            InstantiateActivationFunction();
        }

        public DniLayer(DniLayerType layerType, int nodeCount, DniNamedFunctionParameters activationParameters)
        {
            ActivationParameters = activationParameters;
            LayerType = layerType;
            NodeCount = nodeCount;
            ActivationType = DniActivationType.None;
            Activations = new double[nodeCount];
            InstantiateActivationFunction();
        }

        public DniLayer()
        {
            //Only used for deserialization.
            Activations = Array.Empty<double>();
            ActivationParameters = new();
        }

        public double[] Activate()
        {
            if (ActivationFunction != null)
            {
                return ActivationFunction.Activation(Activations);
            }
            else
            {
                return Activations;
            }
        }

        public double ActivateDerivative(int nodeIndex)
        {
            if (ActivationFunction != null)
            {
                return ActivationFunction.Derivative(Activations[nodeIndex]);
            }
            else
            {
                return Activations[nodeIndex];
            }
        }

        internal void InstantiateActivationFunction()
        {
            ActivationFunction = ActivationType switch
            {
                DniActivationType.None => null,
                DniActivationType.Identity => new DniIdentityFunction(ActivationParameters),
                DniActivationType.ReLU => new DniReLUFunction(),
                DniActivationType.PiecewiseLinear => new DniPiecewiseLinearFunction(ActivationParameters),
                DniActivationType.Linear => new DniLinearFunction(ActivationParameters),
                DniActivationType.Sigmoid => new DniSigmoidFunction(),
                DniActivationType.Tanh => new DniTanhFunction(),
                DniActivationType.LeakyReLU => new DniLeakyReLUFunction(),
                DniActivationType.SoftMax => new DniSoftMaxFunction(),
                _ => throw new NotImplementedException($"Unknown activation type: [{ActivationType}].")
            };
        }
    }
}
