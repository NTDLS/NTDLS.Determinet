using Newtonsoft.Json;
using NTDLS.Determinet.ActivationFunctions;
using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    internal class DniLayer
    {
        public DniLayerType LayerType { get; set; }
        public double[] Activations { get; internal set; }
        public int NodeCount { get; internal set; }
        public DniActivationType ActivationType { get; internal set; }

        [JsonIgnore]
        public IDniActivationFunction? ActivationFunction { get; internal set; }

        public DniLayer(DniLayerType layerType, int nodeCount, DniActivationType activationType)
        {
            LayerType = layerType;
            NodeCount = nodeCount;
            ActivationType = activationType;
            Activations = new double[nodeCount];
            InstantiateActivationFunction();
        }

        public DniLayer(DniLayerType layerType, int nodeCount)
        {
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
                //DniActivationType.Identity => new DniIdentityFunction(_activationParameter),
                DniActivationType.ReLU => new DniReLUFunction(),
                //DniActivationType.Linear => new DniLinearFunction(_activationParameter),
                DniActivationType.Sigmoid => new DniSigmoidFunction(),
                DniActivationType.Tanh => new DniTanhFunction(),
                DniActivationType.LeakyReLU => new DniLeakyReLUFunction(),
                DniActivationType.SoftMax => new DniSoftMaxFunction(),
                _ => throw new NotImplementedException($"Unknown activation type: [{ActivationType}].")
            };
        }
    }
}
