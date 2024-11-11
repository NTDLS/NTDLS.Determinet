using Newtonsoft.Json;
using NTDLS.Determinet.ActivationFunctions;
using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    public class DniLayer
    {
        public double[] Activations { get; internal set; }
        public int NodeCount { get; internal set; }
        public DniActivationType ActivationType { get; internal set; }

        [JsonIgnore]
        public IDniActivationFunction? ActivationFunction { get; internal set; }

        public DniLayer(int nodeCount, DniActivationType activationType)
        {
            NodeCount = nodeCount;
            ActivationType = activationType;
            Activations = new double[nodeCount];
        }

        public DniLayer(int nodeCount)
        {
            NodeCount = nodeCount;
            ActivationType = DniActivationType.None;
            Activations = new double[nodeCount];
        }

        public DniLayer()
        {
            //Only used for deserialization.
            Activations = Array.Empty<double>();
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
                _ => throw new NotImplementedException($"Unknown activation type: [{ActivationType}].")
            };
        }
    }
}
