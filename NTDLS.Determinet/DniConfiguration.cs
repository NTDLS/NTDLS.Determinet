using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet
{
    /// <summary>
    /// Represents the configuration for a deep neural network, including its layers, learning rate, and structure.
    /// </summary>
    /// <remarks>This class provides methods to define the input layer, intermediate layers, and output layer
    /// of the network. The configuration must include exactly one input layer and one output layer, with any number of
    /// intermediate layers.</remarks>
    public class DniConfiguration
    {
        private DniConfigurationLayer? _outputLayer;

        public double LearningRate { get; set; } = Network.DefaultLearningRate;
        public int InputNodes { get; private set; }
        public DniConfigurationLayer OutputLayer => _outputLayer ?? throw new Exception("Output layer is not defined.");
        public List<DniConfigurationLayer> IntermediateLayers { get; set; } = new();

        public void AddInputLayer(int nodes)
        {
            if (InputNodes != 0)
            {
                throw new Exception("Input layer is already defined.");
            }
            InputNodes = nodes;
        }

        public void AddIntermediateLayer(int nodes, DniActivationType activationType, DniNamedFunctionParameters? activationParameters = null)
        {
            IntermediateLayers.Add(new(DniLayerType.Intermediate, nodes, activationType, activationParameters ?? new()));
        }

        public void AddOutputLayer(int nodes, DniActivationType activationType, DniNamedFunctionParameters? activationParameters = null)
        {
            if (_outputLayer != null)
            {
                throw new Exception("Output layer is already defined.");
            }
            _outputLayer = new(DniLayerType.Output, nodes, activationType, activationParameters ?? new());
        }
    }
}
