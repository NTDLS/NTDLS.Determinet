using NTDLS.Determinet.Types;

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

        public double LearningRate { get; set; }
        public int InputNodes { get; private set; }
        public DniConfigurationLayer OutputLayer => _outputLayer ?? throw new Exception("Output layer is not defined.");
        public List<DniConfigurationLayer> IntermediateLayers { get; set; } = new();

        /// <summary>
        /// Optional labels for the input and output layer.
        /// </summary>
        public string[]? InputLabels { get; set; }

        /// <summary>
        /// Optional labels for the input and output layer.
        /// </summary>
        public string[]? OutputLabels { get; set; }

        public void AddInputLayer(int nodes, string[]? labels = null)
        {
            if (InputNodes != 0)
                throw new Exception("Input layer is already defined.");

            if (labels != null && labels.Length != nodes)
                throw new Exception("Input layer label count does not match node count.");

            InputLabels = labels;
            InputNodes = nodes;
        }

        public void AddIntermediateLayer(int nodes, DniActivationType activationType, DniNamedParameterCollection? activationParameters = null)
        {
            IntermediateLayers.Add(new(DniLayerType.Intermediate, nodes, activationType, activationParameters ?? new()));
        }

        public void AddOutputLayer(int nodes, DniActivationType activationType, DniNamedParameterCollection? activationParameters = null, string[]? labels = null)
        {
            if (_outputLayer != null)
                throw new Exception("Output layer is already defined.");

            if (labels != null && labels.Length != nodes)
                throw new Exception("Output layer label count does not match node count.");

            OutputLabels = labels;
            _outputLayer = new(DniLayerType.Output, nodes, activationType, activationParameters ?? new());
        }

        public void AddOutputLayer(int nodes, DniActivationType activationType, string[] labels)
        {
            if (_outputLayer != null)
                throw new Exception("Output layer is already defined.");

            if (labels != null && labels.Length != nodes)
                throw new Exception("Output layer label count does not match node count.");

            OutputLabels = labels;
            _outputLayer = new(DniLayerType.Output, nodes, activationType, new());
        }
    }
}
