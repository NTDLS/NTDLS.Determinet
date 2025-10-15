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

        /// <summary>
        /// Gets or sets the learning rate used to adjust the magnitude of updates during training.
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// Gets the number of input nodes in the system.
        /// </summary>
        public int InputNodes { get; private set; }

        /// <summary>
        /// Gets the output layer configuration for the current instance.
        /// </summary>
        public DniConfigurationLayer OutputLayer => _outputLayer ?? throw new Exception("Output layer is not defined.");

        /// <summary>
        /// Gets or sets the collection of intermediate configuration layers.
        /// </summary>
        public List<DniConfigurationLayer> IntermediateLayers { get; set; } = new();

        /// <summary>
        /// Optional labels for the input and output layer.
        /// </summary>
        public string[]? InputLabels { get; set; }

        /// <summary>
        /// Optional labels for the input and output layer.
        /// </summary>
        public string[]? OutputLabels { get; set; }

        /// <summary>
        /// Defines the input layer of the neural network with the specified number of nodes and optional labels.
        /// </summary>
        /// <remarks>This method can only be called once to define the input layer. Subsequent calls will
        /// result in an exception.</remarks>
        /// <param name="nodes">The number of nodes in the input layer. Must be greater than zero.</param>
        /// <param name="labels">An optional array of labels for the input nodes. If provided, the length of the array must match the number
        /// of nodes.</param>
        /// <exception cref="Exception">Thrown if the input layer is already defined or if the number of labels does not match the number of nodes.</exception>
        public void AddInputLayer(int nodes, string[]? labels = null)
        {
            if (InputNodes != 0)
                throw new Exception("Input layer is already defined.");

            if (labels != null && labels.Length != nodes)
                throw new Exception("Input layer label count does not match node count.");

            InputLabels = labels;
            InputNodes = nodes;
        }

        /// <summary>
        /// Adds an intermediate layer to the network with the specified number of nodes and activation type.
        /// </summary>
        /// <remarks>This method appends a new intermediate layer to the network's configuration. The
        /// layer is defined by the specified number of nodes, activation type, and any additional activation
        /// parameters.</remarks>
        /// <param name="nodes">The number of nodes in the intermediate layer. Must be a positive integer.</param>
        /// <param name="activationType">The activation function to be used by the layer.</param>
        /// <param name="activationParameters">Optional parameters for configuring the activation function. If not provided, an empty collection is used.</param>
        public void AddIntermediateLayer(int nodes, DniActivationType activationType, DniNamedParameterCollection? activationParameters = null)
        {
            IntermediateLayers.Add(new(DniLayerType.Intermediate, nodes, activationType, activationParameters ?? new()));
        }

        /// <summary>
        /// Adds an output layer to the model with the specified number of nodes, activation type, and optional
        /// parameters.
        /// </summary>
        /// <remarks>This method can only be called once to define the output layer. Subsequent calls will
        /// result in an exception.</remarks>
        /// <param name="nodes">The number of nodes in the output layer. Must be a positive integer.</param>
        /// <param name="activationType">The activation function to use for the output layer.</param>
        /// <param name="activationParameters">Optional parameters for the activation function. If not provided, default parameters will be used.</param>
        /// <param name="labels">Optional labels for the output layer nodes. If provided, the number of labels must match the number of
        /// nodes.</param>
        /// <exception cref="Exception"></exception>
        public void AddOutputLayer(int nodes, DniActivationType activationType, DniNamedParameterCollection? activationParameters = null, string[]? labels = null)
        {
            if (_outputLayer != null)
                throw new Exception("Output layer is already defined.");

            if (labels != null && labels.Length != nodes)
                throw new Exception("Output layer label count does not match node count.");

            OutputLabels = labels;
            _outputLayer = new(DniLayerType.Output, nodes, activationType, activationParameters ?? new());
        }

        /// <summary>
        /// Adds an output layer to the model with the specified number of nodes, activation type, and optional labels.
        /// </summary>
        /// <remarks>This method can only be called once to define the output layer. Subsequent calls will
        /// result in an exception.</remarks>
        /// <param name="nodes">The number of nodes in the output layer. Must be greater than zero.</param>
        /// <param name="activationType">The activation function to use for the output layer.</param>
        /// <param name="labels">An optional array of labels corresponding to the output nodes. If provided, the length of the array must
        /// match the number of nodes.</param>
        /// <exception cref="Exception"></exception>
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
