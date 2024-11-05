using Newtonsoft.Json;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    public class DniNeuralNetwork
    {
        [JsonProperty]
        private readonly List<DniLayer> _layers = new();

        public DniNeuralNetwork(DniConfiguration configuration)
        {
            if (configuration.InputNodes < 1)
            {
                throw new Exception("Input layer is not defined.");
            }

            if (configuration.Layers.Count < 1)
            {
                throw new Exception("Configuration must contain at least one hidden layer.");
            }

            if (configuration.OutputLayer == null || (configuration.OutputLayer?.Nodes ?? 0) < 1)
            {
                throw new Exception("Output layer is not defined.");
            }

            //For simplicity, we just add the output layer to the layers.
            configuration.Layers.Add(configuration.OutputLayer ?? throw new Exception("Output layer is not defined."));

            _layers = new List<DniLayer>
            {
                new DniLayer(DniLayerType.Input, configuration.InputNodes, configuration.Layers[0].Nodes, configuration.Layers[0].ActivationType)
            };

            for (int i = 1; i < configuration.Layers.Count; i++)
            {
                _layers.Add(new DniLayer(
                    configuration.Layers[i].LayerType,
                    configuration.Layers[i - 1].Nodes,
                    configuration.Layers[i].Nodes,
                    configuration.Layers[i].ActivationType));
            }
        }

        private DniNeuralNetwork(List<DniLayer> layers)
        {
            _layers = layers;
        }

        public double[] FeedForward(double[] inputs)
        {
            double[] activations = inputs;

            foreach (var layer in _layers)
            {
                activations = layer.FeedForward(activations);
            }

            return activations; // Final output layer activations
        }

        public double[] FeedForward(double[] inputs, out double highestActivationNode)
        {
            var results = FeedForward(inputs);
            highestActivationNode = DniUtility.GetIndexOfMaxValue(results, out _);
            return results;
        }

        public double[] FeedForward(double[] inputs, out int highestActivationNode, out double confidence)
        {
            var results = FeedForward(inputs);
            highestActivationNode = DniUtility.GetIndexOfMaxValue(results, out confidence);
            return results;
        }

        public void Train(double[] inputs, double[] trueLabel, double learningRate = 0.1)
        {
            // Feedforward and store activations and weighted sums
            double[] activations = inputs;
            var activationsList = new List<double[]> { activations };

            foreach (var layer in _layers)
            {
                activations = layer.FeedForward(activations);
                activationsList.Add(activations);
            }

            // Backpropagation
            double[] error = _layers[^1].CalculateOutputLayerError(activationsList[^1], trueLabel);

            // Adjust weights and biases for each layer in reverse
            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                error = _layers[i].Backpropagate(error, activationsList[i], learningRate, trueLabel);
            }
        }

        public void SaveToFile(string fileName)
        {
            var jsonText = JsonConvert.SerializeObject(_layers, Formatting.Indented);
            File.WriteAllText(fileName, jsonText);
        }

        public static DniNeuralNetwork LoadFromFile(string fileName)
        {
            var jsonText = File.ReadAllText(fileName);
            var layers = JsonConvert.DeserializeObject<List<DniLayer>>(jsonText)
                ?? throw new Exception("Failed to deserialize the network.");

            foreach (var layer in layers)
            {
                layer.InstantiateActivationFunction();
            }

            return new DniNeuralNetwork(layers);
        }
    }
}
