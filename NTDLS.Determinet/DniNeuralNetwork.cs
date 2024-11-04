using Newtonsoft.Json;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    public class DniNeuralNetwork
    {
        [JsonProperty]
        private readonly List<DniLayer> _layers = new();

        public DniNeuralNetwork(int[] layerSizes)
        {
            _layers = new List<DniLayer>();

            for (int i = 1; i < layerSizes.Length; i++)
            {
                _layers.Add(new DniLayer(layerSizes[i - 1], layerSizes[i], DniActivationType.Sigmoid));
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

        public void Train(double[] inputs, double[] expectedOutputs, double learningRate = 0.1)
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
            double[] error = _layers[^1].CalculateOutputLayerError(activationsList[^1], expectedOutputs);

            // Adjust weights and biases for each layer in reverse
            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                error = _layers[i].Backpropagate(error, activationsList[i], learningRate);
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
