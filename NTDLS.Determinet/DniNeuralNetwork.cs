using Newtonsoft.Json;

namespace NTDLS.Determinet
{
    public class DniNeuralNetwork
    {
        private readonly double learningRate;

        [JsonProperty]
        private List<int> _layerSizes = new();
        [JsonProperty]
        private List<double[]> _activations = new();
        [JsonProperty]
        private List<double[,]> weights = new();
        [JsonProperty]
        private List<double[]> biases = new();

        public DniNeuralNetwork(DniConfiguration configuration)
        {
            learningRate = configuration.LearningRate;

            // Define layer sizes including input, hidden, and output layers.
            _layerSizes.Add(configuration.InputNodes);

            foreach (var layer in configuration.IntermediateLayers)
            {
                _layerSizes.Add(layer.Nodes);
            }
            _layerSizes.Add(configuration.OutputLayer.Nodes);

            InitializeLayers();
            InitializeWeightsAndBiases();
        }

        public DniNeuralNetwork()
        {
        }

        /// <summary>
        /// Initialize each layer based on layer sizes.
        /// </summary>
        private void InitializeLayers()
        {
            _activations = new List<double[]>();
            foreach (int size in _layerSizes)
                _activations.Add(new double[size]);
        }

        /// <summary>
        /// Initialize weights and biases for each layer.
        /// </summary>
        private void InitializeWeightsAndBiases()
        {
            Random rand = new Random();

            weights = new List<double[,]>();
            biases = new List<double[]>();

            for (int i = 0; i < _layerSizes.Count - 1; i++)
            {
                int currentSize = _layerSizes[i];
                int nextSize = _layerSizes[i + 1];

                double[,] layerWeights = new double[currentSize, nextSize];
                double[] layerBiases = new double[nextSize];

                // Initialize weights and biases with small random values
                for (int j = 0; j < currentSize; j++)
                    for (int k = 0; k < nextSize; k++)
                        layerWeights[j, k] = rand.NextDouble() - 0.5;
                for (int j = 0; j < nextSize; j++)
                    layerBiases[j] = rand.NextDouble() - 0.5;

                weights.Add(layerWeights);
                biases.Add(layerBiases);
            }
        }

        // Forward pass through the network
        public double[] Forward(double[] inputs)
        {
            _activations[0] = inputs;

            for (int i = 1; i < _activations.Count; i++)
            {
                _activations[i] = ActivateLayer(_activations[i - 1], weights[i - 1], biases[i - 1]);

                // Apply softmax only to the output layer
                if (i == _activations.Count - 1)
                    _activations[i] = Softmax(_activations[i]);
                else
                    _activations[i] = _activations[i].Select(Sigmoid).ToArray(); // Sigmoid for hidden layers
            }

            return _activations.Last(); // Return the output layer
        }

        // Sigmoid activation function
        private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        // Sigmoid derivative function
        private static double SigmoidDerivative(double x) => x * (1.0 - x);

        // Softmax activation for the output layer
        private static double[] Softmax(double[] x)
        {
            double max = x.Max();
            double scale = x.Sum(v => Math.Exp(v - max));
            return x.Select(v => Math.Exp(v - max) / scale).ToArray();
        }

        // Activates a layer
        private static double[] ActivateLayer(double[] inputs, double[,] weights, double[] biases)
        {
            int layerSize = biases.Length;
            double[] output = new double[layerSize];

            for (int j = 0; j < layerSize; j++)
            {
                output[j] = biases[j];
                for (int i = 0; i < inputs.Length; i++)
                    output[j] += inputs[i] * weights[i, j];
            }
            return output;
        }

        // Cross-entropy loss derivative
        private static double[] CrossEntropyLossGradient(double[] predicted, double[] actual)
        {
            double[] gradient = new double[predicted.Length];
            for (int i = 0; i < predicted.Length; i++)
                gradient[i] = predicted[i] - actual[i];
            return gradient;
        }

        // Backpropagation for updating weights and biases
        public void Backpropagate(double[] inputs, double[] actualOutput)
        {
            Forward(inputs);

            double[] outputError = CrossEntropyLossGradient(_activations.Last(), actualOutput);
            List<double[]> errors = new List<double[]> { outputError };

            // Calculate errors for each layer back through all hidden layers
            for (int i = _activations.Count - 2; i > 0; i--)
            {
                double[] layerError = new double[_layerSizes[i]];
                for (int j = 0; j < _layerSizes[i]; j++)
                {
                    layerError[j] = 0.0;
                    for (int k = 0; k < _layerSizes[i + 1]; k++)
                        layerError[j] += errors.First()[k] * weights[i][j, k];
                    layerError[j] *= SigmoidDerivative(_activations[i][j]); // Use SigmoidDerivative here
                }
                errors.Insert(0, layerError);
            }

            // Update weights and biases with gradient descent
            for (int i = 0; i < weights.Count; i++)
            {
                for (int j = 0; j < weights[i].GetLength(0); j++)
                    for (int k = 0; k < weights[i].GetLength(1); k++)
                        weights[i][j, k] -= learningRate * errors[i][k] * _activations[i][j];

                for (int j = 0; j < biases[i].Length; j++)
                    biases[i][j] -= learningRate * errors[i][j];
            }
        }

        /// <summary>
        /// Training method for multiple epochs.
        /// </summary>
        public double Train(double[] inputs, double[] outputs)
        {
            // Forward pass to get the prediction
            double[] predictions = Forward(inputs);

            // Calculate and accumulate the loss
            double loss = -Math.Log(predictions[Array.IndexOf(outputs, 1.0)]);

            // Perform backpropagation to update weights
            Backpropagate(inputs, outputs);

            return loss;
        }

        /// <summary>
        /// Training method for multiple epochs.
        /// </summary>
        public double Train(double[] inputs, double[] outputs, int epochs)
        {
            double loss = 0.0;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Forward pass to get the prediction
                double[] predictions = Forward(inputs);

                // Calculate and accumulate the loss
                loss = -Math.Log(predictions[Array.IndexOf(outputs, 1.0)]);

                // Perform backpropagation to update weights
                Backpropagate(inputs, outputs);
            }

            // Return the final loss after all epochs
            return loss;
        }

        public void SaveToFile(string fileName)
        {
            var jsonText = JsonConvert.SerializeObject(this, Formatting.Indented);
            File.WriteAllText(fileName, jsonText);
        }

        public static DniNeuralNetwork LoadFromFile(string fileName)
        {
            var jsonText = File.ReadAllText(fileName);

            var dni = JsonConvert.DeserializeObject<DniNeuralNetwork>(jsonText)
                ?? throw new Exception("Failed to deserialize the network.");

            //InstantiateActivationFunction();

            return dni;
        }
    }
}
