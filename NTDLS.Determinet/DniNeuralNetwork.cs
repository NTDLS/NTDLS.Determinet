using Newtonsoft.Json;

namespace NTDLS.Determinet
{
    public class DniNeuralNetwork
    {
        private readonly double learningRate;

        public DniStateOfBeing State { get; private set; } = new();

        public DniNeuralNetwork(DniConfiguration configuration)
        {
            learningRate = configuration.LearningRate;

            // Define layer sizes including input, hidden, and output layers.
            State.Layers.Add(new DniLayer(configuration.InputNodes));

            foreach (var layerConfig in configuration.IntermediateLayers)
            {
                State.Layers.Add(new DniLayer(layerConfig.Nodes, layerConfig.ActivationType));
            }
            State.Layers.Add(new DniLayer(configuration.OutputLayer.Nodes, configuration.OutputLayer.ActivationType));

            InitializeWeightsAndBiases();
        }

        public DniNeuralNetwork()
        {
        }

        /// <summary>
        /// Initialize weights and biases for each layer.
        /// </summary>
        private void InitializeWeightsAndBiases()
        {
            for (int i = 0; i < State.Layers.Count - 1; i++)
            {
                int currentSize = State.Layers[i].NodeCount;
                int nextSize = State.Layers[i + 1].NodeCount;

                var layerWeights = new double[currentSize, nextSize];
                var layerBiases = new double[nextSize];

                // Initialize weights and biases with small random values
                for (int j = 0; j < currentSize; j++)
                {
                    for (int k = 0; k < nextSize; k++)
                    {
                        layerWeights[j, k] = DniUtility.Random.NextDouble() - 0.5;
                    }
                }

                for (int j = 0; j < nextSize; j++)
                {
                    layerBiases[j] = DniUtility.Random.NextDouble() - 0.5;
                }

                State.Synapses.Add(new DniSynapse(layerWeights, layerBiases));
            }
        }

        // Forward pass through the network
        public double[] Forward(double[] inputs)
        {
            State.Layers[0].Activations = inputs;

            for (int i = 1; i < State.Layers.Count; i++)
            {
                State.Layers[i].Activations = ActivateLayer(State.Layers[i - 1].Activations, State.Synapses[i - 1].Weights, State.Synapses[i - 1].Biases);

                // Apply softmax only to the output layer
                if (i == State.Layers.Count - 1)
                    State.Layers[i].Activations = Softmax(State.Layers[i].Activations);
                else
                    State.Layers[i].Activations = State.Layers[i].Activations.Select(Sigmoid).ToArray(); // Sigmoid for hidden layers
            }

            return State.Layers.Last().Activations; // Return the output layer
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

            double[] outputError = CrossEntropyLossGradient(State.Layers.Last().Activations, actualOutput);
            List<double[]> errors = new List<double[]> { outputError };

            // Calculate errors for each layer back through all hidden layers
            for (int i = State.Layers.Count - 2; i > 0; i--)
            {
                double[] layerError = new double[State.Layers[i].NodeCount];
                for (int j = 0; j < State.Layers[i].NodeCount; j++)
                {
                    layerError[j] = 0.0;
                    for (int k = 0; k < State.Layers[i + 1].NodeCount; k++)
                        layerError[j] += errors.First()[k] * State.Synapses[i].Weights[j, k];
                    layerError[j] *= SigmoidDerivative(State.Layers[i].Activations[j]); // Use SigmoidDerivative here
                }
                errors.Insert(0, layerError);
            }

            // Update weights and biases with gradient descent
            for (int i = 0; i < State.Synapses.Count; i++)
            {
                for (int j = 0; j < State.Synapses[i].Weights.GetLength(0); j++)
                {
                    for (int k = 0; k < State.Synapses[i].Weights.GetLength(1); k++)
                    {
                        State.Synapses[i].Weights[j, k] -= learningRate * errors[i][k] * State.Layers[i].Activations[j];
                    }
                }

                for (int j = 0; j < State.Synapses[i].Biases.Length; j++)
                {
                    State.Synapses[i].Biases[j] -= learningRate * errors[i][j];
                }
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
            var jsonText = JsonConvert.SerializeObject(State, Formatting.Indented);
            File.WriteAllText(fileName, jsonText);
        }

        public static DniNeuralNetwork LoadFromFile(string fileName)
        {
            var jsonText = File.ReadAllText(fileName);

            var state = JsonConvert.DeserializeObject<DniStateOfBeing>(jsonText)
                ?? throw new Exception("Failed to deserialize the network.");

            foreach (var layer in state.Layers)
            {
                layer.InstantiateActivationFunction();
            }

            return new DniNeuralNetwork()
            {
                State = state
            };
        }
    }
}
