using Newtonsoft.Json;
using NTDLS.Determinet.ActivationFunctions;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    public class DniNeuralNetwork
    {
        public double LearningRate { get; set; }
        internal DniStateOfBeing State { get; private set; } = new();

        public DniNeuralNetwork(DniConfiguration configuration)
        {
            LearningRate = configuration.LearningRate;

            //Add input layer.
            State.Layers.Add(new DniLayer(DniLayerType.Input, configuration.InputNodes));

            //Add hidden layer(s).
            foreach (var layerConfig in configuration.IntermediateLayers)
            {
                State.Layers.Add(new DniLayer(DniLayerType.Intermediate, layerConfig.Nodes, layerConfig.ActivationType));
            }

            //Add output layer.
            State.Layers.Add(new DniLayer(DniLayerType.Output, configuration.OutputLayer.Nodes, configuration.OutputLayer.ActivationType));

            InitializeWeightsAndBiases();
        }

        public DniNeuralNetwork()
        {
            //Only used for deserialization.
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

        /// <summary>
        /// Forward pass through the network
        /// </summary>
        public double[] Forward(double[] inputs)
        {
            State.Layers[0].Activations = inputs;

            for (int i = 1; i < State.Layers.Count; i++)
            {
                State.Layers[i].Activations = ActivateLayer(State.Layers[i - 1].Activations, State.Synapses[i - 1].Weights, State.Synapses[i - 1].Biases);
                State.Layers[i].Activations = State.Layers[i].Activate();
            }

            return State.Layers.Last().Activations; // Return the output layer
        }

        /// <summary>
        /// Activates a layer
        /// </summary>
        private static double[] ActivateLayer(double[] inputs, double[,] weights, double[] biases)
        {
            int layerSize = biases.Length;
            double[] output = new double[layerSize];

            for (int j = 0; j < layerSize; j++)
            {
                output[j] = biases[j];
                for (int i = 0; i < inputs.Length; i++)
                {
                    output[j] += inputs[i] * weights[i, j];
                }
            }
            return output;
        }

        /// <summary>
        /// Cross-entropy loss derivative, used for SoftMax output activation function.
        /// </summary>
        private static double[] CrossEntropyLossGradient(double[] predicted, double[] actual)
        {
            var gradient = new double[predicted.Length];
            for (int i = 0; i < predicted.Length; i++)
            {
                gradient[i] = predicted[i] - actual[i];
            }
            return gradient;
        }

        /// <summary>
        /// Backpropagation for updating weights and biases
        /// </summary>
        private void Backpropagate(double[] inputs, double[] actualOutput)
        {
            Forward(inputs);

            List<double[]>? errors;

            if (State.Layers.Last().ActivationFunction is DniSoftMaxFunction)
            {
                var outputError = CrossEntropyLossGradient(State.Layers.Last().Activations, actualOutput);
                errors = new List<double[]> { outputError };
            }
            else
            {
                // Calculate the output error for the output layer, adjusting for the activation function
                var outputError = new double[State.Layers.Last().Activations.Length];
                for (int i = 0; i < outputError.Length; i++)
                {
                    var predicted = State.Layers.Last().Activations[i];
                    var target = actualOutput[i];

                    // Cross-entropy error combined with output activation derivative
                    outputError[i] = (predicted - target) * State.Layers.Last().ActivateDerivative(i);
                }
                errors = new List<double[]> { outputError };
            }

            // Calculate errors for each layer back through all hidden layers
            for (int i = State.Layers.Count - 2; i > 0; i--)
            {
                var layerError = new double[State.Layers[i].NodeCount];
                for (int j = 0; j < State.Layers[i].NodeCount; j++)
                {
                    layerError[j] = 0.0;
                    for (int k = 0; k < State.Layers[i + 1].NodeCount; k++)
                    {
                        layerError[j] += errors.First()[k] * State.Synapses[i].Weights[j, k];
                    }
                    layerError[j] *= State.Layers[i].ActivateDerivative(j);
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
                        State.Synapses[i].Weights[j, k] -= LearningRate * errors[i][k] * State.Layers[i].Activations[j];
                    }
                }

                for (int j = 0; j < State.Synapses[i].Biases.Length; j++)
                {
                    State.Synapses[i].Biases[j] -= LearningRate * errors[i][j];
                }
            }
        }

        /// <summary>
        /// Training function for single epoch.
        /// </summary>
        public double Train(double[] inputs, double[] outputs)
        {
            // Forward pass to get the prediction
            var predictions = Forward(inputs);

            // Calculate and accumulate the loss
            var loss = -Math.Log(predictions[Array.IndexOf(outputs, 1.0)]);

            // Perform backpropagation to update weights
            Backpropagate(inputs, outputs);

            return loss;
        }

        /// <summary>
        /// Training function for multiple epochs.
        /// </summary>
        public double Train(double[] inputs, double[] outputs, int epochs)
        {
            double loss = 0.0;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Forward pass to get the prediction
                var predictions = Forward(inputs);

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
