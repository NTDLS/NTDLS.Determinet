using NTDLS.Determinet.ActivationFunctions;
using NTDLS.Determinet.Types;
using ProtoBuf;

namespace NTDLS.Determinet
{
    [ProtoContract]
    public class DniNeuralNetwork
    {
        public double LearningRate
        {
            get => State.LearningRate;
            set => State.LearningRate = value;
        }

        [ProtoMember(2)] internal DniStateOfBeing State { get; private set; } = new();

        public DniNeuralNetwork(DniConfiguration configuration)
        {
            State.LearningRate = configuration.LearningRate;

            //Add input layer.
            State.Layers.Add(new DniLayer(DniLayerType.Input, configuration.InputNodes, new()));

            //Add hidden layer(s).
            foreach (var layerConfig in configuration.IntermediateLayers)
            {
                State.Layers.Add(new DniLayer(DniLayerType.Intermediate, layerConfig.Nodes, layerConfig.ActivationType, layerConfig.ActivationParameters));
            }

            //Add output layer.
            State.Layers.Add(new DniLayer(DniLayerType.Output, configuration.OutputLayer.Nodes, configuration.OutputLayer.ActivationType, configuration.OutputLayer.ActivationParameters));

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
                var layer = State.Layers[i];

                layer.Activations = // Weighted sum
                     ActivateLayer(State.Layers[i - 1].Activations, State.Synapses[i - 1].Weights, State.Synapses[i - 1].Biases);

                if (layer.ActivationParameters.Get("UseBatchNorm", false))
                {
                    BatchNormalize(layer);
                }

                // Nonlinear activation
                layer.Activations = layer.Activate();
            }

            return State.Layers.Last().Activations;
        }

        /// <summary>
        /// Normalizes the values in the specified array of activations using batch normalization.
        /// </summary>
        /// <remarks>This method applies batch normalization to the input array by adjusting the values to
        /// have a mean of 0 and a standard deviation of 1, followed by optional scaling and shifting.  The
        /// normalization is performed in place, modifying the original array.</remarks>
        /// <param name="activations">An array of activation values to be normalized. The array is modified in place.</param>
        private static void BatchNormalize(DniLayer layer)
        {
            double mean = layer.Activations.Average();
            double variance = layer.Activations.Select(a => Math.Pow(a - mean, 2)).Average();
            double stdDev = Math.Sqrt(variance + 1e-8);

            double gamma = layer.ActivationParameters.Get("BatchNormGamma", 1);  // scale
            double beta = layer.ActivationParameters.Get("BatchNormBeta", 1); ;  // shift

            for (int i = 0; i < layer.Activations.Length; i++)
                layer.Activations[i] = gamma * ((layer.Activations[i] - mean) / stdDev) + beta;
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
                double sum = biases[j];

                for (int i = 0; i < inputs.Length; i++)
                {
                    double v = inputs[i] * weights[i, j];

                    if (double.IsNaN(v) || double.IsInfinity(v))
                        v = 0; // neutralize bad math

                    sum += v;
                }

                // prevent overflow from huge weights
                sum = Math.Clamp(sum, -1e6, 1e6);
                output[j] = sum;
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
                        State.Synapses[i].Weights[j, k] -= State.LearningRate * errors[i][k] * State.Layers[i].Activations[j];
                    }
                }

                for (int j = 0; j < State.Synapses[i].Biases.Length; j++)
                {
                    State.Synapses[i].Biases[j] -= State.LearningRate * errors[i][j];
                }
            }
        }

        /// <summary>
        /// Training function for single epoch.
        /// Returns loss with SoftMax + cross-entropy,
        /// The loss is a measure of how surprised the model is by the correct answer.
        /// </summary>
        public double Train(double[] inputs, double[] expected)
        {
            if (State.LearningRate <= 0)
                throw new Exception("Learning rate must be greater than zero.");

            var predictions = Forward(inputs);

            // Compute numerically-stable loss
            double loss = CrossEntropy(predictions, expected);

            Backpropagate(inputs, expected);
            return loss;
        }

        private static double CrossEntropy(double[] predicted, double[] expected)
        {
            const double EPS = 1e-12; // prevent log(0)
            double loss = 0.0;
            for (int i = 0; i < expected.Length; i++)
            {
                double p = Math.Clamp(predicted[i], EPS, 1.0 - EPS);
                loss -= expected[i] * Math.Log(p);
            }
            return loss;
        }

        public void SaveToFile(string filePath)
        {
            foreach (var synapse in State.Synapses)
                synapse.PrepareForSerialization();

            using var fs = File.Create(filePath);
            Serializer.Serialize(fs, State);
        }

        public static DniNeuralNetwork LoadFromFile(string filePath)
        {
            using var fs = File.OpenRead(filePath);
            var state = Serializer.Deserialize<DniStateOfBeing>(fs);

            foreach (var synapse in state.Synapses)
                synapse.RebuildAfterDeserialization();

            foreach (var layer in state.Layers)
                layer.InstantiateActivationFunction();

            return new DniNeuralNetwork { State = state };
        }
    }
}
