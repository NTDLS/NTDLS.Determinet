using NTDLS.Determinet.ActivationFunctions;
using NTDLS.Determinet.Types;
using ProtoBuf;
using System.IO.Compression;

namespace NTDLS.Determinet
{
    /// <summary>
    /// Represents a neural network designed for training and inference tasks.
    /// </summary>
    /// <remarks>The <see cref="DniNeuralNetwork"/> class provides functionality for creating, training, and
    /// evaluating a neural network. It supports configurable input, hidden, and output layers, as well as various
    /// activation functions and parameters. The network can be trained using backpropagation and gradient descent, and
    /// it calculates loss using the cross-entropy method with SoftMax activation for classification tasks.  This class
    /// also supports saving and loading the network's state to and from a file for persistence.</remarks>
    [ProtoContract]
    public class DniNeuralNetwork
    {
        public double LearningRate
        {
            get => State.LearningRate;
            set => State.LearningRate = value;
        }

        private static readonly ParallelOptions _parallelOpts = new()
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount
        };

        [ProtoMember(2)] public DniStateOfBeing State { get; private set; } = new();

        public DniNeuralNetwork(DniConfiguration configuration)
        {
            State.LearningRate = configuration.LearningRate;

            //Add input layer.
            State.Layers.Add(new DniLayer(DniLayerType.Input, configuration.InputNodes, new()));

            //Add hidden layer(s).
            foreach (var layerConfig in configuration.IntermediateLayers)
            {
                State.Layers.Add(new DniLayer(DniLayerType.Intermediate, layerConfig.Nodes, layerConfig.ActivationType, layerConfig.Parameters));
            }

            //Add output layer.
            State.Layers.Add(new DniLayer(DniLayerType.Output, configuration.OutputLayer.Nodes, configuration.OutputLayer.ActivationType, configuration.OutputLayer.Parameters));

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
                        double std = Math.Sqrt(2.0 / currentSize); // He init
                        layerWeights[j, k] = DniUtility.NextGaussian(0, std);
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
            => Forward(inputs, false);

        private double[] Forward(double[] inputs, bool isTraining)
        {
            State.Layers[0].Activations = inputs;

            for (int i = 1; i < State.Layers.Count; i++)
            {
                var layer = State.Layers[i];

                layer.Activations = // Weighted sum
                     ActivateLayer(State.Layers[i - 1].Activations, State.Synapses[i - 1].Weights, State.Synapses[i - 1].Biases);

                if (layer.Parameters.Get(DniParameters.LayerParameters.UseBatchNorm, false))
                {
                    BatchNormalize(layer, isTraining);
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
        private static void BatchNormalize(DniLayer layer, bool isTraining)
        {
            if (layer.Gamma == null || layer.Beta == null)
                return;

            double momentum = layer.Parameters.Get(DniParameters.LayerParameters.BatchNormMomentum, 0.2);

            // --- TRAINING ---
            if (isTraining)
            {
                // Compute global batch statistics for this layer.
                double batchMean = layer.Activations.Average();
                double batchVar = layer.Activations.Select(a => Math.Pow(a - batchMean, 2)).Average();

                // Update running stats (EMA)
                layer.RunningMean = momentum * layer.RunningMean + (1 - momentum) * batchMean;
                layer.RunningVariance = momentum * layer.RunningVariance + (1 - momentum) * batchVar;

                // Normalize using batch statistics.
                double stdDev = Math.Sqrt(batchVar + 1e-8);
                for (int i = 0; i < layer.Activations.Length; i++)
                {
                    layer.Activations[i] = layer.Gamma[i] * ((layer.Activations[i] - batchMean) / stdDev) + layer.Beta[i];
                }
            }
            // --- INFERENCE ---
            else
            {
                double stdDev = Math.Sqrt(layer.RunningVariance + 1e-8);
                for (int i = 0; i < layer.Activations.Length; i++)
                {
                    layer.Activations[i] = layer.Gamma[i] * ((layer.Activations[i] - layer.RunningMean) / stdDev) + layer.Beta[i];
                }
            }
        }


        /// <summary>
        /// Activates a layer
        /// </summary>
        private static double[] ActivateLayer(double[] inputs, double[,] weights, double[] biases)
        {
            int layerSize = biases.Length;
            double[] output = new double[layerSize];

            Parallel.For(0, layerSize, _parallelOpts, j =>
            {
                double sum = biases[j];
                for (int i = 0; i < inputs.Length; i++)
                {
                    double v = inputs[i] * weights[i, j];
                    if (double.IsNaN(v) || double.IsInfinity(v))
                        v = 0;
                    sum += v;
                }
                sum = Math.Clamp(sum, -1e6, 1e6);
                output[j] = sum;
            });

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

                Parallel.For(0, State.Layers[i].NodeCount, _parallelOpts, j =>
                {
                    double sum = 0.0;
                    for (int k = 0; k < State.Layers[i + 1].NodeCount; k++)
                        sum += errors.First()[k] * State.Synapses[i].Weights[j, k];

                    layerError[j] = sum * State.Layers[i].ActivateDerivative(j);
                });

                errors.Insert(0, layerError);
            }


            // Update weights and biases with gradient descent
            for (int i = 0; i < State.Synapses.Count; i++)
            {
                //These could be inlined, but are placed here to reduce index lookups and array/property accessors.
                var synapse = State.Synapses[i];
                var weights = synapse.Weights;
                var biases = synapse.Biases;
                var activations = State.Layers[i].Activations;
                var error = errors[i];
                double lr = State.LearningRate;

                // Parallelize across input neurons (rows)
                Parallel.For(0, weights.GetLength(0), _parallelOpts, j =>
                {
                    for (int k = 0; k < weights.GetLength(1); k++)
                    {
                        weights[j, k] -= lr * error[k] * activations[j];
                    }
                });

                // Bias updates (cheap — don’t bother parallelizing)
                for (int j = 0; j < biases.Length; j++)
                {
                    biases[j] -= lr * error[j];
                }

                // --- BatchNorm parameter updates (γ and β) ---
                if (State.Layers[i + 1].Parameters.Get(DniParameters.LayerParameters.UseBatchNorm, false))
                {
                    var bnLayer = State.Layers[i + 1];
                    if (bnLayer.Gamma != null && bnLayer.Beta != null)
                    {
                        var layerError = errors[i];
                        var batchMean = bnLayer.Activations.Average();
                        var batchVar = bnLayer.Activations.Select(a => Math.Pow(a - batchMean, 2)).Average();
                        var stdDev = Math.Sqrt(batchVar + 1e-8);

                        int nodeCount = Math.Min(bnLayer.NodeCount, layerError.Length);
                        for (int j = 0; j < nodeCount; j++)
                        {
                            var normalized = (bnLayer.Activations[j] - batchMean) / stdDev;

                            bnLayer.Gamma[j] -= lr * layerError[j] * normalized;
                            bnLayer.Beta[j] -= lr * layerError[j];

                            bnLayer.Gamma[j] -= 1e-5 * (bnLayer.Gamma[j] - 1.0);
                            bnLayer.Gamma[j] = Math.Clamp(bnLayer.Gamma[j], 0.01, 10.0);
                        }
                    }
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

            var predictions = Forward(inputs, true);

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
            using var zip = new DeflateStream(fs, CompressionLevel.SmallestSize);
            Serializer.Serialize(zip, State);
        }

        public static DniNeuralNetwork LoadFromFile(string filePath)
        {
            using var fs = File.OpenRead(filePath);
            using var zip = new DeflateStream(fs, CompressionMode.Decompress);
            var state = Serializer.Deserialize<DniStateOfBeing>(zip);

            foreach (var synapse in state.Synapses)
                synapse.RebuildAfterDeserialization();

            foreach (var layer in state.Layers)
                layer.InstantiateActivationFunction();

            return new DniNeuralNetwork { State = state };
        }
    }
}
