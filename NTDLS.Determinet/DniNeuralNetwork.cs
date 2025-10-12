using NTDLS.Determinet.ActivationFunctions;
using NTDLS.Determinet.Types;
using ProtoBuf;
using System.IO.Compression;
using static NTDLS.Determinet.DniParameters;

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
        /// Initializes the weights and biases for the neural network layers.
        /// </summary>
        /// <remarks>This method uses the He initialization technique to set the weights to small random
        /// values  based on the size of the current layer. Biases are initialized to random values in the range [-0.5, 0.5].
        /// The initialized weights and biases are stored in the <see cref="State.Synapses"/>
        /// collection.</remarks>
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
        /// Computes the output of the model for the given input values.
        /// </summary>
        /// <param name="inputs">An array of input values to be processed by the model. Cannot be null.</param>
        /// <returns>An array of output values resulting from processing the inputs. The length and content of the output depend
        /// on the model's configuration.</returns>
        public double[] Forward(double[] inputs)
            => Forward(inputs, false);

        /// <summary>
        /// Propagates the input values forward through the network, computing activations for each layer.
        /// </summary>
        /// <remarks>This method computes the activations for each layer in the network sequentially,
        /// starting from the input layer and ending with the output layer.  If batch normalization is enabled for a
        /// layer, it is applied during the forward pass. Nonlinear activation functions are applied to each layer's
        /// activations.</remarks>
        /// <param name="inputs">An array of input values to be fed into the network. The length of the array must match the number of input
        /// neurons in the first layer.</param>
        /// <param name="isTraining">A boolean value indicating whether the network is in training mode. If <see langword="true"/>, certain
        /// operations, such as batch normalization, may behave differently to account for training-specific
        /// adjustments.</param>
        /// <returns>An array of output values representing the activations of the final layer of the network.</returns>
        private double[] Forward(double[] inputs, bool isTraining)
        {
            State.Layers[0].Activations = inputs;

            for (int i = 1; i < State.Layers.Count; i++)
            {
                var layer = State.Layers[i];

                layer.Activations = // Weighted sum
                     ActivateLayer(State.Layers[i - 1].Activations, State.Synapses[i - 1].Weights, State.Synapses[i - 1].Biases);

                if (layer.Parameters.Get(Layer.UseBatchNorm, false))
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

            double momentum = layer.Parameters.Get(Layer.BatchNormMomentum, 0.2);

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
        /// Computes the activation values for a layer in a neural network based on the provided inputs, weights, and
        /// biases.
        /// </summary>
        /// <remarks>This method performs a weighted sum of the inputs and biases for each neuron in the
        /// layer, clamping the result to the range [-1e6, 1e6]. The computation is parallelized for improved
        /// performance on large layers.</remarks>
        /// <param name="inputs">An array of input values to the layer. The length of this array must match the number of rows in <paramref
        /// name="weights"/>.</param>
        /// <param name="weights">A 2D array representing the weights of the connections between the input layer and the current layer. The
        /// number of rows must match the length of <paramref name="inputs"/>, and the number of columns must match the
        /// length of <paramref name="biases"/>.</param>
        /// <param name="biases">An array of bias values for the layer. The length of this array determines the size of the output layer.</param>
        /// <returns>An array of activation values for the layer. The length of the returned array matches the length of
        /// <paramref name="biases"/>.</returns>
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
        /// Computes the gradient of the cross-entropy loss function with respect to the predicted values.
        /// </summary>
        /// <param name="predicted">An array of predicted probabilities, where each value represents the model's predicted probability for a
        /// class. Values should be in the range [0, 1].</param>
        /// <param name="actual">An array of actual class probabilities, where each value represents the true probability for a class.
        /// Typically, this is a one-hot encoded array.</param>
        /// <returns>An array representing the gradient of the cross-entropy loss for each class. Each value is the difference
        /// between the predicted and actual probabilities.</returns>
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
        /// Performs the backpropagation algorithm to compute gradients and update the weights and biases of the neural
        /// network based on the provided inputs and expected outputs.
        /// </summary>
        /// <remarks>This method calculates the error for each layer of the neural network, starting from
        /// the output layer and propagating backward through the hidden layers. The errors are used to adjust the
        /// weights and biases of the network using gradient descent. If batch normalization is enabled, the method also
        /// updates the batch normalization parameters (gamma and beta). <para> The method supports parallelization for
        /// performance optimization during weight updates and error calculations. </para></remarks>
        /// <param name="inputs">The input values fed into the neural network during the forward pass.</param>
        /// <param name="actualOutput">The expected output values used to calculate the error during backpropagation.</param>
        private void Backpropagate(double[] inputs, double[] actualOutput)
        {
            List<double[]>? errors;

            if (State.Layers.Last().ActivationFunction is DniSimpleSoftMaxFunction)
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
                if (State.Layers[i + 1].Parameters.Get(Layer.UseBatchNorm, false))
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
        /// Trains the model using the provided input data and expected output values.
        /// </summary>
        /// <param name="inputs">An array of input values representing the features for training.</param>
        /// <param name="expected">An array of expected output values corresponding to the inputs.</param>
        /// <returns>The computed loss value, which quantifies the difference between the model's predictions and the expected outputs.</returns>
        /// <exception cref="Exception">Thrown if the learning rate is less than or equal to zero.</exception>
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

        /// <summary>
        /// Calculates the cross-entropy loss between the predicted probabilities and the expected values.
        /// </summary>
        /// <remarks>The method ensures numerical stability by clamping the predicted probabilities to
        /// avoid logarithms of zero.</remarks>
        /// <param name="predicted">An array of predicted probabilities, where each value represents the model's confidence for a specific
        /// class. Each value must be in the range [0, 1].</param>
        /// <param name="expected">An array of expected values, where each value represents the true probability for a specific class. Each
        /// value must be in the range [0, 1].</param>
        /// <returns>The cross-entropy loss as a non-negative double value. A lower value indicates better alignment between the
        /// predicted and expected probabilities.</returns>
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

        /// <summary>
        /// Saves the current state to a file at the specified path.
        /// </summary>
        /// <remarks>The state is serialized and compressed before being written to the file.  Ensure that
        /// the specified file path is valid and accessible to avoid exceptions.</remarks>
        /// <param name="filePath">The full path of the file where the state will be saved. The directory must exist, and the caller must have
        /// write permissions.</param>
        public void SaveToFile(string filePath)
        {
            foreach (var synapse in State.Synapses)
                synapse.PrepareForSerialization();

            using var fs = File.Create(filePath);
            using var zip = new DeflateStream(fs, CompressionLevel.SmallestSize);
            Serializer.Serialize(zip, State);
        }

        /// <summary>
        /// Loads a <see cref="DniNeuralNetwork"/> from a file containing its serialized state.
        /// </summary>
        /// <remarks>This method expects the file to contain a compressed serialized representation of the
        /// neural network's state. The method will decompress the file, deserialize the state, and reconstruct the
        /// neural network, including reinitializing any necessary runtime components such as activation functions and
        /// synapse connections.</remarks>
        /// <param name="filePath">The path to the file that contains the serialized state of the neural network. Must be a valid file path and
        /// point to an existing file.</param>
        /// <returns>A <see cref="DniNeuralNetwork"/> instance reconstructed from the serialized state in the specified file.</returns>
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
