using NTDLS.Determinet.ActivationFunctions.Interfaces;
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
    /// evaluating a fully-connected feed-forward neural network. It is trained by backpropagation using either SGD or
    /// Adam. The loss is chosen from the output layer: a SoftMax output uses cross-entropy loss, any other output uses
    /// mean squared error (0.5 * sum((prediction - target)^2)).
    /// <para>Inference via <see cref="Forward(double[])"/> does not modify the network, so it may be called
    /// concurrently from multiple threads. Training methods mutate the network and must not run concurrently with
    /// each other or with inference.</para></remarks>
    public class DniNeuralNetwork
    {
        private const double LayerNormEpsilon = 1e-5;
        private const double AdamBeta1 = 0.9;
        private const double AdamBeta2 = 0.999;
        private const double AdamEpsilon = 1e-8;

        /// <summary>
        /// Below this amount of multiply-adds, loops run sequentially because thread dispatch would cost more than it saves.
        /// </summary>
        private const long ParallelWorkThreshold = 1 << 15;

        /// <summary>
        /// Block of input nodes processed per task when propagating error backward through a weight matrix.
        /// </summary>
        private const int BackpropBlockSize = 256;

        /// <summary>
        /// Gets the current state of being for the DNI (Digital Neural Interface).
        /// </summary>
        internal DniStateOfBeing State { get; private set; } = new();

        /// <summary>
        /// Reusable gradient accumulators, allocated on first training call.
        /// </summary>
        private DniGradients? _gradients;

        #region State Passthroughs.

        /// <summary>
        /// Gets the collection of named parameters associated with the current state.
        /// </summary>
        public DniNamedParameterCollection Parameters => State.Parameters;
        /// <summary>
        /// Gets the collection of layers associated with the current state.
        /// </summary>
        public List<DniLayer> Layers => State.Layers;
        /// <summary>
        /// Gets the collection of synapses associated with the current state.
        /// </summary>
        public List<DniSynapse> Synapses => State.Synapses;
        /// <summary>
        /// Gets the labels associated with the input fields.
        /// </summary>
        public string[]? InputLabels => State.InputLabels;
        /// <summary>
        /// Gets the array of output labels associated with the current state.
        /// </summary>
        public string[]? OutputLabels => State.OutputLabels;

        #endregion

        #region Constructors.

        /// <summary>
        /// Initializes a new instance of the <see cref="DniNeuralNetwork"/> class using the specified configuration.
        /// </summary>
        /// <param name="configuration">The configuration settings for the neural network, including learning rate, input layer, hidden layers, and
        /// output layer.</param>
        public DniNeuralNetwork(DniConfiguration configuration)
        {
            if (configuration.InputNodes <= 0)
                throw new ArgumentException("Input layer is not defined.", nameof(configuration));

            State.Parameters.Set(Network.LearningRate, configuration.LearningRate);

            //Add input layer.
            State.Layers.Add(new DniLayer(DniLayerType.Input, configuration.InputNodes, DniActivationType.None, new(), configuration.InputLabels));

            //Add hidden layer(s).
            foreach (var layerConfig in configuration.IntermediateLayers)
            {
                State.Layers.Add(new DniLayer(DniLayerType.Intermediate, layerConfig.Nodes, layerConfig.ActivationType, layerConfig.Parameters, null));
            }

            //Add output layer.
            State.Layers.Add(new DniLayer(DniLayerType.Output, configuration.OutputLayer.Nodes,
                configuration.OutputLayer.ActivationType, configuration.OutputLayer.Parameters, configuration.OutputLabels));

            InitializeWeightsAndBiases();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DniNeuralNetwork"/> class.
        /// </summary>
        /// <remarks>This constructor is intended for use during deserialization and should not be used
        /// directly in application code.</remarks>
        public DniNeuralNetwork()
        {
            //Only used for deserialization.
        }

        #endregion

        #region Initialization.

        /// <summary>
        /// Initializes weights with a zero-mean Gaussian whose variance is matched to the receiving layer's activation
        /// function, and initializes all biases to zero.
        /// </summary>
        /// <remarks>
        /// He (std = sqrt(2 / fanIn)) for ReLU-family activations, LeCun (std = sqrt(1 / fanIn)) for SELU, and
        /// Glorot/Xavier (std = sqrt(2 / (fanIn + fanOut))) for everything else (sigmoid, tanh, softmax, linear, ...).
        /// These keep activation variance roughly constant from layer to layer so deep networks neither explode nor vanish.
        /// </remarks>
        private void InitializeWeightsAndBiases()
        {
            for (int l = 1; l < State.Layers.Count; l++)
            {
                int fanIn = State.Layers[l - 1].NodeCount;
                int fanOut = State.Layers[l].NodeCount;

                double std = State.Layers[l].ActivationType switch
                {
                    DniActivationType.ReLU or DniActivationType.LeakyReLU or DniActivationType.ELU
                        or DniActivationType.Swish or DniActivationType.Mish or DniActivationType.SoftPlus
                        => Math.Sqrt(2.0 / fanIn),
                    DniActivationType.SELU
                        => Math.Sqrt(1.0 / fanIn),
                    _ => Math.Sqrt(2.0 / (fanIn + fanOut)),
                };

                var synapse = new DniSynapse(fanIn, fanOut);
                for (int w = 0; w < synapse.Weights.Length; w++)
                {
                    synapse.Weights[w] = DniUtility.NextGaussian(0, std);
                }

                State.Synapses.Add(synapse);
            }
        }

        #endregion

        #region Forward.

        /// <summary>
        /// Computes the output of the model for the given input values.
        /// </summary>
        /// <remarks>This method does not modify the network and is safe to call concurrently (but not concurrently with training).</remarks>
        /// <param name="inputs">An array of input values to be processed by the model. Cannot be null.</param>
        /// <returns>A new array of output values.</returns>
        public double[] Forward(double[] inputs)
        {
            ValidateInputs(inputs);

            var activations = inputs;
            for (int l = 1; l < State.Layers.Count; l++)
            {
                activations = ComputeLayer(l, activations, record: false);
            }
            return activations;
        }

        /// <summary>
        /// Computes the output of the model for the given labeled input values.
        /// </summary>
        /// <param name="labelValues">The labeled input values to process, represented as a <see cref="DniNamedLabelValues"/> object.</param>
        /// <returns>An array of <see cref="double"/> values representing the computed output of the model.</returns>
        public double[] Forward(DniNamedLabelValues labelValues)
            => Forward(State.Layers[0].GetLabelValues(labelValues));

        /// <summary>
        /// Computes the output of the model for the given input values and provides the corresponding label values.
        /// </summary>
        /// <param name="inputs">An array of input values to be processed by the model. The array must not be null.</param>
        /// <param name="outputLabelValues">When this method returns, contains the label values associated with the output of the model.</param>
        /// <returns>An array of output values produced by the model.</returns>
        public double[] Forward(double[] inputs, out DniNamedLabelValues outputLabelValues)
        {
            var outputs = Forward(inputs);
            outputLabelValues = State.Layers.Last().SetLabelValues(outputs);
            return outputs;
        }

        /// <summary>
        /// Processes the input label values through the network and produces the output values.
        /// </summary>
        /// <param name="labelValues">The input label values to be processed by the network.</param>
        /// <param name="outputLabelValues">When this method returns, contains the output label values corresponding to the final layer of the network.</param>
        /// <returns>An array of double values representing the output of the network after processing the input label values.</returns>
        public double[] Forward(DniNamedLabelValues labelValues, out DniNamedLabelValues outputLabelValues)
            => Forward(State.Layers[0].GetLabelValues(labelValues), out outputLabelValues);

        /// <summary>
        /// Forward pass that records every layer's intermediate values for use by <see cref="Backward"/>.
        /// </summary>
        private double[] ForwardTraining(double[] inputs)
        {
            ValidateInputs(inputs);

            var activations = (double[])inputs.Clone();
            State.Layers[0].PreActivations = activations;
            State.Layers[0].Activations = activations;

            for (int l = 1; l < State.Layers.Count; l++)
            {
                activations = ComputeLayer(l, activations, record: true);
            }
            return activations;
        }

        /// <summary>
        /// Computes one layer: weighted sum, optional layer normalization, then the activation function.
        /// </summary>
        /// <param name="layerIndex">Index of the layer being computed (must be &gt; 0).</param>
        /// <param name="input">Activations of the previous layer.</param>
        /// <param name="record">When true, stores intermediate values on the layer for backpropagation.</param>
        private double[] ComputeLayer(int layerIndex, double[] input, bool record)
        {
            var layer = State.Layers[layerIndex];

            var z = WeightedSum(input, State.Synapses[layerIndex - 1]);

            if (layer.UsesLayerNorm)
            {
                z = LayerNormalize(layer, z, record);
            }

            var activations = layer.Activate(z);

            if (record)
            {
                layer.PreActivations = z;
                layer.Activations = activations;
            }

            return activations;
        }

        /// <summary>
        /// Computes bias + W·input for every node of the receiving layer.
        /// </summary>
        private static double[] WeightedSum(double[] input, DniSynapse synapse)
        {
            int inputCount = synapse.InputCount;
            var weights = synapse.Weights;
            var biases = synapse.Biases;
            var output = new double[synapse.OutputCount];

            ParallelFor(synapse.OutputCount, inputCount, o =>
            {
                output[o] = biases[o] + DniMath.Dot(weights.AsSpan(o * inputCount, inputCount), input);
            });

            return output;
        }

        /// <summary>
        /// Normalizes <paramref name="u"/> to zero mean and unit variance across the layer's nodes, then applies the
        /// learned per-node scale (gamma) and shift (beta).
        /// </summary>
        private static double[] LayerNormalize(DniLayer layer, double[] u, bool record)
        {
            var gamma = layer.Gamma!;
            var beta = layer.Beta!;
            int n = u.Length;

            double mean = 0.0;
            for (int i = 0; i < n; i++)
                mean += u[i];
            mean /= n;

            double variance = 0.0;
            for (int i = 0; i < n; i++)
                variance += (u[i] - mean) * (u[i] - mean);
            variance /= n;

            double inverseStdDev = 1.0 / Math.Sqrt(variance + LayerNormEpsilon);

            var normalized = new double[n];
            var z = new double[n];
            for (int i = 0; i < n; i++)
            {
                normalized[i] = (u[i] - mean) * inverseStdDev;
                z[i] = gamma[i] * normalized[i] + beta[i];
            }

            if (record)
            {
                layer.Normalized = normalized;
                layer.InverseStdDev = inverseStdDev;
            }

            return z;
        }

        #endregion

        #region Training.

        /// <summary>
        /// Performs one optimizer step on a single sample.
        /// </summary>
        /// <param name="inputs">An array of input values representing the features for training.</param>
        /// <param name="expected">An array of expected output values corresponding to the inputs.</param>
        /// <returns>The loss for this sample, computed before the update.</returns>
        public double Train(double[] inputs, double[] expected)
        {
            var gradients = PrepareGradients();

            double loss = AccumulateSample(inputs, expected, gradients);
            ApplyUpdate(gradients, 1);

            State.Parameters.Set(Network.ComputedLoss, loss);
            return loss;
        }

        /// <summary>
        /// Trains the network on a mini-batch: gradients are averaged over the samples and a single optimizer step is applied.
        /// </summary>
        /// <param name="batchSize">The maximum number of samples to include in the training batch. Must be greater than 0.</param>
        /// <param name="dataProvider">Called once per sample; returns the input and expected values, or <see langword="null"/>
        /// if no more data is available (the batch is then processed with however many samples were gathered).</param>
        /// <returns>The average loss over the batch, computed before the update. Returns 0.0 if no samples were processed.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if <paramref name="batchSize"/> is less than or equal to 0.</exception>
        public double TrainBatch(int batchSize, Func<(double[] inputs, double[] expected)?> dataProvider)
        {
            if (batchSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be > 0.");

            return TrainBatch(EnumerateProvider(batchSize, dataProvider));
        }

        /// <summary>
        /// Trains the network on a mini-batch: gradients are averaged over the samples and a single optimizer step is applied.
        /// </summary>
        /// <param name="batch">The samples in the batch.</param>
        /// <returns>The average loss over the batch, computed before the update. Returns 0.0 if the batch is empty.</returns>
        public double TrainBatch(IEnumerable<(double[] inputs, double[] expected)> batch)
        {
            var gradients = PrepareGradients();

            double totalLoss = 0.0;
            int sampleCount = 0;

            foreach (var (inputs, expected) in batch)
            {
                totalLoss += AccumulateSample(inputs, expected, gradients);
                sampleCount++;
            }

            if (sampleCount == 0)
                return 0.0;

            ApplyUpdate(gradients, sampleCount);

            double loss = totalLoss / sampleCount;
            State.Parameters.Set(Network.ComputedLoss, loss);
            return loss;
        }

        /// <summary>
        /// Computes the loss of the network on a sample without modifying the network.
        /// </summary>
        public double ComputeLoss(double[] inputs, double[] expected)
        {
            ValidateInputs(inputs);
            ValidateExpected(expected);

            // Same as Forward(), but keeps the output layer's pre-activations so the loss can be evaluated stably from logits.
            var activations = inputs;
            var preActivations = inputs;
            var outputLayer = State.Layers.Last();

            for (int l = 1; l < State.Layers.Count; l++)
            {
                var layer = State.Layers[l];
                preActivations = WeightedSum(activations, State.Synapses[l - 1]);
                if (layer.UsesLayerNorm)
                    preActivations = LayerNormalize(layer, preActivations, record: false);
                activations = layer.Activate(preActivations);
            }

            return Loss(outputLayer, preActivations, activations, expected);
        }

        private static IEnumerable<(double[] inputs, double[] expected)> EnumerateProvider(
            int batchSize, Func<(double[] inputs, double[] expected)?> dataProvider)
        {
            for (int b = 0; b < batchSize; b++)
            {
                var sample = dataProvider();
                if (sample == null)
                    yield break;
                yield return sample.Value;
            }
        }

        private DniGradients PrepareGradients()
        {
            if (_gradients == null || !_gradients.Matches(State))
                _gradients = new DniGradients(State);
            else
                _gradients.Clear();

            return _gradients;
        }

        /// <summary>
        /// Runs a training forward pass on one sample, then backpropagates and adds its gradients to <paramref name="gradients"/>.
        /// </summary>
        /// <returns>The sample's loss.</returns>
        private double AccumulateSample(double[] inputs, double[] expected, DniGradients gradients)
        {
            ValidateExpected(expected);

            var outputLayer = State.Layers.Last();
            var predicted = ForwardTraining(inputs);

            double loss = Loss(outputLayer, outputLayer.PreActivations, predicted, expected);
            if (!double.IsFinite(loss))
                throw new InvalidOperationException(
                    $"Training diverged: loss is {loss}. Lower the learning rate or enable gradient clipping.");

            Backward(expected, gradients);

            return loss;
        }

        /// <summary>
        /// Loss of one sample. SoftMax outputs use cross-entropy evaluated from the logits via log-sum-exp (exact and
        /// stable even when a probability underflows to zero); all other outputs use 0.5 * squared error.
        /// </summary>
        private static double Loss(DniLayer outputLayer, double[] preActivations, double[] predicted, double[] expected)
        {
            if (outputLayer.ActivationFunction is IDniSoftMaxFunction softMax)
            {
                double invTemp = 1.0 / softMax.Temperature;
                double logSumExp = DniMath.LogSumExp(preActivations, invTemp);

                double loss = 0.0;
                for (int i = 0; i < expected.Length; i++)
                {
                    if (expected[i] != 0)
                    {
                        // log(p_i) = z_i / T - logSumExp
                        loss -= expected[i] * (preActivations[i] * invTemp - logSumExp);
                    }
                }
                return loss;
            }
            else
            {
                double loss = 0.0;
                for (int i = 0; i < expected.Length; i++)
                {
                    double diff = predicted[i] - expected[i];
                    loss += 0.5 * diff * diff;
                }
                return loss;
            }
        }

        /// <summary>
        /// Backpropagates the loss of the most recent <see cref="ForwardTraining"/> pass and adds the resulting
        /// gradients to <paramref name="gradients"/>.
        /// </summary>
        private void Backward(double[] expected, DniGradients gradients)
        {
            var layers = State.Layers;
            var outputLayer = layers.Last();

            // delta = dLoss / dPreActivation for the layer currently being processed.
            var delta = new double[outputLayer.NodeCount];

            if (outputLayer.ActivationFunction is IDniSoftMaxFunction softMax)
            {
                // Combined SoftMax + cross-entropy gradient. With p = softmax(z / T):
                // dL/dz_i = (p_i * sum(t) - t_i) / T, which is the familiar (p_i - t_i) / T for targets summing to 1.
                double targetSum = expected.Sum();
                double invTemp = 1.0 / softMax.Temperature;
                for (int i = 0; i < delta.Length; i++)
                    delta[i] = (outputLayer.Activations[i] * targetSum - expected[i]) * invTemp;
            }
            else
            {
                // Squared error: dL/dz_i = (a_i - t_i) * f'(z_i).
                for (int i = 0; i < delta.Length; i++)
                    delta[i] = (outputLayer.Activations[i] - expected[i]) * outputLayer.ActivateDerivative(i);
            }

            for (int l = layers.Count - 1; l >= 1; l--)
            {
                var layer = layers[l];
                var synapse = State.Synapses[l - 1];
                var previousActivations = layers[l - 1].Activations;

                if (layer.UsesLayerNorm)
                {
                    delta = LayerNormBackward(layer, delta, gradients.Gamma[l]!, gradients.Beta[l]!);
                }

                // delta is now dL/d(weighted sum) for this layer: dL/dW = delta ⊗ previousActivations and dL/db = delta.
                // Both arrays are freshly allocated per forward/backward pass, so they can be retained as-is.
                gradients.WeightFactors[l - 1].Add((delta, previousActivations));
                var biasGradient = gradients.Biases[l - 1];
                for (int o = 0; o < delta.Length; o++)
                    biasGradient[o] += delta[o];

                if (l > 1)
                {
                    var previousLayer = layers[l - 1];
                    var previousDelta = PropagateError(synapse, delta);
                    for (int i = 0; i < previousDelta.Length; i++)
                        previousDelta[i] *= previousLayer.ActivateDerivative(i);
                    delta = previousDelta;
                }
            }
        }

        /// <summary>
        /// Backward pass through layer normalization. Accumulates gamma/beta gradients and converts
        /// dL/d(normalized output) into dL/d(weighted sum).
        /// </summary>
        /// <remarks>
        /// With x̂ = (u - mean) * invStd and z = gamma * x̂ + beta:
        /// dL/du_i = invStd * (g_i - mean(g) - x̂_i * mean(g * x̂)), where g = dL/dx̂ = gamma * dL/dz.
        /// </remarks>
        private static double[] LayerNormBackward(DniLayer layer, double[] delta, double[] gammaGradient, double[] betaGradient)
        {
            var gamma = layer.Gamma!;
            var normalized = layer.Normalized;
            int n = delta.Length;

            var g = new double[n];
            double meanG = 0.0;
            double meanGX = 0.0;

            for (int i = 0; i < n; i++)
            {
                gammaGradient[i] += delta[i] * normalized[i];
                betaGradient[i] += delta[i];

                g[i] = delta[i] * gamma[i];
                meanG += g[i];
                meanGX += g[i] * normalized[i];
            }
            meanG /= n;
            meanGX /= n;

            var result = new double[n];
            for (int i = 0; i < n; i++)
                result[i] = layer.InverseStdDev * (g[i] - meanG - normalized[i] * meanGX);

            return result;
        }

        /// <summary>
        /// Computes Wᵀ·delta: the loss gradient with respect to the previous layer's activations.
        /// </summary>
        /// <remarks>Partitioned by blocks of input nodes so each task writes a disjoint, contiguous slice.</remarks>
        private static double[] PropagateError(DniSynapse synapse, double[] delta)
        {
            int inputCount = synapse.InputCount;
            int outputCount = synapse.OutputCount;
            var weights = synapse.Weights;
            var result = new double[inputCount];

            int blockCount = (inputCount + BackpropBlockSize - 1) / BackpropBlockSize;

            ParallelFor(blockCount, (long)BackpropBlockSize * outputCount, block =>
            {
                int start = block * BackpropBlockSize;
                int length = Math.Min(BackpropBlockSize, inputCount - start);
                var destination = result.AsSpan(start, length);

                for (int o = 0; o < outputCount; o++)
                {
                    if (delta[o] != 0)
                    {
                        DniMath.Axpy(delta[o], weights.AsSpan(o * inputCount + start, length), destination);
                    }
                }
            });

            return result;
        }

        #endregion

        #region Optimization.

        /// <summary>
        /// Averages the accumulated gradients over <paramref name="sampleCount"/>, applies global-norm clipping, and
        /// updates every parameter with SGD or Adam.
        /// </summary>
        private void ApplyUpdate(DniGradients gradients, int sampleCount)
        {
            double learningRate = State.Parameters.Get<double>(Network.LearningRate);
            double weightDecay = State.Parameters.Get<double>(Network.WeightDecay);
            double gradientClip = State.Parameters.Get<double>(Network.GradientClip);
            bool useAdam = State.Parameters.Get<bool>(Network.UseAdamOptimization);

            if (!(learningRate > 0) || !double.IsFinite(learningRate))
                throw new InvalidOperationException($"Learning rate must be a finite value > 0, got {learningRate}.");

            double scale = 1.0 / sampleCount;
            double norm = Math.Sqrt(gradients.SumOfSquares()) * scale;

            if (!double.IsFinite(norm))
                throw new InvalidOperationException(
                    "Training diverged: the gradient is not finite. Lower the learning rate or enable gradient clipping.");

            if (gradientClip > 0 && norm > gradientClip)
                scale *= gradientClip / norm;

            if (useAdam)
            {
                State.AdamTimeStep++;
                double correction1 = 1.0 - Math.Pow(AdamBeta1, State.AdamTimeStep);
                double correction2 = 1.0 - Math.Pow(AdamBeta2, State.AdamTimeStep);
                // Fold both bias corrections into the step size: lr * mHat / sqrt(vHat) == stepSize * m / sqrt(v).
                double stepSize = learningRate * Math.Sqrt(correction2) / correction1;
                double epsilon = AdamEpsilon * Math.Sqrt(correction2);

                for (int s = 0; s < State.Synapses.Count; s++)
                {
                    var synapse = State.Synapses[s];
                    synapse.EnsureAdamBuffers();

                    AdamUpdateWeights(synapse, gradients.WeightFactors[s], scale, stepSize, epsilon, learningRate * weightDecay);
                    AdamUpdate(synapse.Biases, gradients.Biases[s], synapse.AdamMeanBiases, synapse.AdamVarianceBiases,
                        scale, stepSize, epsilon, 0.0);
                }

                for (int l = 0; l < State.Layers.Count; l++)
                {
                    var layer = State.Layers[l];
                    if (layer.UsesLayerNorm)
                    {
                        layer.EnsureAdamBuffers();
                        AdamUpdate(layer.Gamma!, gradients.Gamma[l]!, layer.AdamMeanGamma, layer.AdamVarianceGamma,
                            scale, stepSize, epsilon, 0.0);
                        AdamUpdate(layer.Beta!, gradients.Beta[l]!, layer.AdamMeanBeta, layer.AdamVarianceBeta,
                            scale, stepSize, epsilon, 0.0);
                    }
                }
            }
            else
            {
                for (int s = 0; s < State.Synapses.Count; s++)
                {
                    var synapse = State.Synapses[s];
                    SgdUpdateWeights(synapse, gradients.WeightFactors[s], scale, learningRate, weightDecay);
                    SgdUpdate(synapse.Biases, gradients.Biases[s], scale, learningRate, 0.0);
                }

                for (int l = 0; l < State.Layers.Count; l++)
                {
                    var layer = State.Layers[l];
                    if (layer.UsesLayerNorm)
                    {
                        SgdUpdate(layer.Gamma!, gradients.Gamma[l]!, scale, learningRate, 0.0);
                        SgdUpdate(layer.Beta!, gradients.Beta[l]!, scale, learningRate, 0.0);
                    }
                }
            }
        }

        /// <summary>
        /// SGD with L2 weight decay applied to a weight matrix whose gradient is held as outer-product factors:
        /// W -= lr * (scale * sum(delta ⊗ input) + decay * W), fused into a single pass over each row.
        /// </summary>
        private static void SgdUpdateWeights(DniSynapse synapse, List<(double[] Delta, double[] Input)> factors,
            double scale, double learningRate, double weightDecay)
        {
            int inputCount = synapse.InputCount;
            var weights = synapse.Weights;
            double keep = 1.0 - learningRate * weightDecay;

            ParallelFor(synapse.OutputCount, (long)inputCount * (factors.Count + 1), o =>
            {
                var row = weights.AsSpan(o * inputCount, inputCount);

                if (keep != 1.0)
                    DniMath.Scale(keep, row);

                foreach (var (delta, input) in factors)
                {
                    if (delta[o] != 0)
                        DniMath.Axpy(-learningRate * scale * delta[o], input, row);
                }
            });
        }

        /// <summary>
        /// Per-thread scratch row used to expand factored weight gradients for Adam.
        /// </summary>
        [ThreadStatic] private static double[]? _gradientRow;

        /// <summary>
        /// Adam with decoupled weight decay (AdamW) applied to a weight matrix whose gradient is held as outer-product
        /// factors. Each row's gradient is expanded into a scratch buffer and consumed immediately.
        /// </summary>
        private static void AdamUpdateWeights(DniSynapse synapse, List<(double[] Delta, double[] Input)> factors,
            double scale, double stepSize, double epsilon, double decoupledDecay)
        {
            int inputCount = synapse.InputCount;
            var weights = synapse.Weights;
            var mean = synapse.AdamMeanWeights;
            var variance = synapse.AdamVarianceWeights;

            ParallelFor(synapse.OutputCount, (long)inputCount * (factors.Count + 4), o =>
            {
                if (_gradientRow == null || _gradientRow.Length < inputCount)
                    _gradientRow = new double[inputCount];

                var gradient = _gradientRow.AsSpan(0, inputCount);
                gradient.Clear();

                foreach (var (delta, input) in factors)
                {
                    if (delta[o] != 0)
                        DniMath.Axpy(scale * delta[o], input, gradient);
                }

                int offset = o * inputCount;
                for (int i = 0; i < inputCount; i++)
                {
                    int w = offset + i;
                    double g = gradient[i];
                    mean[w] = AdamBeta1 * mean[w] + (1.0 - AdamBeta1) * g;
                    variance[w] = AdamBeta2 * variance[w] + (1.0 - AdamBeta2) * g * g;
                    weights[w] -= stepSize * mean[w] / (Math.Sqrt(variance[w]) + epsilon) + decoupledDecay * weights[w];
                }
            });
        }

        /// <summary>
        /// SGD with L2 weight decay: p -= lr * (scale * g + decay * p).
        /// </summary>
        private static void SgdUpdate(double[] parameters, double[] gradient, double scale, double learningRate, double weightDecay)
        {
            ParallelFor(parameters.Length, 1, i =>
            {
                parameters[i] -= learningRate * (scale * gradient[i] + weightDecay * parameters[i]);
            });
        }

        /// <summary>
        /// Adam with decoupled weight decay (AdamW).
        /// </summary>
        private static void AdamUpdate(double[] parameters, double[] gradient, double[] mean, double[] variance,
            double scale, double stepSize, double epsilon, double decoupledDecay)
        {
            ParallelFor(parameters.Length, 1, i =>
            {
                double g = scale * gradient[i];
                mean[i] = AdamBeta1 * mean[i] + (1.0 - AdamBeta1) * g;
                variance[i] = AdamBeta2 * variance[i] + (1.0 - AdamBeta2) * g * g;
                parameters[i] -= stepSize * mean[i] / (Math.Sqrt(variance[i]) + epsilon) + decoupledDecay * parameters[i];
            });
        }

        #endregion

        #region Helpers.

        private void ValidateInputs(double[] inputs)
        {
            ArgumentNullException.ThrowIfNull(inputs);

            if (inputs.Length != State.Layers[0].NodeCount)
                throw new ArgumentException($"Input length {inputs.Length} does not match expected size {State.Layers[0].NodeCount}.", nameof(inputs));
        }

        private void ValidateExpected(double[] expected)
        {
            ArgumentNullException.ThrowIfNull(expected);

            if (expected.Length != State.Layers.Last().NodeCount)
                throw new ArgumentException($"Expected length {expected.Length} does not match output size {State.Layers.Last().NodeCount}.", nameof(expected));
        }

        /// <summary>
        /// Runs <paramref name="body"/> for 0..count-1, in parallel only when the total work justifies it.
        /// </summary>
        private static void ParallelFor(int count, long workPerItem, Action<int> body)
        {
            if (count * workPerItem < ParallelWorkThreshold)
            {
                for (int i = 0; i < count; i++)
                    body(i);
            }
            else
            {
                Parallel.For(0, count, DniUtility.ParallelOptions, body);
            }
        }

        #endregion

        #region Serialization.

        /// <summary>
        /// Saves the network (including optimizer state) to a file.
        /// </summary>
        /// <param name="filePath">The full path of the file where the state will be saved.</param>
        public void SaveToFile(string filePath)
        {
            using var fs = File.Create(filePath);
            Save(fs);
        }

        /// <summary>
        /// Writes the network (including optimizer state) to a stream as compressed protobuf.
        /// </summary>
        public void Save(Stream stream)
        {
            using var zip = new DeflateStream(stream, CompressionLevel.SmallestSize, leaveOpen: true);
            Serializer.Serialize(zip, State);
        }

        /// <summary>
        /// Loads a <see cref="DniNeuralNetwork"/> from a file previously written by <see cref="SaveToFile"/>.
        /// Files written by earlier versions of this library are also supported.
        /// </summary>
        /// <param name="filePath">The path to the file that contains the serialized network.</param>
        public static DniNeuralNetwork LoadFromFile(string filePath)
        {
            using var fs = File.OpenRead(filePath);
            return Load(fs);
        }

        /// <summary>
        /// Reads a network previously written by <see cref="Save"/>.
        /// </summary>
        public static DniNeuralNetwork Load(Stream stream)
        {
            using var zip = new DeflateStream(stream, CompressionMode.Decompress, leaveOpen: true);
            var state = Serializer.Deserialize<DniStateOfBeing>(zip);

            foreach (var synapse in state.Synapses)
                synapse.AfterDeserialization();

            foreach (var layer in state.Layers)
                layer.AfterDeserialization();

            if (state.Layers.Count < 2 || state.Synapses.Count != state.Layers.Count - 1)
                throw new InvalidDataException($"Network has {state.Layers.Count} layers but {state.Synapses.Count} synapses.");

            for (int s = 0; s < state.Synapses.Count; s++)
            {
                var synapse = state.Synapses[s];
                if (synapse.InputCount != state.Layers[s].NodeCount || synapse.OutputCount != state.Layers[s + 1].NodeCount)
                    throw new InvalidDataException($"Synapse {s} shape {synapse.InputCount}x{synapse.OutputCount} does not match its layers.");
            }

            return new DniNeuralNetwork { State = state };
        }

        #endregion
    }
}
