using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    /// <summary>
    /// Provides a collection of utility methods and properties for common operations,  including random number
    /// generation, array processing, checksum calculation,  probabilistic operations, and data
    /// compression/decompression.
    /// </summary>
    /// <remarks>This static class includes methods for generating random values, calculating checksums, 
    /// flipping coins with specified probabilities, and ensuring non-null values.</remarks>
    public static class DniUtility
    {
        private static Random? _random = null;

        internal static readonly ParallelOptions ParallelOptions = new()
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount
        };

        public static Random Random
        {
            get
            {
                if (_random == null)
                {
                    lock (typeof(DniUtility))
                    {
                        _random ??= new Random(Guid.NewGuid().GetHashCode());
                    }
                }
                return _random;
            }
        }

        public static double NextGaussian(double mean = 0, double stdDev = 1)
        {
            // Use Box-Muller transform to generate a normally distributed value
            double u1 = 1.0 - Random.NextDouble(); // Uniform(0,1] random doubles
            double u2 = 1.0 - Random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); // Standard normal (0,1)
            return mean + stdDev * randStdNormal; // Scale and shift to desired mean and standard deviation
        }

        public static int IndexOfMaxValue(this double[] values, out double confidence)
        {
            int maxIndex = 0;
            double maxValue = values[0];

            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] > maxValue)
                {
                    maxValue = values[i];
                    maxIndex = i;
                }
            }

            confidence = maxValue;

            return maxIndex;
        }

        public static double NextDouble(double minValue, double maxValue)
        {
            if (maxValue == minValue)
                return maxValue;

            if (maxValue <= minValue)
                throw new ArgumentOutOfRangeException(nameof(maxValue), "maxValue must be greater than minValue.");

            return Random.NextDouble() * (maxValue - minValue) + minValue;
        }

        /// <summary>
        /// Determines whether a random event occurs based on the specified probability.
        /// </summary>
        /// <param name="probability">A value between 0.0 and 1.0 representing the probability of the event occurring.  Must be greater than or
        /// equal to 0.0 and less than or equal to 1.0.</param>
        /// <returns><see langword="true"/> if the random event occurs based on the specified probability;  otherwise, <see
        /// langword="false"/>.</returns>
        public static bool ChanceIn(double probability)
            => Random.NextDouble() < probability;

        /// <summary>
        /// Simulates a coin flip and returns the result.
        /// </summary>
        /// <returns><see langword="true"/> if the result of the coin flip is heads; otherwise, <see langword="false"/> for
        /// tails.</returns>
        public static bool FlipCoin()
            => Random.NextDouble() < 0.5;

        /// <summary>
        /// Retrieves an array of label values for the specified layer based on the provided label-value mapping.
        /// </summary>
        /// <param name="layer">The layer containing labels and nodes. The layer must have labels defined, and the number of labels must
        /// match the node count.</param>
        /// <param name="labelValues">A mapping of label names to their corresponding values. Labels not present in this mapping will be assigned
        /// a value of 0.</param>
        /// <returns>An array of double values representing the label values for each node in the layer. If a label is not found
        /// in <paramref name="labelValues"/>, its corresponding value will be 0.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the layer does not have labels defined, or if the number of labels does not match the node count.</exception>
        internal static double[] GetLabelValues(this DniLayer layer, DniNamedLabelValues labelValues)
        {
            if (layer.Labels == null || layer.Labels.Length == 0)
                throw new InvalidOperationException("Input layer does not have labels defined.");

            if (layer.Labels.Length != layer.NodeCount)
                throw new InvalidOperationException("Input layer labels count does not match node count.");

            var values = new double[layer.NodeCount];

            for (int i = 0; i < layer.Labels.Length; i++)
            {
                if (labelValues.TryGetValue(layer.Labels[i], out var labelValue))
                {
                    values[i] = labelValue;
                }
                else
                {
                    values[i] = 0;
                }
            }

            return values;
        }

        /// <summary>
        /// Creates a <see cref="DniNamedLabelValues"/> instance by associating the labels of the specified layer with
        /// the corresponding values provided.
        /// </summary>
        /// <param name="layer">The layer containing the labels to be used. The layer must have labels defined, and the number of labels
        /// must match the node count.</param>
        /// <param name="values">An array of values to associate with the layer's labels. The length of this array must match the number of
        /// labels in the layer.</param>
        /// <returns>A <see cref="DniNamedLabelValues"/> instance containing the layer's labels and their corresponding values.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the layer does not have labels defined, or if the number of labels does not match the node count.</exception>
        internal static DniNamedLabelValues SetLabelValues(this DniLayer layer, double[] values)
        {
            if (layer.Labels == null || layer.Labels.Length == 0)
                throw new InvalidOperationException("Layer does not have labels defined.");

            if (layer.Labels.Length != layer.NodeCount)
                throw new InvalidOperationException("Layer labels count does not match node count.");

            var labelValues = new DniNamedLabelValues();

            for (int i = 0; i < layer.Labels.Length; i++)
            {
                labelValues.Set(layer.Labels[i], values[i]);
            }

            return labelValues;
        }
    }
}
