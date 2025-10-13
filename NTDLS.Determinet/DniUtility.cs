using NTDLS.Determinet.Types;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Text;

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

        public static Random Random
        {
            get
            {
                if (_random == null)
                {
                    var seed = Guid.NewGuid().GetHashCode();
                    _random = new Random(seed);
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

        public static int IndexOfMaxValue(double[] values, out double confidence)
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

        public static int Checksum(string buffer)
        {
            return Checksum(Encoding.ASCII.GetBytes(buffer));
        }

        public static int Checksum(byte[] buffer)
        {
            int sum = 0;
            foreach (var b in buffer)
            {
                sum += (int)(sum ^ b);
            }
            return sum;
        }

        /// <summary>
        /// Flips a coin with a probability between 0.0 - 1.0.
        /// </summary>
        /// <param name="probability"></param>
        /// <returns></returns>
        public static bool FlipCoin(double probability)
        {
            return (Random.Next(0, 1000) / 1000 >= probability);
        }

        public static bool FlipCoin()
        {
            return Random.Next(0, 100) >= 50;
        }

        public static double GetRandomNeuronValue()
        {
            if (FlipCoin())
            {
                return (double)(Random.NextDouble() / 0.5);
            }
            return (double)((Random.NextDouble() / 0.5f) * -1);
        }

        public static double GetRandomBiasValue()
        {
            if (FlipCoin())
            {
                return (double)(Random.NextDouble() / 0.5);
            }
            return (double)((Random.NextDouble() / 0.5f) * -1);
        }

        public static double GetRandomWeightValue()
        {
            if (FlipCoin())
            {
                return (double)(Random.NextDouble() / 0.5);
            }
            return (double)((Random.NextDouble() / 0.5f) * -1);
        }

        public static double NextDouble(double minimum, double maximum)
        {
            if (minimum < 0)
            {
                minimum = Math.Abs(minimum);
                if (FlipCoin())
                {
                    return (Random.NextDouble() * (maximum - minimum) + minimum) * -1;
                }
            }
            return Random.NextDouble() * (maximum - minimum) + minimum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void EnsureNotNull<T>([NotNull] T? value, string? message = null, [CallerArgumentExpression(nameof(value))] string strName = "")
        {
            if (value == null)
            {
                if (message == null)
                {
                    throw new Exception($"Value should not be null: '{strName}'.");
                }
                else
                {
                    throw new Exception(message);
                }
            }
        }

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
        internal static double[] GetLayerLabelValues(DniLayer layer, DniNamedLabelValues labelValues)
        {
            if (layer.Labels == null || layer.Labels.Length == 0)
                throw new InvalidOperationException("Input layer does not have labels defined.");

            if (layer.Labels.Length != layer.NodeCount)
                throw new InvalidOperationException("Input layer labels count does not match node count.");

            double[] values = new double[layer.NodeCount];

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
        internal static DniNamedLabelValues SetLayerLabelValues(DniLayer layer, double[] values)
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
