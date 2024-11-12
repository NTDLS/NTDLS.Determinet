using System.Diagnostics.CodeAnalysis;
using System.IO.Compression;
using System.Runtime.CompilerServices;
using System.Text;

namespace NTDLS.Determinet
{
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

        public static int GetIndexOfMaxValue(double[] values, out double confidence)
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

        public static byte[] Compress(string data)
        {
            var byteArray = Encoding.UTF8.GetBytes(data);

            using var compressedStream = new MemoryStream();
            using (var gzipStream = new GZipStream(compressedStream, CompressionMode.Compress))
            {
                gzipStream.Write(byteArray, 0, byteArray.Length);
            }
            return compressedStream.ToArray();
        }

        public static string Decompress(byte[] compressedBytes)
        {
            using var compressedStream = new MemoryStream(compressedBytes);
            using var gzipStream = new GZipStream(compressedStream, CompressionMode.Decompress);
            using var decompressedStream = new MemoryStream();
            gzipStream.CopyTo(decompressedStream);
            var decompressedBytes = decompressedStream.ToArray();
            return Encoding.UTF8.GetString(decompressedBytes);
        }
    }
}
