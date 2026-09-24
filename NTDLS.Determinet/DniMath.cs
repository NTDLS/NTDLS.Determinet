using System.Numerics;

namespace NTDLS.Determinet
{
    /// <summary>
    /// Numerically stable scalar helpers shared by activation functions and loss computations.
    /// </summary>
    internal static class DniMath
    {
        /// <summary>
        /// Logistic sigmoid, evaluated so that Math.Exp is only ever called with a non-positive argument.
        /// </summary>
        public static double Sigmoid(double x)
        {
            if (x >= 0)
                return 1.0 / (1.0 + Math.Exp(-x));

            double e = Math.Exp(x);
            return e / (1.0 + e);
        }

        /// <summary>
        /// SoftPlus ln(1 + e^x), stable for large positive and negative x.
        /// </summary>
        public static double SoftPlus(double x)
            => Math.Max(x, 0.0) + Math.Log(1.0 + Math.Exp(-Math.Abs(x)));

        /// <summary>
        /// Computes log(sum(exp(values[i] * scale))) without overflow.
        /// </summary>
        public static double LogSumExp(double[] values, double scale)
        {
            double max = double.NegativeInfinity;
            for (int i = 0; i < values.Length; i++)
                max = Math.Max(max, values[i] * scale);

            double sum = 0.0;
            for (int i = 0; i < values.Length; i++)
                sum += Math.Exp(values[i] * scale - max);

            return max + Math.Log(sum);
        }

        /// <summary>
        /// Dot product of two equal-length spans, vectorized where the hardware supports it.
        /// </summary>
        public static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
        {
            int i = 0;
            double sum = 0.0;

            if (Vector.IsHardwareAccelerated && a.Length >= Vector<double>.Count)
            {
                var acc = Vector<double>.Zero;
                for (; i <= a.Length - Vector<double>.Count; i += Vector<double>.Count)
                    acc += new Vector<double>(a.Slice(i)) * new Vector<double>(b.Slice(i));
                sum = Vector.Sum(acc);
            }

            for (; i < a.Length; i++)
                sum += a[i] * b[i];

            return sum;
        }

        /// <summary>
        /// y *= alpha, vectorized where the hardware supports it.
        /// </summary>
        public static void Scale(double alpha, Span<double> y)
        {
            int i = 0;

            if (Vector.IsHardwareAccelerated && y.Length >= Vector<double>.Count)
            {
                var va = new Vector<double>(alpha);
                for (; i <= y.Length - Vector<double>.Count; i += Vector<double>.Count)
                    (new Vector<double>(y.Slice(i)) * va).CopyTo(y.Slice(i));
            }

            for (; i < y.Length; i++)
                y[i] *= alpha;
        }

        /// <summary>
        /// y += alpha * x, vectorized where the hardware supports it.
        /// </summary>
        public static void Axpy(double alpha, ReadOnlySpan<double> x, Span<double> y)
        {
            int i = 0;

            if (Vector.IsHardwareAccelerated && x.Length >= Vector<double>.Count)
            {
                var va = new Vector<double>(alpha);
                for (; i <= x.Length - Vector<double>.Count; i += Vector<double>.Count)
                    (new Vector<double>(y.Slice(i)) + va * new Vector<double>(x.Slice(i))).CopyTo(y.Slice(i));
            }

            for (; i < x.Length; i++)
                y[i] += alpha * x[i];
        }
    }
}
