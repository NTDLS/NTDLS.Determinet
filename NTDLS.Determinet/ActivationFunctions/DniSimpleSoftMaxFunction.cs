using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the SoftMax activation function, commonly used in neural networks to convert raw output values into
    /// probabilities.
    /// </summary>
    /// <remarks>The SoftMax function normalizes an input array of values into a probability distribution,
    /// where each value is scaled exponentially and divided by the sum of all exponentials. This ensures that the
    /// output values are non-negative and sum to 1. The function is numerically stable, handling cases where input
    /// values are large or small to avoid overflow or underflow.</remarks>
    public class DniSimpleSoftMaxFunction : IDniActivationFunction
    {
        public DniSimpleSoftMaxFunction(DniNamedFunctionParameters param)
        {
        }

        public double[] Activation(double[] nodes)
        {
            double max = nodes.Max();

            // subtract max for stability
            double[] exps = nodes.Select(v =>
            {
                double e = Math.Exp(v - max);
                if (double.IsNaN(e) || double.IsInfinity(e))
                    e = 0.0;
                return e;
            }).ToArray();

            double sum = exps.Sum();

            // avoid divide-by-zero
            if (sum == 0 || double.IsNaN(sum) || double.IsInfinity(sum))
                sum = 1e-12;

            for (int i = 0; i < exps.Length; i++)
                exps[i] /= sum;

            return exps;
        }

        public double Derivative(double x)
        {
            // Not used directly; handled by cross-entropy gradient
            return 1.0;
        }
    }
}
