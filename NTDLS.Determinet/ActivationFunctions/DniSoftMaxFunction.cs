using NTDLS.Determinet.ActivationFunctions.Interfaces;

namespace NTDLS.Determinet.ActivationFunctions
{

    /// <summary>
    /// Represents the SoftMax activation function, commonly used in neural networks to convert raw output values into probabilities.
    /// </summary>
    /// <remarks>The SoftMax function normalizes an array of input values such that the output values
    /// represent probabilities, summing to 1. This is achieved by exponentiating the inputs (after subtracting the
    /// maximum value for numerical stability) and dividing each exponentiated value by the sum of all exponentiated
    /// values.</remarks>
    public class DniSoftMaxFunction : IDniActivationFunction
    {
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
