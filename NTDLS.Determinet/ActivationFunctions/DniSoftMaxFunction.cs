using NTDLS.Determinet.ActivationFunctions.Interfaces;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// The sigmoid activation function, also called the logistic function, is traditionally a very popular activation function for neural networks.
    /// The input to the function is transformed into a value between 0.0 and 1.0. Inputs that are much larger than 1.0 are transformed to the value 1.0,
    /// similarly, values much smaller than 0.0 are snapped to 0.0. The shape of the function for all possible inputs is an S-shape from zero up
    /// through 0.5 to 1.0. For a long time, through the early 1990s, it was the default activation used on neural networks.
    /// </summary>
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
