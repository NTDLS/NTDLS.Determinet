using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Gaussian (radial basis) activation: e^(-x^2).
    /// </summary>
    public class DniGaussianFunction : IDniActivationFunction
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DniGaussianFunction"/> class.
        /// </summary>
        public DniGaussianFunction(DniNamedParameterCollection param)
        {
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                result[i] = Math.Exp(-x * x);
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
            => -2.0 * x * Math.Exp(-x * x);
    }
}
