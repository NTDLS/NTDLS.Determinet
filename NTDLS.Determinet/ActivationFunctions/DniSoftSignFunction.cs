using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// SoftSign: x / (1 + |x|).
    /// </summary>
    public class DniSoftSignFunction : IDniActivationFunction
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DniSoftSignFunction"/> class.
        /// </summary>
        public DniSoftSignFunction(DniNamedParameterCollection param)
        {
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                result[i] = x / (1.0 + Math.Abs(x));
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
        {
            double denom = 1.0 + Math.Abs(x);
            return 1.0 / (denom * denom);
        }
    }
}
