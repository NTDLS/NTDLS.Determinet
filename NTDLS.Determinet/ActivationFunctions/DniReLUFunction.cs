using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Rectified Linear Unit: max(0, x).
    /// </summary>
    public class DniReLUFunction : IDniActivationFunction
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DniReLUFunction"/> class.
        /// </summary>
        public DniReLUFunction(DniNamedParameterCollection param)
        {
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                result[i] = x > 0 ? x : 0.0;
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
            => x > 0 ? 1.0 : 0.0;
    }
}
