using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Piecewise-linear approximation of tanh: clamp(x, -1, 1).
    /// </summary>
    public class DniHardTanhFunction : IDniActivationFunction
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DniHardTanhFunction"/> class.
        /// </summary>
        public DniHardTanhFunction(DniNamedParameterCollection param)
        {
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                result[i] = Math.Clamp(x, -1.0, 1.0);
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
            => (x > -1.0 && x < 1.0) ? 1.0 : 0.0;
    }
}
