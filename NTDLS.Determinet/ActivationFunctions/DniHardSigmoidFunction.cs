using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Piecewise-linear approximation of sigmoid: clamp(0.2 * x + 0.5, 0, 1).
    /// </summary>
    public class DniHardSigmoidFunction : IDniActivationFunction
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DniHardSigmoidFunction"/> class.
        /// </summary>
        public DniHardSigmoidFunction(DniNamedParameterCollection param)
        {
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                result[i] = Math.Clamp(0.2 * x + 0.5, 0.0, 1.0);
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
            => (x > -2.5 && x < 2.5) ? 0.2 : 0.0;
    }
}
