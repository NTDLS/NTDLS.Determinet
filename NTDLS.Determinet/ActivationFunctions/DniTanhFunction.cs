using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Hyperbolic tangent.
    /// </summary>
    public class DniTanhFunction : IDniActivationFunction
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DniTanhFunction"/> class.
        /// </summary>
        public DniTanhFunction(DniNamedParameterCollection param)
        {
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                result[i] = Math.Tanh(x);
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
        {
            double t = Math.Tanh(x);
            return 1.0 - t * t;
        }
    }
}
