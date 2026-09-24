using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Identity activation: passes values through unchanged.
    /// </summary>
    public class DniIdentityFunction : IDniActivationFunction
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DniIdentityFunction"/> class.
        /// </summary>
        public DniIdentityFunction(DniNamedParameterCollection param)
        {
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                result[i] = x;
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
            => 1.0;
    }
}
