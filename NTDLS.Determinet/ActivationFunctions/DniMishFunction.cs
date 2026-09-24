using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Mish: x * tanh(softplus(x)).
    /// </summary>
    public class DniMishFunction : IDniActivationFunction
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DniMishFunction"/> class.
        /// </summary>
        public DniMishFunction(DniNamedParameterCollection param)
        {
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                result[i] = x * Math.Tanh(DniMath.SoftPlus(x));
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
        {
            double tsp = Math.Tanh(DniMath.SoftPlus(x));
            // d/dx = tanh(sp) + x * sech^2(sp) * sigmoid(x), with sech^2 = 1 - tanh^2.
            return tsp + x * (1.0 - tsp * tsp) * DniMath.Sigmoid(x);
        }
    }
}
