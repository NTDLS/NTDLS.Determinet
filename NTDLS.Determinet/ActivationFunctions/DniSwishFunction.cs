using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Swish (SiLU): x * sigmoid(x).
    /// </summary>
    public class DniSwishFunction : IDniActivationFunction
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DniSwishFunction"/> class.
        /// </summary>
        public DniSwishFunction(DniNamedParameterCollection param)
        {
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                result[i] = x * DniMath.Sigmoid(x);
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
        {
            double s = DniMath.Sigmoid(x);
            return s + x * s * (1.0 - s);
        }
    }
}
