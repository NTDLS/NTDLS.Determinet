using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Logistic sigmoid: 1 / (1 + e^-x).
    /// </summary>
    public class DniSigmoidFunction : IDniActivationFunction
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DniSigmoidFunction"/> class.
        /// </summary>
        public DniSigmoidFunction(DniNamedParameterCollection param)
        {
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                result[i] = DniMath.Sigmoid(x);
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
        {
            double s = DniMath.Sigmoid(x);
            return s * (1.0 - s);
        }
    }
}
