using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// SoftPlus: ln(1 + e^x), a smooth approximation of ReLU.
    /// </summary>
    public class DniSoftPlusFunction : IDniActivationFunction
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DniSoftPlusFunction"/> class.
        /// </summary>
        public DniSoftPlusFunction(DniNamedParameterCollection param)
        {
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                result[i] = DniMath.SoftPlus(x);
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
            => DniMath.Sigmoid(x);
    }
}
