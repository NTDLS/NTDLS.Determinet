using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the Softsign activation function: f(x) = x / (1 + |x|)
    /// </summary>
    /// <remarks>
    /// Similar to tanh but simpler and numerically stable.
    /// </remarks>
    public class DniSoftSignFunction : IDniActivationFunction
    {
        public DniSoftSignFunction(DniNamedParameterCollection param) { }

        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => x / (1.0 + Math.Abs(x))).ToArray();
        }

        public double Derivative(double x)
        {
            double denom = 1.0 + Math.Abs(x);
            return 1.0 / (denom * denom);
        }
    }
}
