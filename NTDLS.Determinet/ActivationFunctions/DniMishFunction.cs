using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the Mish activation function: f(x) = x * tanh(softplus(x))
    /// </summary>
    /// <remarks>
    /// Softplus(x) = ln(1 + exp(x))
    /// Mish provides smooth, non-monotonic behavior improving gradient flow.
    /// </remarks>
    public class DniMishFunction : IDniActivationFunction
    {
        public DniMishFunction(DniNamedParameterCollection param) { }

        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => x * Math.Tanh(Math.Log(1 + Math.Exp(x)))).ToArray();
        }

        public double Derivative(double x)
        {
            double sp = Math.Log(1 + Math.Exp(x)); // softplus
            double sech2 = 1 / Math.Cosh(sp); sech2 *= sech2;
            double sigmoid = 1 / (1 + Math.Exp(-x));
            return Math.Tanh(sp) + x * sigmoid * sech2;
        }
    }
}
