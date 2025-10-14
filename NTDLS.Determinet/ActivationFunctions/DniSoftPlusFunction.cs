using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the Softplus activation function: f(x) = ln(1 + exp(x))
    /// </summary>
    /// <remarks>
    /// Softplus is a smooth approximation of ReLU, providing continuous gradients everywhere.
    /// </remarks>
    public class DniSoftPlusFunction : IDniActivationFunction
    {
        public DniSoftPlusFunction(DniNamedParameterCollection param) { }

        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => Math.Log(1.0 + Math.Exp(x))).ToArray();
        }

        public double Derivative(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x)); // sigmoid(x)
        }
    }
}
