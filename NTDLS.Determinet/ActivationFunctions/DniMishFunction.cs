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
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = false;

        /// <summary>
        /// Default constructor for Mish activation function.
        /// </summary>
        public DniMishFunction(DniNamedParameterCollection param)
        {
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => x * Math.Tanh(Math.Log(1 + Math.Exp(x)))).ToArray();
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            double sp = Math.Log(1 + Math.Exp(x)); // softplus
            double sech2 = 1 / Math.Cosh(sp); sech2 *= sech2;
            double sigmoid = 1 / (1 + Math.Exp(-x));
            return Math.Tanh(sp) + x * sigmoid * sech2;
        }
    }
}
