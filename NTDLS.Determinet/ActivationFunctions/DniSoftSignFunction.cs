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
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = false;

        /// <summary>
        /// Default constructor for Softsign activation function.
        /// </summary>
        public DniSoftSignFunction(DniNamedParameterCollection param)
        {
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => x / (1.0 + Math.Abs(x))).ToArray();
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            double denom = 1.0 + Math.Abs(x);
            return 1.0 / (denom * denom);
        }
    }
}
