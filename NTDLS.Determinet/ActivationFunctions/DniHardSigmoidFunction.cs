using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents a hard (linear) approximation of the Sigmoid activation function.
    /// </summary>
    /// <remarks>
    /// f(x) = clamp(0.2 * x + 0.5, 0, 1)
    /// Faster than full Sigmoid, useful for constrained inference or embedded models.
    /// </remarks>
    public class DniHardSigmoidFunction : IDniActivationFunction
    {
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = false;

        /// <summary>
        /// Default constructor for Hard Sigmoid activation function.
        /// </summary>
        public DniHardSigmoidFunction(DniNamedParameterCollection param)
        {
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => Math.Min(1.0, Math.Max(0.0, 0.2 * x + 0.5))).ToArray();
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            return (x > -2.5 && x < 2.5) ? 0.2 : 0.0;
        }
    }
}
