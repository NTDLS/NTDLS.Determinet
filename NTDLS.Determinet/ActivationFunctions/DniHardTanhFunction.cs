using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the HardTanh activation function.
    /// </summary>
    /// <remarks>
    /// f(x) = clamp(x, -1, 1)
    /// </remarks>
    public class DniHardTanhFunction : IDniActivationFunction
    {
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = false;

        /// <summary>
        /// Default constructor for HardTanh activation function.
        /// </summary>
        public DniHardTanhFunction(DniNamedParameterCollection param)
        {
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => Math.Max(-1.0, Math.Min(1.0, x))).ToArray();
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            return (x > -1.0 && x < 1.0) ? 1.0 : 0.0;
        }
    }
}
