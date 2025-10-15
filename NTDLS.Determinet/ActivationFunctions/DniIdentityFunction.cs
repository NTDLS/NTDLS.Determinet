using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents an identity activation function, which returns the input as the output without modification.
    /// </summary>
    public class DniIdentityFunction : IDniActivationFunction
    {
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = false;

        /// <summary>
        /// Default constructor for the identity activation function.
        /// </summary>
        public DniIdentityFunction(DniNamedParameterCollection param)
        {
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes) => nodes;

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x) => 1;
    }
}
