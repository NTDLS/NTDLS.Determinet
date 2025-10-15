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
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = false;

        /// <summary>
        /// Default constructor for Softplus activation function.
        /// </summary>
        public DniSoftPlusFunction(DniNamedParameterCollection param)
        {
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => Math.Log(1.0 + Math.Exp(x))).ToArray();
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x)); // sigmoid(x)
        }
    }
}
