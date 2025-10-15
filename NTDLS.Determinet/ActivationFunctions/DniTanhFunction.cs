using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the hyperbolic tangent (tanh) activation function, commonly used in neural networks.
    /// </summary>
    /// <remarks>This class provides methods to compute the activation and derivative of the tanh function.
    /// The tanh function maps input values to the range [-1, 1], making it useful for normalizing data in neural
    /// network layers. The derivative is used during backpropagation to compute gradients.</remarks>
    public class DniTanhFunction : IDniActivationFunction
    {
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = false;

        /// <summary>
        /// Default constructor for the Tanh activation function.
        /// </summary>
        public DniTanhFunction(DniNamedParameterCollection param)
        {
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            return nodes.Select(o => (double)Math.Tanh(o)).ToArray();
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            return 1 - (x * x);
        }
    }
}
