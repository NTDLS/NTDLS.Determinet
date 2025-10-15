using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the sigmoid activation function, commonly used in neural networks.
    /// </summary>
    /// <remarks>The sigmoid function maps input values to a range between 0 and 1, making it useful for 
    /// applications such as binary classification. This class also provides the derivative of  the sigmoid function,
    /// which is often used during backpropagation in training neural networks.</remarks>
    public class DniSigmoidFunction : IDniActivationFunction
    {
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = false;

        /// <summary>
        /// Default constructor for the Sigmoid activation function.
        /// </summary>
        public DniSigmoidFunction(DniNamedParameterCollection param)
        {
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            return nodes.Select(o => 1.0 / (1.0 + Math.Exp(-o))).ToArray();
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            return x * (1 - x);
        }
    }
}
