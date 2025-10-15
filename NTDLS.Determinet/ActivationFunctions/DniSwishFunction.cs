using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the Swish activation function: f(x) = x * sigmoid(x)
    /// </summary>
    /// <remarks>
    /// Swish is smooth and non-monotonic, improving gradient flow and generalization
    /// in many deep learning architectures compared to ReLU.
    /// </remarks>
    public class DniSwishFunction : IDniActivationFunction
    {
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = false;

        /// <summary>
        /// Default constructor for Swish activation function.
        /// </summary>
        public DniSwishFunction(DniNamedParameterCollection param)
        {
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => x / (1.0 + Math.Exp(-x))).ToArray();
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            double sigmoid = 1.0 / (1.0 + Math.Exp(-x));
            return sigmoid + x * sigmoid * (1 - sigmoid);
        }
    }
}
