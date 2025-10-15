using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the Rectified Linear Unit (ReLU) activation function, commonly used in neural networks.
    /// </summary>
    /// <remarks>The ReLU function outputs the input value if it is greater than zero; otherwise, it outputs
    /// zero. It is widely used in deep learning due to its simplicity and effectiveness in introducing
    /// non-linearity.</remarks>
    public class DniReLUFunction : IDniActivationFunction
    {
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = false;

        /// <summary>
        /// Default constructor for ReLU activation function.
        /// </summary>
        public DniReLUFunction(DniNamedParameterCollection param)
        {
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            return nodes.Select(o => o > 0 ? o : 0).ToArray();
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            if (x > 0)
            {
                return 1;
            }
            return 0;
        }
    }
}
