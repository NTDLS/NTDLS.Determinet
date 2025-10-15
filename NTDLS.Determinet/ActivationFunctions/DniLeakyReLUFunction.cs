using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the Leaky ReLU (Rectified Linear Unit) activation function, which introduces a small slope for
    /// negative input values.
    /// </summary>
    /// <remarks>The Leaky ReLU activation function is commonly used in neural networks to address the "dying
    /// ReLU" problem by allowing a small, non-zero gradient for negative input values. The slope for negative values is
    /// determined by the <see cref="Alpha"/> parameter.</remarks>
    public class DniLeakyReLUFunction : IDniActivationFunction
    {
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = false;

        /// <summary>
        /// Gets the alpha value, which represents a coefficient or parameter used in calculations.
        /// </summary>
        public double Alpha { get; private set; }

        /// <summary>
        /// Default constructor for Leaky ReLU activation function.
        /// </summary>
        public DniLeakyReLUFunction(DniNamedParameterCollection param)
        {
            Alpha = param.Get<double>(LeakyReLU.Alpha);
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                if (double.IsNaN(x) || double.IsInfinity(x))
                    x = 0;
                result[i] = x <= 0 ? Alpha * x : x;
            }
            return result;
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            if (double.IsNaN(x) || double.IsInfinity(x))
                return 0;
            return x <= 0 ? Alpha : 1.0;
        }
    }
}
