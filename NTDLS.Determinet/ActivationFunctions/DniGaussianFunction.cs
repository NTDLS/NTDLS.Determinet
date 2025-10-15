using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the Gaussian (RBF) activation function: f(x) = exp(-x²)
    /// </summary>
    /// <remarks>
    /// Used in radial basis function networks and kernel approximations.
    /// </remarks>
    public class DniGaussianFunction : IDniActivationFunction
    {
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = false;

        /// <summary>
        /// Default constructor for Gaussian activation function.
        /// </summary>
        public DniGaussianFunction(DniNamedParameterCollection param)
        {
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => Math.Exp(-x * x)).ToArray();
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        /// 
        public double Derivative(double x)
        {
            return -2 * x * Math.Exp(-x * x);
        }
    }
}
