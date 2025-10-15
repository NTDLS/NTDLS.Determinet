using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the Scaled Exponential Linear Unit (SELU) activation function.
    /// </summary>
    /// <remarks>
    /// SELU: f(x) = λ * (x if x > 0 else α*(exp(x)-1))
    /// Default α = 1.67326, λ = 1.0507
    /// </remarks>
    public class DniSELUFunction : IDniActivationFunction
    {
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = false;

        /// <summary>
        /// Gets the alpha value used in calculations or operations.
        /// </summary>
        public double Alpha { get; private set; }

        /// <summary>
        /// Gets the value of the lambda parameter used in calculations.
        /// </summary>
        public double Lambda { get; private set; }

        /// <summary>
        /// Default constructor for SELU activation function.
        /// </summary>
        public DniSELUFunction(DniNamedParameterCollection param)
        {
            Alpha = param.Get<double>(SELU.Alpha);
            Lambda = param.Get<double>(SELU.Lambda);
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => Lambda * (x > 0 ? x : Alpha * (Math.Exp(x) - 1))).ToArray();
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            return Lambda * (x > 0 ? 1 : Alpha * Math.Exp(x));
        }
    }
}
