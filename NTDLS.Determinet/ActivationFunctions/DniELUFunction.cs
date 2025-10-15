using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the Exponential Linear Unit (ELU) activation function.
    /// </summary>
    /// <remarks>
    /// ELU: f(x) = x if x >= 0 else α * (exp(x) - 1)
    /// Default α = 1.0
    /// </remarks>
    public class DniELUFunction : IDniActivationFunction
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
        /// Default constructor for ELU activation function.
        /// </summary>
        /// <param name="param"></param>
        public DniELUFunction(DniNamedParameterCollection param)
        {
            Alpha = param.Get<double>(ELU.Alpha);
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => x >= 0 ? x : Alpha * (Math.Exp(x) - 1.0)).ToArray();
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            return x >= 0 ? 1.0 : Alpha * Math.Exp(x);
        }
    }
}
