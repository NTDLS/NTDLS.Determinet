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
        public double Alpha { get; private set; }
        public double Lambda { get; private set; }

        public DniSELUFunction(DniNamedParameterCollection param)
        {
            Alpha = param.Get<double>(SELU.Alpha);
            Lambda = param.Get<double>(SELU.Lambda);
        }

        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => Lambda * (x > 0 ? x : Alpha * (Math.Exp(x) - 1))).ToArray();
        }

        public double Derivative(double x)
        {
            return Lambda * (x > 0 ? 1 : Alpha * Math.Exp(x));
        }
    }
}
