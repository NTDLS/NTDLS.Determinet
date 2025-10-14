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
        public double Alpha { get; private set; }

        public DniELUFunction(DniNamedParameterCollection param)
        {
            Alpha = param.Get<double>(ELU.Alpha);
        }

        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => x >= 0 ? x : Alpha * (Math.Exp(x) - 1.0)).ToArray();
        }

        public double Derivative(double x)
        {
            return x >= 0 ? 1.0 : Alpha * Math.Exp(x);
        }
    }
}
