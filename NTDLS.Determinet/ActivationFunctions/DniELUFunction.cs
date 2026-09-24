using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Exponential Linear Unit: x for x &gt;= 0, otherwise Alpha * (e^x - 1).
    /// </summary>
    public class DniELUFunction : IDniActivationFunction
    {
        /// <summary>
        /// Scale of the negative saturation region.
        /// </summary>
        public double Alpha { get; private set; }


        /// <summary>
        /// Initializes a new instance of the <see cref="DniELUFunction"/> class.
        /// </summary>
        public DniELUFunction(DniNamedParameterCollection param)
        {
            Alpha = param.Get<double>(ELU.Alpha);
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                result[i] = x >= 0 ? x : Alpha * (Math.Exp(x) - 1.0);
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
            => x >= 0 ? 1.0 : Alpha * Math.Exp(x);
    }
}
