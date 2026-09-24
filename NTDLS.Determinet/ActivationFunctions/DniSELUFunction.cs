using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Scaled Exponential Linear Unit: Lambda * (x for x &gt; 0, otherwise Alpha * (e^x - 1)).
    /// </summary>
    public class DniSELUFunction : IDniActivationFunction
    {
        /// <summary>
        /// Scale of the negative saturation region.
        /// </summary>
        public double Alpha { get; private set; }

        /// <summary>
        /// Output scale.
        /// </summary>
        public double Lambda { get; private set; }


        /// <summary>
        /// Initializes a new instance of the <see cref="DniSELUFunction"/> class.
        /// </summary>
        public DniSELUFunction(DniNamedParameterCollection param)
        {
            Alpha = param.Get<double>(SELU.Alpha);
            Lambda = param.Get<double>(SELU.Lambda);
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                result[i] = Lambda * (x > 0 ? x : Alpha * (Math.Exp(x) - 1.0));
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
            => Lambda * (x > 0 ? 1.0 : Alpha * Math.Exp(x));
    }
}
