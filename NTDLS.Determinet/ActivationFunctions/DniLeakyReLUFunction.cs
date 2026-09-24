using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Leaky ReLU: x for x &gt; 0, otherwise Alpha * x.
    /// </summary>
    public class DniLeakyReLUFunction : IDniActivationFunction
    {
        /// <summary>
        /// Slope used for non-positive inputs.
        /// </summary>
        public double Alpha { get; private set; }


        /// <summary>
        /// Initializes a new instance of the <see cref="DniLeakyReLUFunction"/> class.
        /// </summary>
        public DniLeakyReLUFunction(DniNamedParameterCollection param)
        {
            Alpha = param.Get<double>(LeakyReLU.Alpha);
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                result[i] = x > 0 ? x : Alpha * x;
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
            => x > 0 ? 1.0 : Alpha;
    }
}
