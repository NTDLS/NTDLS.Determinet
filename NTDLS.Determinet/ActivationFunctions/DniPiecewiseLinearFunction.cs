using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Continuous piecewise-linear activation: slope 1 inside [Range.Min, Range.Max] and slope Alpha outside it.
    /// </summary>
    /// <remarks>
    /// f(x) = x                              when Range.Min &lt; x &lt; Range.Max
    /// f(x) = Range.Max + Alpha * (x - Range.Max) when x &gt;= Range.Max
    /// f(x) = Range.Min + Alpha * (x - Range.Min) when x &lt;= Range.Min
    /// </remarks>
    public class DniPiecewiseLinearFunction : IDniActivationFunction
    {
        /// <summary>
        /// Slope used outside of <see cref="Range"/>.
        /// </summary>
        public double Alpha { get; private set; }

        /// <summary>
        /// Input range within which the function has slope 1.
        /// </summary>
        public DniRange Range { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="DniPiecewiseLinearFunction"/> class.
        /// </summary>
        public DniPiecewiseLinearFunction(DniNamedParameterCollection param)
        {
            Alpha = param.Get<double>(Piecewise.Alpha);
            Range = param.Get<DniRange>(Piecewise.Range);
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                if (x >= Range.Max)
                    result[i] = Range.Max + Alpha * (x - Range.Max);
                else if (x <= Range.Min)
                    result[i] = Range.Min + Alpha * (x - Range.Min);
                else
                    result[i] = x;
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
            => (x >= Range.Max || x <= Range.Min) ? Alpha : 1.0;
    }
}
