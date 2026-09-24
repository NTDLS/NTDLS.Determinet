using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Scaled linear activation clamped to an output range: clamp(Alpha * x, Range.Min, Range.Max).
    /// </summary>
    public class DniLinearFunction : IDniActivationFunction
    {
        /// <summary>
        /// Linear slope value.
        /// </summary>
        public double Alpha { get; private set; }

        /// <summary>
        /// Function output range.
        /// </summary>
        public DniRange Range { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="DniLinearFunction"/> class.
        /// </summary>
        public DniLinearFunction(DniNamedParameterCollection param)
        {
            Alpha = param.Get<double>(Linear.Alpha);
            Range = param.Get<DniRange>(Linear.Range);
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                result[i] = Math.Clamp(Alpha * nodes[i], Range.Min, Range.Max);
            }
            return result;
        }

        /// <inheritdoc/>
        public double Derivative(double x)
        {
            double y = Alpha * x;
            return (y > Range.Min && y < Range.Max) ? Alpha : 0.0;
        }
    }
}
