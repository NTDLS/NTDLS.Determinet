using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents a piecewise linear activation function with configurable slope and output range.
    /// Combines a linear segment for certain input range with a Leaky ReLU-like behavior for values outside that range. 
    /// </summary>
    /// <remarks>This activation function applies a linear transformation to input values based on the
    /// specified range: - For inputs less than or equal to the minimum of the range, the output is scaled by the slope
    /// value. - For inputs greater than or equal to the maximum of the range, the output is also scaled by the slope
    /// value. - For inputs within the range, the output is equal to the input.  The derivative of the function is
    /// constant outside the range (equal to the slope) and 1 within the range.</remarks>
    public class DniPiecewiseLinearFunction : IDniActivationFunction
    {
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = false;

        /// <summary>
        /// Gets the linear slope value used in calculations.
        /// </summary>
        public double Alpha { get; private set; } //Linear slope value.

        /// <summary>
        /// Gets the range of valid DNI values for the operation.
        /// </summary>
        public DniRange Range { get; private set; } //Function output range.

        /// <summary>
        /// Default constructor for Piecewise Linear activation function.
        /// </summary>
        public DniPiecewiseLinearFunction(DniNamedParameterCollection param)
        {
            Alpha = param.Get<double>(Piecewise.Alpha);
            Range = param.Get<DniRange>(Piecewise.Range);
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            var result = new List<double>();

            foreach (var node in nodes)
            {
                if (node <= Range.Min)
                    result.Add(Alpha * node);
                else if (node >= Range.Max)
                    result.Add(Alpha * node);
                else
                    result.Add(node);
            }

            return result.ToArray();
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            if (x <= Range.Min)
                return Alpha;
            else if (x >= Range.Max)
                return Alpha;
            else
                return 1;
        }
    }
}