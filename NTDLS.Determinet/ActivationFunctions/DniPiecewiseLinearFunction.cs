using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

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
        public double Alpha { get; private set; } = 0.1; //Linear slope value.
        public DniRange Range { get; private set; } = new DniRange(-1, +1); //Function output range.

        public DniPiecewiseLinearFunction(DniNamedFunctionParameters param)
        {
            Alpha = param.Get<double>("alpha", 0.1);
            Range = param.Get<DniRange>("range", new DniRange(-1, +1));
        }

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