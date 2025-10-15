using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents a linear activation function with configurable slope and output range.
    /// </summary>
    /// <remarks>This activation function applies a linear transformation to the input values, scaling them by
    /// the  <see cref="Alpha"/> parameter. The output is clamped to the specified <see cref="Range"/> to ensure  it
    /// remains within the defined bounds. This function is commonly used in neural network models where  a simple
    /// linear transformation is required.</remarks>
    public class DniLinearFunction : IDniActivationFunction
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
        /// Default constructor for Linear activation function.
        /// </summary>
        public DniLinearFunction(DniNamedParameterCollection param)
        {
            Alpha = param.Get<double>(Linear.Alpha);
            Range = param.Get<DniRange>(Linear.Range);
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            var result = new List<double>();

            foreach (var node in nodes)
            {
                double y = Alpha * node;

                if (y > Range.Max)
                {
                    result.Add(Range.Max);
                }
                else if (y < Range.Min)
                {
                    result.Add(Range.Min);
                }
                else
                {
                    result.Add(y);
                }
            }

            return result.ToArray();
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            double y = Alpha * x;

            if (y <= Range.Min || y >= Range.Max)
            {
                return 0;
            }
            return Alpha;
        }
    }
}
