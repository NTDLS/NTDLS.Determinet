using Newtonsoft.Json;
using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Function that combines a linear segment for certain input range with a Leaky ReLU-like behavior for values outside that range. 
    /// </summary>
    public class DniPiecewiseLinearFunction : IDniActivationFunction
    {
        /// <summary>
        /// Linear slope value.
        /// </summary>
        [JsonProperty]
        public double Alpha { get; set; }

        /// <summary>
        /// Function output range.
        /// </summary>
        [JsonProperty]
        public DniRange Range { get; set; }

        public DniPiecewiseLinearFunction(DniNamedFunctionParameters? param)
        {
            if (param == null)
            {
                Alpha = 0.1;
                Range = new DniRange(-1, +1);
            }
            else
            {
                Alpha = param.Get<double>("alpha", 1);
                Range = param.Get<DniRange>("range", new DniRange(-1, +1));
            }
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