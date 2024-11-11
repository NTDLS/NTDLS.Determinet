using Newtonsoft.Json;
using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Linear bounded activation function.
    /// </summary>
    public class DniLinearFunction : IDniActivationFunction
    {
        // linear slope value
        private double _alpha;

        // function output range
        private DniRange _range;

        [JsonProperty]
        public double Alpha //Linear slope value.
        {
            get { return _alpha; }
        }

        [JsonProperty]
        public DniRange Range //Function output range.
        {
            get { return _range; }
        }

        public DniLinearFunction(DniNamedFunctionParameters? param)
        {
            if (param == null) throw new ArgumentNullException(nameof(param));

            _alpha = param.Get<double>("Alpha", 1);
            _range = param.Get("Range", new DniRange(-1, +1));
        }

        public double[] Activation(double[] nodes)
        {
            var result = new List<double>();

            foreach (var node in nodes)
            {
                double y = _alpha * node;

                if (y > _range.Max)
                {
                    result.Add(_range.Max);
                }
                else if (y < _range.Min)
                {
                    result.Add(_range.Min);
                }
                else
                {
                    result.Add(y);
                }
            }

            return result.ToArray();
        }

        public double Derivative(double x)
        {
            double y = _alpha * x;

            if (y <= _range.Min || y >= _range.Max)
            {
                return 0;
            }
            return _alpha;
        }
    }
}
