using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using ProtoBuf;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Function that combines a linear segment for certain input range with a Leaky ReLU-like behavior for values outside that range. 
    /// </summary>
    [ProtoContract]
    public class DniPiecewiseLinearFunction : IDniActivationFunction
    {
        [ProtoMember(1)]
        public double Alpha { get; private set; } = 0.1; //Linear slope value.

        [ProtoMember(2)]
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