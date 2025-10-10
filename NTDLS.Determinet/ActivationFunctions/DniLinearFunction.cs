using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using ProtoBuf;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Linear bounded activation function.
    /// </summary>
    [ProtoContract]
    public class DniLinearFunction : IDniActivationFunction
    {
        [ProtoMember(1)]
        public double Alpha { get; private set; } = 1; //Linear slope value.

        [ProtoMember(2)]
        public DniRange Range { get; private set; } = new DniRange(-1, +1); //Function output range.

        public DniLinearFunction(DniNamedFunctionParameters param)
        {
            Alpha = param.Get<double>("Alpha", 1);
            Range = param.Get("Range", new DniRange(-1, +1));
        }

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
