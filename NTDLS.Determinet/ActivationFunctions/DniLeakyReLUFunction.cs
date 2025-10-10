using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using ProtoBuf;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Both ReLU and Leaky ReLU are activation functions used in neural networks. The main difference between them is that ReLU sets all negative
    /// values to zero while Leaky ReLU allows a small, non-zero gradient for negative input values. This helps to avoid discarding potentially
    /// important information and thus perform better than ReLU in scenarios where the data has a lot of noise or outliers1. ReLU is computationally
    /// efficient and simpler than Leaky ReLU, which makes it more suitable for shallow architectures.
    /// </summary>
    public class DniLeakyReLUFunction : IDniActivationFunction
    {
        [ProtoMember(1)]
        public double Alpha { get; private set; } = 0.01; //Linear slope value.

        public DniLeakyReLUFunction()
        {
        }

        public DniLeakyReLUFunction(DniNamedFunctionParameters param)
        {
            Alpha = param.Get("Alpha", 0.01);
        }

        public double[] Activation(double[] nodes)
        {
            var result = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                if (double.IsNaN(x) || double.IsInfinity(x))
                    x = 0;
                result[i] = x <= 0 ? Alpha * x : x;
            }
            return result;
        }

        public double Derivative(double x)
        {
            if (double.IsNaN(x) || double.IsInfinity(x))
                return 0;
            return x <= 0 ? Alpha : 1.0;
        }
    }
}
