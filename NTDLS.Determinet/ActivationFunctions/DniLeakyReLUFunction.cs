using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the Leaky ReLU (Rectified Linear Unit) activation function, which introduces a small slope for
    /// negative input values.
    /// </summary>
    /// <remarks>The Leaky ReLU activation function is commonly used in neural networks to address the "dying
    /// ReLU" problem by allowing a small, non-zero gradient for negative input values. The slope for negative values is
    /// determined by the <see cref="Alpha"/> parameter.</remarks>
    public class DniLeakyReLUFunction : IDniActivationFunction
    {
        public double Alpha { get; private set; }

        public DniLeakyReLUFunction(DniNamedFunctionParameters param)
        {
            Alpha = param.Get(LeakyReLU.Alpha, 0.01);
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
