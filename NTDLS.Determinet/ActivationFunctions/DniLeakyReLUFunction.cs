using NTDLS.Determinet.ActivationFunctions.Interfaces;

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
        public double[] Activation(double[] nodes)
        {
            return nodes.Select(o=> (0 >= o) ? 0.01f * o : o).ToArray();
        }

        public double Derivative(double x)
        {
            return (0 >= x) ? 0.01f : 1;
        }
    }
}
