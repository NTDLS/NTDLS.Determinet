using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the sigmoid activation function, commonly used in neural networks.
    /// </summary>
    /// <remarks>The sigmoid function maps input values to a range between 0 and 1, making it useful for 
    /// applications such as binary classification. This class also provides the derivative of  the sigmoid function,
    /// which is often used during backpropagation in training neural networks.</remarks>
    public class DniSigmoidFunction : IDniActivationFunction
    {
        public DniSigmoidFunction(DniNamedParameterCollection param)
        {
        }

        public double[] Activation(double[] nodes)
        {
            return nodes.Select(o => 1.0 / (1.0 + Math.Exp(-o))).ToArray();
        }

        public double Derivative(double x)
        {
            return x * (1 - x);
        }
    }
}
