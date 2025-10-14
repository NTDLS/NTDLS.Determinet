using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the Swish activation function: f(x) = x * sigmoid(x)
    /// </summary>
    /// <remarks>
    /// Swish is smooth and non-monotonic, improving gradient flow and generalization
    /// in many deep learning architectures compared to ReLU.
    /// </remarks>
    public class DniSwishFunction : IDniActivationFunction
    {
        public DniSwishFunction(DniNamedParameterCollection param) { }

        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => x / (1.0 + Math.Exp(-x))).ToArray();
        }

        public double Derivative(double x)
        {
            double sigmoid = 1.0 / (1.0 + Math.Exp(-x));
            return sigmoid + x * sigmoid * (1 - sigmoid);
        }
    }
}
