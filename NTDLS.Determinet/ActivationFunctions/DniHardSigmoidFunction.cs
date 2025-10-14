using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents a hard (linear) approximation of the Sigmoid activation function.
    /// </summary>
    /// <remarks>
    /// f(x) = clamp(0.2 * x + 0.5, 0, 1)
    /// Faster than full Sigmoid, useful for constrained inference or embedded models.
    /// </remarks>
    public class DniHardSigmoidFunction : IDniActivationFunction
    {
        public DniHardSigmoidFunction(DniNamedParameterCollection param) { }

        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => Math.Min(1.0, Math.Max(0.0, 0.2 * x + 0.5))).ToArray();
        }

        public double Derivative(double x)
        {
            return (x > -2.5 && x < 2.5) ? 0.2 : 0.0;
        }
    }
}
