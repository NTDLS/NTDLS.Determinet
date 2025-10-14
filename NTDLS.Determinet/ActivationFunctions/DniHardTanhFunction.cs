using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the HardTanh activation function.
    /// </summary>
    /// <remarks>
    /// f(x) = clamp(x, -1, 1)
    /// </remarks>
    public class DniHardTanhFunction : IDniActivationFunction
    {
        public DniHardTanhFunction(DniNamedParameterCollection param) { }

        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => Math.Max(-1.0, Math.Min(1.0, x))).ToArray();
        }

        public double Derivative(double x)
        {
            return (x > -1.0 && x < 1.0) ? 1.0 : 0.0;
        }
    }
}
