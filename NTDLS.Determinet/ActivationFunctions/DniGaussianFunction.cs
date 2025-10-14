using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents the Gaussian (RBF) activation function: f(x) = exp(-x²)
    /// </summary>
    /// <remarks>
    /// Used in radial basis function networks and kernel approximations.
    /// </remarks>
    public class DniGaussianFunction : IDniActivationFunction
    {
        public DniGaussianFunction(DniNamedParameterCollection param) { }

        public double[] Activation(double[] nodes)
        {
            return nodes.Select(x => Math.Exp(-x * x)).ToArray();
        }

        public double Derivative(double x)
        {
            return -2 * x * Math.Exp(-x * x);
        }
    }
}
