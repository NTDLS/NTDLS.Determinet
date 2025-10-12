using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents an identity activation function, which returns the input as the output without modification.
    /// </summary>
    public class DniIdentityFunction : IDniActivationFunction
    {
        public DniIdentityFunction(DniNamedFunctionParameters param)
        {
        }

        public double[] Activation(double[] nodes) => nodes;
        public double Derivative(double x) => 1;
    }
}
