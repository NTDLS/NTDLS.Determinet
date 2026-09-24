using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Standard SoftMax (temperature fixed at 1): e^x_i / sum(e^x_j).
    /// </summary>
    /// <remarks>
    /// SoftMax is only valid on the output layer, where it is trained with cross-entropy loss.
    /// </remarks>
    public class DniSimpleSoftMaxFunction : IDniSoftMaxFunction
    {
        /// <summary>
        /// Always 1.
        /// </summary>
        public double Temperature => 1.0;

        /// <summary>
        /// Initializes a new instance of the <see cref="DniSimpleSoftMaxFunction"/> class.
        /// </summary>
        public DniSimpleSoftMaxFunction(DniNamedParameterCollection param)
        {
        }

        /// <inheritdoc/>
        public double[] Activation(double[] nodes)
        {
            if (nodes.Length == 0)
                return Array.Empty<double>();

            double invTemp = 1.0 / Temperature;

            // Subtracting the max keeps every exponent <= 0, so this cannot overflow,
            // and the max element contributes e^0 = 1, so the sum is always >= 1.
            double max = double.NegativeInfinity;
            for (int i = 0; i < nodes.Length; i++)
                max = Math.Max(max, nodes[i] * invTemp);

            var result = new double[nodes.Length];
            double sum = 0.0;
            for (int i = 0; i < nodes.Length; i++)
            {
                result[i] = Math.Exp(nodes[i] * invTemp - max);
                sum += result[i];
            }

            for (int i = 0; i < result.Length; i++)
                result[i] /= sum;

            return result;
        }

        /// <summary>
        /// Not supported: SoftMax is not element-wise. The output-layer gradient is computed jointly with cross-entropy.
        /// </summary>
        public double Derivative(double x)
            => throw new NotSupportedException("SoftMax has no element-wise derivative; it may only be used on the output layer.");
    }
}
