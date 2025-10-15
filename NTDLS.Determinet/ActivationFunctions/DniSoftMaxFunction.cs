using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// Represents a softmax activation function with temperature scaling and numerical stability enhancements.
    /// </summary>
    /// <remarks>The softmax function is commonly used in machine learning for converting a vector of raw
    /// scores (logits)  into probabilities. This implementation includes temperature scaling to control the sharpness
    /// of the  output probabilities and clamping of logits to prevent numerical instability during
    /// computation.</remarks>
    public class DniSoftMaxFunction : IDniActivationFunction
    {
        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        public bool UsesCrossEntropy { get; } = true;

        /// <summary>
        /// Temperature scaling factor. 
        /// Higher values soften probabilities; lower values sharpen them.
        /// </summary>
        public double Temperature { get; private set; }

        /// <summary>
        /// Maximum absolute logit value allowed before clamping.
        /// Prevents overflow in exp() and stabilizes training.
        /// </summary>
        public double MaxLogit { get; private set; }

        /// <summary>
        /// Default constructor for SoftMax activation function.
        /// </summary>
        public DniSoftMaxFunction(DniNamedParameterCollection param)
        {
            Temperature = param.Get<double>(SoftMax.Temperature);
            MaxLogit = param.Get<double>(SoftMax.MaxLogit);
        }

        /// <summary>
        /// Applies the activation function to each element in the input array.
        /// </summary>
        public double[] Activation(double[] nodes)
        {
            if (nodes.Length == 0)
                return Array.Empty<double>();

            // Clamp logits to prevent extreme exponentials
            double[] clamped = nodes.Select(v => Math.Clamp(v, -MaxLogit, MaxLogit)).ToArray();

            // Numerical stability: subtract the max before exp
            double max = clamped.Max();

            // Apply temperature scaling
            double invTemp = 1.0 / Temperature;

            double[] exps = clamped.Select(v =>
            {
                double e = Math.Exp((v - max) * invTemp);
                if (double.IsNaN(e) || double.IsInfinity(e))
                    e = 0.0;
                return e;
            }).ToArray();

            double sum = exps.Sum();

            if (sum <= 1e-12 || double.IsNaN(sum) || double.IsInfinity(sum))
                sum = 1e-12;

            for (int i = 0; i < exps.Length; i++)
                exps[i] /= sum;

            return exps;
        }

        /// <summary>
        /// Calculates the derivative of the activation function at the specified input value.
        /// </summary>
        public double Derivative(double x)
        {
            // Not used directly; handled by cross-entropy gradient
            return 1.0;
        }
    }
}
