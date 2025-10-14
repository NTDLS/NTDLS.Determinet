namespace NTDLS.Determinet.Types
{
    [Serializable]
    public enum DniLayerType
    {
        Input,
        Intermediate,
        Output
    }

    [Serializable]
    public enum DniActivationType
    {
        None,

        /// <summary>
        /// Represents an identity activation function, which returns the input as the output without modification.
        /// </summary>
        Identity,

        /// <summary>
        /// Represents a piecewise linear activation function with configurable slope and output range.
        /// Combines a linear segment for certain input range with a Leaky ReLU-like behavior for values outside that range.
        /// </summary>
        /// <remarks>
        /// This activation function applies a linear transformation to input values based on the specified range:
        /// - For inputs less than or equal to the minimum of the range, the output is scaled by the slope value.
        /// - For inputs greater than or equal to the maximum of the range, the output is also scaled by the slope value.
        /// - For inputs within the range, the output is equal to the input.
        /// </remarks>
        PiecewiseLinear,

        /// <summary>
        /// Represents a linear activation function with configurable slope and output range.
        /// </summary>
        /// <remarks>
        /// This activation function applies a linear transformation to the input values, scaling them by the <see cref="Alpha"/> parameter.
        /// The output is clamped to the specified <see cref="Range"/> to ensure it remains within defined bounds.
        /// Commonly used in regression networks or as an output layer for continuous values.
        /// </remarks>
        Linear,

        /// <summary>
        /// Represents the Rectified Linear Unit (ReLU) activation function, commonly used in neural networks.
        /// </summary>
        /// <remarks>
        /// The ReLU function outputs the input value if it is greater than zero; otherwise, it outputs zero.
        /// It is widely used due to its simplicity, fast convergence, and ability to introduce non-linearity.
        /// </remarks>
        ReLU,

        /// <summary>
        /// Represents the sigmoid activation function, commonly used in neural networks.
        /// </summary>
        /// <remarks>
        /// The sigmoid function maps input values to a range between 0 and 1, making it useful for binary classification and probabilistic outputs.
        /// </remarks>
        Sigmoid,

        /// <summary>
        /// Represents the hyperbolic tangent (tanh) activation function.
        /// </summary>
        /// <remarks>
        /// The tanh function maps input values to the range [-1, 1], providing zero-centered outputs.
        /// Useful in networks where both positive and negative activations are meaningful, such as RNNs.
        /// </remarks>
        Tanh,

        /// <summary>
        /// Represents the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
        /// </summary>
        /// <remarks>
        /// Leaky ReLU allows a small, non-zero gradient for negative input values to avoid “dead neurons.”
        /// The slope for negative inputs is controlled by the <see cref="Alpha"/> parameter.
        /// </remarks>
        LeakyReLU,

        /// <summary>
        /// Represents a SoftMax activation function with temperature scaling and numerical stability enhancements.
        /// </summary>
        /// <remarks>
        /// SoftMax converts a vector of raw scores (logits) into probabilities that sum to 1.
        /// Temperature scaling adjusts confidence sharpness; clamping improves numerical stability.
        /// Typically used for multi-class classification output layers.
        /// </remarks>
        SoftMax,

        /// <summary>
        /// Represents a simplified SoftMax activation function.
        /// </summary>
        /// <remarks>
        /// This version omits temperature scaling but still converts logits into normalized probabilities.
        /// Useful for standard classification problems where only one class is correct.
        /// </remarks>
        SimpleSoftMax,

        /// <summary>
        /// Represents the Exponential Linear Unit (ELU) activation function.
        /// </summary>
        /// <remarks>
        /// ELU behaves like ReLU for positive values, but for negative inputs, it smoothly approaches -α.
        /// Helps maintain zero-centered activations and improves convergence stability.
        /// </remarks>
        ELU,

        /// <summary>
        /// Represents the Gaussian (Radial Basis Function) activation function.
        /// </summary>
        /// <remarks>
        /// Outputs values in a bell-shaped curve around zero using f(x) = exp(-x²).
        /// Commonly used in Radial Basis Function networks and anomaly detection.
        /// </remarks>
        Gaussian,

        /// <summary>
        /// Represents a hard (linear) approximation of the sigmoid activation function.
        /// </summary>
        /// <remarks>
        /// The Hard Sigmoid is a fast, piecewise-linear version of sigmoid: f(x) = clamp(0.2x + 0.5, 0, 1).
        /// Used in real-time or embedded systems where performance is critical.
        /// </remarks>
        HardSigmoid,

        /// <summary>
        /// Represents the HardTanh activation function.
        /// </summary>
        /// <remarks>
        /// A piecewise linear variant of tanh that clips output to the range [-1, 1].
        /// Useful for bounded activations and faster computations.
        /// </remarks>
        HardTanh,

        /// <summary>
        /// Represents the Mish activation function.
        /// </summary>
        /// <remarks>
        /// Mish is a smooth, non-monotonic function defined as f(x) = x * tanh(ln(1 + exp(x))).
        /// Provides improved gradient flow and stability over ReLU in some architectures.
        /// </remarks>
        Mish,

        /// <summary>
        /// Represents the Scaled Exponential Linear Unit (SELU) activation function.
        /// </summary>
        /// <remarks>
        /// SELU scales ELU by predefined constants (λ ≈ 1.0507, α ≈ 1.67326) to maintain self-normalizing behavior.
        /// Helps preserve mean and variance across layers without explicit normalization.
        /// </remarks>
        SELU,

        /// <summary>
        /// Represents the Softsign activation function.
        /// </summary>
        /// <remarks>
        /// Softsign smoothly scales input as f(x) = x / (1 + |x|).
        /// Similar to tanh but computationally simpler and numerically stable.
        /// </remarks>
        SoftSign,

        /// <summary>
        /// Represents the Swish activation function.
        /// </summary>
        /// <remarks>
        /// Defined as f(x) = x * sigmoid(x), Swish is smooth and self-gating.
        /// Often outperforms ReLU in deep architectures by maintaining gradient flow.
        /// </remarks>
        Swish,

        /// <summary>
        /// Represents the Softplus activation function.
        /// </summary>
        /// <remarks>
        /// Softplus is a smooth approximation of the ReLU function, defined as f(x) = ln(1 + exp(x)).
        /// It produces continuous, non-zero gradients everywhere, avoiding the “dead neuron” problem
        /// while maintaining similar behavior to ReLU for large positive inputs.
        /// Commonly used in deep networks where smoothness and differentiability are desired.
        /// </remarks>
        SoftPlus
    }
}
