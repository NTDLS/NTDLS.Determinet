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
        /// <remarks>This activation function applies a linear transformation to input values based on the
        /// specified range: - For inputs less than or equal to the minimum of the range, the output is scaled by the slope
        /// value. - For inputs greater than or equal to the maximum of the range, the output is also scaled by the slope
        /// value. - For inputs within the range, the output is equal to the input.  The derivative of the function is
        /// constant outside the range (equal to the slope) and 1 within the range.</remarks>
        PiecewiseLinear,
        /// <summary>
        /// Represents a linear activation function with configurable slope and output range.
        /// </summary>
        /// <remarks>This activation function applies a linear transformation to the input values, scaling them by
        /// the  <see cref="Alpha"/> parameter. The output is clamped to the specified <see cref="Range"/> to ensure  it
        /// remains within the defined bounds. This function is commonly used in neural network models where  a simple
        /// linear transformation is required.</remarks>
        Linear,
        /// <summary>
        /// Represents the Rectified Linear Unit (ReLU) activation function, commonly used in neural networks.
        /// </summary>
        /// <remarks>The ReLU function outputs the input value if it is greater than zero; otherwise, it outputs
        /// zero. It is widely used in deep learning due to its simplicity and effectiveness in introducing
        /// non-linearity.</remarks>
        ReLU,
        /// <summary>
        /// Represents the sigmoid activation function, commonly used in neural networks.
        /// </summary>
        /// <remarks>The sigmoid function maps input values to a range between 0 and 1, making it useful for 
        /// applications such as binary classification. This class also provides the derivative of  the sigmoid function,
        /// which is often used during backpropagation in training neural networks.</remarks>
        Sigmoid,
        /// <summary>
        /// Represents the hyperbolic tangent (tanh) activation function, commonly used in neural networks.
        /// </summary>
        /// <remarks>This class provides methods to compute the activation and derivative of the tanh function.
        /// The tanh function maps input values to the range [-1, 1], making it useful for normalizing data in neural
        /// network layers. The derivative is used during backpropagation to compute gradients.</remarks>
        Tanh,
        /// <summary>
        /// Represents the Leaky ReLU (Rectified Linear Unit) activation function, which introduces a small slope for
        /// negative input values.
        /// </summary>
        /// <remarks>The Leaky ReLU activation function is commonly used in neural networks to address the "dying
        /// ReLU" problem by allowing a small, non-zero gradient for negative input values. The slope for negative values is
        /// determined by the <see cref="Alpha"/> parameter.</remarks>
        LeakyReLU,
        /// <summary>
        /// Represents a softmax activation function with temperature scaling and numerical stability enhancements.
        /// </summary>
        /// <remarks>The softmax function is commonly used in machine learning for converting a vector of raw
        /// scores (logits)  into probabilities. This implementation includes temperature scaling to control the sharpness
        /// of the  output probabilities and clamping of logits to prevent numerical instability during
        /// computation.</remarks>
        SoftMax,
        /// <summary>
        /// Represents the SoftMax activation function, commonly used in neural networks to convert raw output values into
        /// probabilities.
        /// </summary>
        /// <remarks>The SoftMax function normalizes an input array of values into a probability distribution,
        /// where each value is scaled exponentially and divided by the sum of all exponentials. This ensures that the
        /// output values are non-negative and sum to 1. The function is numerically stable, handling cases where input
        /// values are large or small to avoid overflow or underflow.</remarks>
        SimpleSoftMax
    }
}
