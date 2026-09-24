namespace NTDLS.Determinet.ActivationFunctions.Interfaces
{
    /// <summary>
    /// Defines an element-wise activation function and its derivative.
    /// </summary>
    public interface IDniActivationFunction
    {
        /// <summary>
        /// Applies the activation function to each node, returning a new array.
        /// Implementations must not modify <paramref name="nodes"/>.
        /// </summary>
        /// <param name="nodes">The pre-activation values.</param>
        double[] Activation(double[] nodes);

        /// <summary>
        /// Computes the derivative of the activation function with respect to its pre-activation input.
        /// </summary>
        /// <param name="x">The pre-activation value (the same value that was passed to <see cref="Activation"/>).</param>
        double Derivative(double x);
    }

    /// <summary>
    /// Marks an activation function as a SoftMax. SoftMax is not element-wise, so it has no meaningful scalar derivative.
    /// It is only valid on the output layer, where it is paired with cross-entropy loss and the combined gradient is
    /// computed analytically as (p - t) / Temperature.
    /// </summary>
    public interface IDniSoftMaxFunction : IDniActivationFunction
    {
        /// <summary>
        /// The temperature the logits are divided by before exponentiation.
        /// </summary>
        double Temperature { get; }
    }
}
