namespace NTDLS.Determinet.ActivationFunctions.Interfaces
{
    /// <summary>
    /// Represents an activation function used in neural network layers, providing methods for  calculating activation
    /// values and their derivatives.
    /// </summary>
    /// <remarks>Activation functions are a key component of neural networks, introducing non-linearity  to
    /// the model. This interface defines the contract for implementing custom activation  functions, which can be
    /// applied to nodes in a layer and used to compute gradients during  backpropagation.</remarks>
    public interface IDniActivationFunction
    {
        /// <summary>
        /// Applies an activation function to the specified array of input nodes.
        /// </summary>
        /// <param name="nodes">An array of double values representing the input nodes to be processed. Cannot be null.</param>
        /// <returns>An array of double values representing the output nodes after the activation function is applied. The length
        /// of the output array matches the input array.</returns>
        double[] Activation(double[] nodes);

        /// <summary>
        /// Calculates the derivative of a mathematical function at the specified point.
        /// </summary>
        /// <param name="x">The point at which to evaluate the derivative.</param>
        /// <returns>The value of the derivative at the specified point.</returns>
        double Derivative(double x);

        /// <summary>
        /// Gets a value indicating whether the cross-entropy method is used in the analysis.
        /// </summary>
        bool UsesCrossEntropy { get; }
    }
}
