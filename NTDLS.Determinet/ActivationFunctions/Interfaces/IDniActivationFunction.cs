namespace NTDLS.Determinet.ActivationFunctions.Interfaces
{
    /*
     * Parameters that are common to all layers:
     *  UseBatchNorm: bool     | whether to use batch normalization or for the layer. See BatchNormalize() method in DniNetwork class.
     *  BatchNormGamma: double | (also called scale), default 1. Used in batch normalization.
     *  BatchNormBeta: double  | (also called shift), default 0. Used in batch normalization.
    */

    /// <summary>
    /// Represents an activation function used in neural network layers, providing methods for  calculating activation
    /// values and their derivatives.
    /// </summary>
    /// <remarks>Activation functions are a key component of neural networks, introducing non-linearity  to
    /// the model. This interface defines the contract for implementing custom activation  functions, which can be
    /// applied to nodes in a layer and used to compute gradients during  backpropagation.</remarks>
    public interface IDniActivationFunction
    {
        double[] Activation(double[] nodes);
        double Derivative(double x);
    }
}
