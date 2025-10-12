namespace NTDLS.Determinet
{
    public static class DniParameters
    {
        /// <summary>
        /// Provides constants representing configuration keys for layer-related settings in a neural network.
        /// </summary>
        /// <remarks>These constants are typically used to configure layer behaviors, such as enabling
        /// batch normalization or adjusting its parameters, in a machine learning framework.</remarks>
        public static class Layer
        {
            /// <summary>
            /// Represents the configuration key for enabling or disabling batch normalization.
            /// </summary>
            public const string UseBatchNorm = "Layer_UseBatchNorm";
            /// <summary>
            /// In BatchNorm, “momentum” controls how quickly the running mean and variance adapt to the most recent batch statistics during training.
            /// </summary>
            public const string BatchNormMomentum = "Layer_BatchNormMomentum";
        }

        /// <summary>
        /// Provides constants and utilities related to the DniSoftMaxFunction activation function.
        /// </summary>
        public static class SoftMax
        {
            /// <summary>
            /// Temperature scaling factor for SoftMax activation function.
            /// Higher values soften probabilities; lower values sharpen them.
            /// </summary>
            public const string Temperature = "SoftMax_Temperature";

            /// <summary>
            /// Maximum absolute logit value allowed before clamping.
            /// Prevents overflow in exp() and stabilizes training.
            /// </summary>

            public const string MaxLogit = "SoftMax_MaxLogit";
        }

        /// <summary>
        /// Provides constants and utilities related to the DniLeakyReLUFunction activation function.
        /// </summary>
        public static class LeakyReLU
        {
            /// <summary>
            /// Represents the parameter name for the alpha value used in the Leaky ReLU activation function.
            /// </summary>
            /// <remarks>The alpha value determines the slope of the function for negative input
            /// values in the Leaky ReLU activation function. This constant can be used as a key or identifier in
            /// contexts where the alpha parameter needs to be specified or retrieved.</remarks>
            public const string Alpha = "LeakyReLU_Alpha";
        }
    }
}
