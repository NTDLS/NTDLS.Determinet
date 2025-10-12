namespace NTDLS.Determinet
{
    public class DniParameters
    {
        public class LayerParameters
        {
            /// <summary>
            /// Represents the configuration key for enabling or disabling batch normalization.
            /// </summary>
            public const string UseBatchNorm = "UseBatchNorm";
            /// <summary>
            /// In BatchNorm, “momentum” controls how quickly the running mean and variance adapt to the most recent batch statistics during training.
            /// </summary>
            public const string BatchNormMomentum = "BatchNormMomentum";

            /// <summary>
            /// Temperature scaling factor for SoftMax activation function.
            /// Higher values soften probabilities; lower values sharpen them.
            /// </summary>
            public const string SoftMaxTemperature = "SoftMaxTemperature";

            /// <summary>
            /// Represents the parameter name for the alpha value used in the Leaky ReLU activation function.
            /// </summary>
            /// <remarks>The alpha value determines the slope of the function for negative input
            /// values in the Leaky ReLU activation function. This constant can be used as a key or identifier in
            /// contexts where the alpha parameter needs to be specified or retrieved.</remarks>
            public const string LeakyReLUAlpha = "LeakyReLUAlpha";
        }
    }
}
