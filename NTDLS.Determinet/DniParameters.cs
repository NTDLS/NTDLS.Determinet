namespace NTDLS.Determinet
{
    public static class DniParameters
    {
        /// <summary>
        /// Provides constants for configuring neural network parameters, such as learning rate, weight decay, and
        /// gradient clipping.
        /// </summary>
        public static class Network
        {
            /// <summary>
            /// Represents the configuration key for specifying the learning rate of the network.
            /// </summary>
            /// <remarks>This constant is typically used to retrieve or set the learning rate value in
            /// a configuration system. The learning rate controls the step size during the optimization process in
            /// machine learning algorithms.</remarks>
            public const string LearningRate = "Network_LearningRate";
            public const double DefaultLearningRate = 0.005;

            /// <summary>
            /// Represents the configuration key for specifying the weight decay parameter in a neural network.
            /// </summary>
            /// <remarks>Weight decay is a regularization technique used to prevent overfitting by
            /// penalizing large weights in the network. This constant can be used as a key in configuration settings or
            /// parameter dictionaries.</remarks>
            public const string WeightDecay = "Network_WeightDecay";
            public const double DefaultWeightDecay = 0.0001;

            /// <summary>
            /// Represents the key used to identify the gradient clipping setting in the network configuration.
            /// </summary>
            public const string GradientClip = "Network_GradientClip";
            public const double DefaultGradientClip = 0.5;
        }

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
            public const bool DefaultUseBatchNorm = false;
            /// <summary>
            /// In BatchNorm, “momentum” controls how quickly the running mean and variance adapt to the most recent batch statistics during training.
            /// </summary>
            public const string BatchNormMomentum = "Layer_BatchNormMomentum";
            public const double DefaultBatchNormMomentum = 0.2;
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
            public const double DefaultTemperature = 1.0;

            /// <summary>
            /// Maximum absolute logit value allowed before clamping.
            /// Prevents overflow in exp() and stabilizes training.
            /// </summary>

            public const string MaxLogit = "SoftMax_MaxLogit";
            public const double DefaultMaxLogit = 50;
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
            public const double DefaultAlpha = 0.01;
        }
    }
}
