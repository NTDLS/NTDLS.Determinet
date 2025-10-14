using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    public static class DniParameters
    {
        /// <summary>
        /// Provides constants for configuring neural network parameters, such as learning rate, weight decay, and
        /// gradient clipping, etc.
        /// </summary>
        public static class Network
        {
            /// <summary>
            /// Represents a named parameter for the computed loss value in a network operation.
            /// This value is set after Backpropagate during each call to Train().
            /// </summary>
            public static readonly DniNamedParameter ComputedLoss = new("Network_ComputedLoss", typeof(double), double.PositiveInfinity);

            /// <summary>
            /// Represents the configuration key for specifying the learning rate of the network.
            /// </summary>
            /// <remarks>This constant is typically used to retrieve or set the learning rate value in
            /// a configuration system. The learning rate controls the step size during the optimization process in
            /// machine learning algorithms.</remarks>
            public static readonly DniNamedParameter LearningRate = new("Network_LearningRate", typeof(double), 0.005);

            /// <summary>
            /// Represents the configuration key for specifying the weight decay parameter in a neural network.
            /// </summary>
            /// <remarks>Weight decay is a regularization technique used to prevent overfitting by
            /// penalizing large weights in the network. This constant can be used as a key in configuration settings or
            /// parameter dictionaries.</remarks>
            public static readonly DniNamedParameter WeightDecay = new("Network_WeightDecay", typeof(double), 0.0001);

            /// <summary>
            /// Represents the key used to identify the gradient clipping setting in the network configuration.
            /// </summary>
            public static readonly DniNamedParameter GradientClip = new("Network_GradientClip", typeof(double), 0.5);

            /// <summary>
            /// Represents a named parameter that specifies whether to enable Adam (Adaptive Moment Estimation) optimization when batch training
            /// </summary>
            /// <remarks>This parameter determines if the Adam optimization algorithm should be
            /// applied to batch processing.  The default value is <see langword="false"/>.</remarks>
            public static readonly DniNamedParameter UseAdamBatchOptimization = new("Network_UseAdamBatchOptimization", typeof(bool), false);
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
            public static readonly DniNamedParameter UseBatchNorm = new("Layer_UseBatchNorm", typeof(bool), false);
            /// <summary>
            /// In BatchNorm, “momentum” controls how quickly the running mean and variance adapt to the most recent batch statistics during training.
            /// </summary>
            public static readonly DniNamedParameter BatchNormMomentum = new("Layer_BatchNormMomentum", typeof(double), 0.2);
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
            public static readonly DniNamedParameter Temperature = new("SoftMax_Temperature", typeof(double), 1.0);

            /// <summary>
            /// Maximum absolute logit value allowed before clamping.
            /// Prevents overflow in exp() and stabilizes training.
            /// </summary>

            public static readonly DniNamedParameter MaxLogit = new("SoftMax_MaxLogit", typeof(double), 50.0);
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
            public static readonly DniNamedParameter Alpha = new("LeakyReLU_Alpha", typeof(double), 0.01);
        }

        /// <summary>
        /// Provides predefined named parameters for configuring linear operations.
        /// </summary>
        /// <remarks>This class contains static fields representing commonly used parameters in linear
        /// computations,  such as the alpha value and its range. These parameters are intended to be used with APIs
        /// that  accept named parameters for configuration.</remarks>
        public static class Linear
        {
            /// <summary>
            /// Represents the alpha parameter for a linear operation.
            /// </summary>
            /// <remarks>This parameter is used to specify the alpha value, which is a coefficient in
            /// linear calculations. The default value is <see langword="1"/>. The parameter type is <see
            /// cref="double"/>.</remarks>
            public static readonly DniNamedParameter Alpha = new("Linear_Alpha", typeof(double), 1);

            /// <summary>
            /// Represents a named parameter with a linear alpha range.
            /// </summary>
            /// <remarks>This parameter is defined with a default range of values from -1 to +1. It is
            /// commonly used in scenarios where a linear range is required for alpha values.</remarks>
            public static readonly DniNamedParameter Range = new("Linear_AlphaRange", typeof(DniRange), new DniRange(-1, +1));
        }

        /// <summary>
        /// Provides a collection of named parameters used for configuring piecewise operations.
        /// </summary>
        /// <remarks>This class defines static readonly fields representing parameters commonly used in
        /// piecewise computations. These parameters include <see cref="Alpha"/>, which represents a scalar value, and
        /// <see cref="Range"/>, which defines a range of values.</remarks>
        public static class Piecewise
        {
            /// <summary>
            /// Represents the alpha parameter used in piecewise calculations.
            /// </summary>
            /// <remarks>This parameter is identified by the name "Piecewise_Alpha" and has a default
            /// value of 1.  It is of type <see cref="double"/>.</remarks>
            public static readonly DniNamedParameter Alpha = new("Piecewise_Alpha", typeof(double), 1);
            /// <summary>
            /// Represents a named parameter that specifies a range of values for piecewise alpha calculations.
            /// </summary>
            /// <remarks>The range is defined by a <see cref="DniRange"/> object, which includes a
            /// minimum and maximum value. The default range is from -1 to +1.</remarks>
            public static readonly DniNamedParameter Range = new("Piecewise_AlphaRange", typeof(DniRange), new DniRange(-1, +1));
        }
    }
}
