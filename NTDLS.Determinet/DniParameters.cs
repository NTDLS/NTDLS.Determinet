using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    /// <summary>
    /// Provides a collection of predefined constants and utilities for configuring neural network parameters,
    /// activation functions, and layer-specific settings in machine learning applications.
    /// </summary>
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
            /// Maximum global L2 norm of the gradient (across every weight, bias and normalization parameter) for a
            /// single update. When the norm exceeds this value the whole gradient is scaled down to it, which preserves
            /// the update direction. Set to 0 to disable clipping.
            /// </summary>
            public static readonly DniNamedParameter GradientClip = new("Network_GradientClip", typeof(double), 5.0);

            /// <summary>
            /// When <see langword="true"/>, updates use the Adam optimizer (with decoupled weight decay, i.e. AdamW)
            /// for both Train() and TrainBatch(). When <see langword="false"/>, plain SGD with L2 weight decay is used.
            /// </summary>
            public static readonly DniNamedParameter UseAdamOptimization = new("Network_UseAdamOptimization", typeof(bool), false);
        }

        /// <summary>
        /// Provides constants representing configuration keys for layer-related settings in a neural network.
        /// </summary>
        public static class Layer
        {
            /// <summary>
            /// Enables layer normalization on this layer: the layer's weighted sums are normalized to zero mean and unit
            /// variance across the layer's nodes (per sample), then scaled and shifted by learned gamma/beta values before
            /// the activation function is applied. Because the statistics are per sample, training and inference behave
            /// identically. Only valid on intermediate layers.
            /// </summary>
            public static readonly DniNamedParameter UseLayerNorm = new("Layer_UseLayerNorm", typeof(bool), false);
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
            public static readonly DniNamedParameter Alpha = new("Linear_Alpha", typeof(double), 1.0);

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
            public static readonly DniNamedParameter Alpha = new("Piecewise_Alpha", typeof(double), 1.0);
            /// <summary>
            /// Represents a named parameter that specifies a range of values for piecewise alpha calculations.
            /// </summary>
            /// <remarks>The range is defined by a <see cref="DniRange"/> object, which includes a
            /// minimum and maximum value. The default range is from -1 to +1.</remarks>
            public static readonly DniNamedParameter Range = new("Piecewise_AlphaRange", typeof(DniRange), new DniRange(-1, +1));
        }

        /// <summary>
        /// Provides a collection of predefined parameters for the Exponential Linear Unit (ELU) activation function.
        /// </summary>
        /// <remarks>This class contains static members representing named parameters commonly used with
        /// the ELU activation function.</remarks>
        public static class ELU
        {
            /// <summary>
            /// Represents the named parameter for the alpha value, typically used in calculations or configurations.
            /// </summary>
            /// <remarks>The parameter is identified by the name "ELU_Alpha" and is associated with
            /// the <see cref="double"/> type.  The default value is set to 1.</remarks>
            public static readonly DniNamedParameter Alpha = new("ELU_Alpha", typeof(double), 1.0);
        }

        /// <summary>
        /// Provides predefined parameters for the Scaled Exponential Linear Unit (SELU) activation function.
        /// </summary>
        /// <remarks>The SELU class defines two constants, <see cref="Alpha"/> and <see cref="Lambda"/>,
        /// which are commonly used in the SELU activation function. These parameters are based on the original
        /// formulation of SELU and are intended for use in machine learning and neural network applications.</remarks>
        public static class SELU
        {
            /// <summary>
            /// Represents the alpha parameter for the SELU activation function.
            /// </summary>
            /// <remarks>This parameter is used in the Scaled Exponential Linear Unit (SELU)
            /// activation function,  which is commonly applied in neural network computations. The default value is
            /// 1.6732632423543772, from the original SELU paper.</remarks>
            public static readonly DniNamedParameter Alpha = new("SELU_Alpha", typeof(double), 1.6732632423543772);
            /// <summary>
            /// Represents the lambda parameter used in the SELU (Scaled Exponential Linear Unit) activation function.
            /// </summary>
            /// <remarks>This parameter is a constant value commonly used in the SELU activation
            /// function to ensure self-normalizing properties. The default value is <c>1.0507009873554805</c>, which is derived
            /// from the original SELU paper.</remarks>
            public static readonly DniNamedParameter Lambda = new("SELU_Lambda", typeof(double), 1.0507009873554805);
        }
    }
}
