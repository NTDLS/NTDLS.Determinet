using NTDLS.Determinet.Types;
using ProtoBuf;

namespace NTDLS.Determinet
{
    /// <summary>
    /// Represents the state of a neural network, including its learning rate, layers, and synapses.
    /// </summary>
    /// <remarks>This class encapsulates the configuration and structure of a neural network at a specific
    /// point in time. It includes the learning rate, which influences the training process, as well as the layers and
    /// synapses that define the network's architecture.</remarks>
    [ProtoContract]
    public class DniStateOfBeing
    {
        /// <summary>
        /// Gets the collection of named parameters associated with the current instance.
        /// </summary>
        [ProtoMember(1)] public DniNamedParameterCollection Parameters { get; private set; } = new();

        /// <summary>
        /// Gets the collection of layers associated with the current object.
        /// </summary>
        [ProtoMember(2)] public List<DniLayer> Layers { get; internal set; } = new();

        /// <summary>
        /// Gets the collection of synapses associated with this instance.
        /// </summary>
        [ProtoMember(3)] public List<DniSynapse> Synapses { get; internal set; } = new();

        #region Adam (Adaptive Moment Estimation) for batch training.

        //TODO: Serialize these Adam properties because the accumulated gradient statistics are gone when we resume training from a saved state.
        //This isn't huge, it just means that the very next updates behave like vanilla SGD for a few iterations until it re-adapts.
        //Unfortunately, ProtoBuf doesn't support serialization of "jagged" arrays.

        /// <summary>
        /// Represents the first moment (mean) estimates for the Adam optimization algorithm.
        /// </summary>
        /// <remarks>This collection stores the accumulated gradient statistics for each weight matrix
        /// during training. These values are used by the Adam optimizer to adjust the learning rate for each parameter.
        /// Note: The accumulated statistics are not serialized, which means they are lost when resuming training from a
        /// saved state. As a result, the optimizer may behave like vanilla SGD for a few iterations until the
        /// statistics are re-adapted.</remarks>
        public List<double[,]> AdamMeanWeights = new(); // first moment (mean).
        /// <summary>
        /// Represents the second moment (variance) weights used in the Adam optimization algorithm.
        /// </summary>
        /// <remarks>This collection stores the variance weights as a list of two-dimensional arrays,
        /// where each array corresponds to a specific layer or parameter group. The Adam optimization algorithm uses
        /// these weights to adjust learning rates during training.</remarks>
        public List<double[,]> AdamVarianceWeights = new(); // second moment (variance).
        /// <summary>
        /// Represents the mean biases used in the Adam optimization algorithm.
        /// </summary>
        /// <remarks>This list stores arrays of double-precision floating-point values, where each array
        /// corresponds to the mean bias values  for a specific layer or parameter group in the optimization process. It
        /// is typically updated during training iterations.</remarks>
        public List<double[]> AdamMeanBiases = new();
        /// <summary>
        /// Represents the variance biases used in the Adam optimization algorithm.
        /// </summary>
        /// <remarks>This list contains arrays of double-precision floating-point numbers, where each
        /// array corresponds to the variance biases  for a specific layer or parameter group in the model. It is
        /// typically used to store and update the second moment estimates  during the optimization process.</remarks>
        public List<double[]> AdamVarianceBiases = new();
        /// <summary>
        /// Represents the iteration counter used for bias correction in the Adam optimization algorithm.
        /// </summary>
        /// <remarks>This value is incremented with each optimization step and is used to compute
        /// bias-corrected estimates of the first and second moment vectors.</remarks>
        public long AdamTimeStep = 0; // iteration counter for bias correction.

        #endregion

        /// <summary>
        /// Gets or sets the labels associated with the input layer of the model.
        /// </summary>
        public string[]? InputLabels
        {
            get
            {
                var layer = Layers.Last()
                    ?? throw new Exception("Input layer is not defined.");
                return layer.Labels;
            }
            set
            {
                var layer = Layers.Last()
                    ?? throw new Exception("Input layer is not defined.");

                if (value != null && value.Length != layer.NodeCount)
                    throw new Exception("Input layer label count does not match node count.");

                layer.Labels = value;
            }
        }

        /// <summary>
        /// Gets or sets the labels associated with the output layer of the model.
        /// </summary>
        public string[]? OutputLabels
        {
            get
            {
                var layer = Layers.Last()
                    ?? throw new Exception("Output layer is not defined.");
                return layer.Labels;
            }
            set
            {
                var layer = Layers.Last()
                    ?? throw new Exception("Output layer is not defined.");

                if (value != null && value.Length != layer.NodeCount)
                    throw new Exception("Output layer label count does not match node count.");

                layer.Labels = value;
            }
        }
    }
}
