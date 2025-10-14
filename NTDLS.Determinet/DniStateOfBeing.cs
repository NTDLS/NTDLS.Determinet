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

        public List<double[,]> AdamMeanWeights = new(); // first moment (mean).
        public List<double[,]> AdamVarianceWeights = new(); // second moment (variance).
        public List<double[]> AdamMeanBiases = new();
        public List<double[]> AdamVarianceBiases = new();
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
