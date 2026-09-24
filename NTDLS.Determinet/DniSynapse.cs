using ProtoBuf;

namespace NTDLS.Determinet
{
    /// <summary>
    /// The trainable connections between two adjacent layers: a weight matrix and one bias per node of the receiving layer.
    /// </summary>
    /// <remarks>Weights are stored flat, one row per receiving (output) node, so that the weight connecting input node
    /// <c>i</c> to output node <c>o</c> is at <c>Weights[o * InputCount + i]</c>. This makes each output node's weighted
    /// sum a contiguous dot product. Use <see cref="GetWeight"/> / <see cref="SetWeight"/> for indexed access.</remarks>
    [ProtoContract]
    public class DniSynapse
    {
        #region Legacy (pre 2.0) serialized layout: [input, output] row-major. Read on load, never written.

        [ProtoMember(1)] private int LegacyRows { get; set; }
        [ProtoMember(2)] private int LegacyCols { get; set; }
        [ProtoMember(3)] private double[] LegacyFlatWeights { get; set; } = [];

        #endregion

        /// <summary>
        /// The bias for each node in the receiving layer.
        /// </summary>
        [ProtoMember(4)] public double[] Biases { get; internal set; } = [];

        /// <summary>
        /// The number of nodes in the sending (previous) layer.
        /// </summary>
        [ProtoMember(5)] public int InputCount { get; private set; }

        /// <summary>
        /// The number of nodes in the receiving (next) layer.
        /// </summary>
        [ProtoMember(6)] public int OutputCount { get; private set; }

        /// <summary>
        /// Flat weight matrix, one row of <see cref="InputCount"/> weights per receiving node.
        /// </summary>
        [ProtoMember(7)] public double[] Weights { get; internal set; } = [];

        #region Adam optimizer state (first and second moments), persisted so training can resume seamlessly.

        [ProtoMember(8)] internal double[] AdamMeanWeights { get; set; } = [];
        [ProtoMember(9)] internal double[] AdamVarianceWeights { get; set; } = [];
        [ProtoMember(10)] internal double[] AdamMeanBiases { get; set; } = [];
        [ProtoMember(11)] internal double[] AdamVarianceBiases { get; set; } = [];

        #endregion

        /// <summary>
        /// Initializes a new synapse with zeroed weights and biases.
        /// </summary>
        public DniSynapse(int inputCount, int outputCount)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            Weights = new double[inputCount * outputCount];
            Biases = new double[outputCount];
        }

        /// <summary>
        /// Used only for deserialization.
        /// </summary>
        public DniSynapse()
        {
        }

        /// <summary>
        /// Gets the weight of the connection from input node <paramref name="input"/> to output node <paramref name="output"/>.
        /// </summary>
        public double GetWeight(int input, int output)
            => Weights[output * InputCount + input];

        /// <summary>
        /// Sets the weight of the connection from input node <paramref name="input"/> to output node <paramref name="output"/>.
        /// </summary>
        public void SetWeight(int input, int output, double value)
            => Weights[output * InputCount + input] = value;

        /// <summary>
        /// Allocates Adam moment buffers if they do not exist yet (or no longer match the weight shape).
        /// </summary>
        internal void EnsureAdamBuffers()
        {
            if (AdamMeanWeights.Length != Weights.Length)
            {
                AdamMeanWeights = new double[Weights.Length];
                AdamVarianceWeights = new double[Weights.Length];
            }
            if (AdamMeanBiases.Length != Biases.Length)
            {
                AdamMeanBiases = new double[Biases.Length];
                AdamVarianceBiases = new double[Biases.Length];
            }
        }

        /// <summary>
        /// Converts a synapse loaded from the legacy [input, output] layout and validates the shape.
        /// </summary>
        internal void AfterDeserialization()
        {
            if (Weights.Length == 0 && LegacyFlatWeights.Length > 0)
            {
                InputCount = LegacyRows;
                OutputCount = LegacyCols;
                Weights = new double[InputCount * OutputCount];
                for (int i = 0; i < InputCount; i++)
                    for (int o = 0; o < OutputCount; o++)
                        Weights[o * InputCount + i] = LegacyFlatWeights[i * OutputCount + o];

                LegacyFlatWeights = [];
                LegacyRows = 0;
                LegacyCols = 0;
            }

            if (Weights.Length != InputCount * OutputCount || Biases.Length != OutputCount)
                throw new InvalidDataException(
                    $"Synapse shape is inconsistent: {InputCount}x{OutputCount} with {Weights.Length} weights and {Biases.Length} biases.");
        }
    }
}
