using ProtoBuf;

namespace NTDLS.Determinet
{
    /// <summary>
    /// Represents a synapse structure with weights and biases, supporting serialization and deserialization.
    /// </summary>
    /// <remarks>This class is designed to store and manage a two-dimensional array of weights and a
    /// one-dimensional array of biases. It provides methods to prepare the data for serialization by flattening the
    /// weights and to rebuild the two-dimensional weights array after deserialization. The class is compatible with
    /// Protocol Buffers for efficient serialization.</remarks>
    [ProtoContract]
    public class DniSynapse
    {
        // Backing data used at runtime (fast access)
        [ProtoIgnore]
        public double[,] Weights { get; set; }

        [ProtoMember(1)]
        public int Rows { get; set; }

        [ProtoMember(2)]
        public int Cols { get; set; }

        // Only used during serialization/deserialization
        [ProtoMember(3)]
        public double[] FlatWeights { get; set; } = Array.Empty<double>();

        [ProtoMember(4)]
        public double[] Biases { get; set; } = Array.Empty<double>();

        /// <summary>
        /// Called before saving to flatten the runtime weights.
        /// </summary>
        public void PrepareForSerialization()
        {
            if (Weights == null)
                return;

            Rows = Weights.GetLength(0);
            Cols = Weights.GetLength(1);
            FlatWeights = new double[Rows * Cols];
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    FlatWeights[i * Cols + j] = Weights[i, j];
        }

        /// <summary>
        /// Called after loading to rebuild the 2D array once.
        /// </summary>
        public void RebuildAfterDeserialization()
        {
            Weights = new double[Rows, Cols];
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Weights[i, j] = FlatWeights[i * Cols + j];
        }

        public DniSynapse(double[,] weights, double[] biases)
        {
            Weights = weights;
            Biases = biases;
        }

        public DniSynapse()
        {
            Weights = new double[0, 0];
            Biases = [];
        }
    }
}
