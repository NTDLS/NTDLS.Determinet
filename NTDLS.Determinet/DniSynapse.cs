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
        /// <summary>
        /// Gets or sets the weight matrix used for calculations or transformations.
        /// </summary>
        /// <remarks>The weight matrix is typically used in mathematical or computational operations where
        /// a two-dimensional structure is required. Ensure that the matrix is properly initialized before use to avoid
        /// runtime errors.</remarks>
        [ProtoIgnore]
        public double[,] Weights { get; set; }

        /// <summary>
        /// Gets or sets the number of rows associated with the current instance.
        /// </summary>
        [ProtoMember(1)]
        public int Rows { get; set; }

        /// <summary>
        /// Gets or sets the number of columns in the data structure.
        /// </summary>
        [ProtoMember(2)]
        public int Cols { get; set; }

        /// <summary>
        /// Gets or sets the flattened array of weights used for serialization and deserialization.
        /// </summary>
        [ProtoMember(3)]
        public double[] FlatWeights { get; set; } = Array.Empty<double>();

        /// <summary>
        /// Gets or sets the bias values used in the computation. 
        /// </summary>
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

        /// <summary>
        /// Initializes a new instance of the <see cref="DniSynapse"/> class with the specified weights and biases.
        /// </summary>
        /// <remarks>The dimensions of the <paramref name="weights"/> and <paramref name="biases"/> arrays
        /// must align with the structure of the neural network layer that this synapse represents. Ensure that the
        /// arrays are properly initialized before passing them to this constructor.</remarks>
        /// <param name="weights">A two-dimensional array representing the weights of the synapse. Each element corresponds to the weight
        /// between two connected nodes.</param>
        /// <param name="biases">A one-dimensional array representing the biases of the synapse. Each element corresponds to the bias for a
        /// specific node.</param>
        public DniSynapse(double[,] weights, double[] biases)
        {
            Weights = weights;
            Biases = biases;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DniSynapse"/> class.
        /// </summary>
        /// <remarks>This constructor initializes the <see cref="Weights"/> property as an empty
        /// two-dimensional array  and the <see cref="Biases"/> property as an empty collection. These properties can be
        /// configured  after instantiation to define the synapse's behavior.</remarks>
        public DniSynapse()
        {
            Weights = new double[0, 0];
            Biases = [];
        }
    }
}
