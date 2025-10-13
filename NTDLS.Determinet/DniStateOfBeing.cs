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
        /// The most recent loss value recorded during training.
        /// </summary>
        [ProtoMember(1)] public double MostRecentLoss { get; internal set; } = new();
        [ProtoMember(2)] public List<DniLayer> Layers { get; internal set; } = new();
        [ProtoMember(3)] public List<DniSynapse> Synapses { get; internal set; } = new();
    }
}
