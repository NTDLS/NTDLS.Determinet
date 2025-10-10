using ProtoBuf;

namespace NTDLS.Determinet
{
    [ProtoContract]

    internal class DniStateOfBeing
    {
        [ProtoMember(1)] public double LearningRate { get; set; }
        [ProtoMember(2)] public List<DniLayer> Layers { get; set; } = new();
        [ProtoMember(3)] public List<DniSynapse> Synapses { get; set; } = new();
    }
}
