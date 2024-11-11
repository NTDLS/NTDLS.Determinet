namespace NTDLS.Determinet
{
    internal class DniStateOfBeing
    {
        public List<DniLayer> Layers { get; set; } = new();
        public List<DniSynapse> Synapses { get; set; } = new();
    }
}
