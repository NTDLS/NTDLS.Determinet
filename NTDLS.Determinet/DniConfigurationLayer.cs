using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    public class DniConfigurationLayer
    {
        public int Nodes { get; set; }
        public DniActivationType ActivationType { get; set; }
        public DniLayerType LayerType { get; set; }

        public DniConfigurationLayer(DniLayerType layerType, int nodes, DniActivationType activationType)
        {
            LayerType = layerType;
            Nodes = nodes;
            ActivationType = activationType;
        }
    }
}
