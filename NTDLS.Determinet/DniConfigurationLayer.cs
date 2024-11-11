using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    public class DniConfigurationLayer
    {
        public int Nodes { get; set; }
        public DniActivationType ActivationType { get; set; }
        public DniLayerType LayerType { get; set; }
        public DniNamedFunctionParameters ActivationParameters { get; set; } = new();

        public DniConfigurationLayer(DniLayerType layerType, int nodes, DniActivationType activationType, DniNamedFunctionParameters activationParameters)
        {
            ActivationParameters = activationParameters;
            LayerType = layerType;
            Nodes = nodes;
            ActivationType = activationType;
        }
    }
}
