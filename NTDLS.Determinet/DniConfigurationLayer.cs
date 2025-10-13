using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    /// <summary>
    /// Represents the configuration for a layer in a neural network, including its type, number of nodes, activation
    /// function, and additional parameters.
    /// </summary>
    /// <remarks>This class is used to define the structure and behavior of a specific layer in a neural
    /// network.  It includes properties for specifying the layer type, the number of nodes, the activation function, 
    /// and any additional parameters required for the layer's configuration.</remarks>
    public class DniConfigurationLayer(DniLayerType layerType, int nodes, DniActivationType activationType,
        DniNamedParameterCollection activationParameters)
    {
        public int Nodes { get; set; } = nodes;
        public DniActivationType ActivationType { get; set; } = activationType;
        public DniLayerType LayerType { get; set; } = layerType;
        public DniNamedParameterCollection Parameters { get; set; } = activationParameters;
    }
}
