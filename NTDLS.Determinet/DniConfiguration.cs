using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    public class DniConfiguration
    {
        public int InputNodes { get; private set; }
        public DniConfigurationLayer? OutputLayer { get; private set; }

        public List<DniConfigurationLayer> Layers { get; set; } = new();

        public void AddInputLayer(int nodes)
        {
            if (InputNodes != 0)
            {
                throw new Exception("Input layer is already defined.");
            }
            InputNodes = nodes;
        }

        public void AddHiddenLayer(int nodes, DniActivationType activationType)
        {
            Layers.Add(new(DniLayerType.Intermediate, nodes, activationType));
        }

        public void AddOutputLayer(int nodes, DniActivationType activationType)
        {
            if (OutputLayer != null)
            {
                throw new Exception("Output layer is already defined.");
            }
            OutputLayer = new(DniLayerType.Output, nodes, activationType);
        }
    }
}
