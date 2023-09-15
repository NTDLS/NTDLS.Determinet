using System;
using Newtonsoft.Json;
using NTDLS.Determinet.Types;
using System.Collections.Generic;
using System.Linq;

namespace NTDLS.Determinet
{
    [Serializable]
    public class DniNeuralNetworkLayers
    {
        [JsonIgnore]
        public DniNeuralNetwork Network
        {
            get
            {
                DniUtility.EnsureNotNull(_network);
                return _network;
            }
            set { _network = value; }
        }
        [JsonIgnore]
        private DniNeuralNetwork? _network = null;

        [JsonProperty]
        internal List<DniNeuralNetworkLayer> Collection { get; private set; } = new();

        [JsonProperty]
        internal int Count => Collection.Count;

        [JsonIgnore]
        internal DniNeuralNetworkLayer Input => Collection.First();

        [JsonIgnore]
        internal DniNeuralNetworkLayer Output => Collection.Last();

        #region IEnumerable.

        [JsonIgnore]
        public DniNeuralNetworkLayer this[int index]
        {
            get
            {
                if (index < 0 || index >= Collection.Count)
                {
                    throw new IndexOutOfRangeException("Index is out of range.");
                }
                return Collection[index];
            }
        }

        #endregion

        public DniNeuralNetworkLayers()
        {
        }

        public DniNeuralNetworkLayers(DniNeuralNetwork network)
        {
            Network = network;
        }

        public void AddInput(ActivationType activationType, int nodesCount, DniNamedFunctionParameters? param = null)
        {
            Collection.Add(new DniNeuralNetworkLayer(Network.Layers, LayerType.Input, nodesCount, activationType, param, null));
        }

        public void AddInput(ActivationType activationType, object[] neuronAliases, DniNamedFunctionParameters? param = null)
        {
            Collection.Add(new DniNeuralNetworkLayer(Network.Layers, LayerType.Input, neuronAliases.Length, activationType, param, neuronAliases));
        }

        public void AddIntermediate(ActivationType activationType, int nodesCount, DniNamedFunctionParameters? param = null)
        {
            Collection.Add(new DniNeuralNetworkLayer(Network.Layers, LayerType.Intermediate, nodesCount, activationType, param, null));
        }

        public void AddOutput(int nodesCount, DniNamedFunctionParameters? param = null)
        {
            Collection.Add(new DniNeuralNetworkLayer(Network.Layers, LayerType.Output, nodesCount, ActivationType.None, param, null));
        }

        public void AddOutput(object[] neuronAliases, DniNamedFunctionParameters? param = null)
        {
            Collection.Add(new DniNeuralNetworkLayer(Network.Layers, LayerType.Output, neuronAliases.Length, ActivationType.None, param, neuronAliases));
        }

        public void AddOutput(ActivationType activationType, int nodesCount, DniNamedFunctionParameters? param = null)
        {
            Collection.Add(new DniNeuralNetworkLayer(Network.Layers, LayerType.Output, nodesCount, activationType, param, null));
        }

        public void AddOutput(ActivationType activationType, object[] neuronAliases, DniNamedFunctionParameters? param = null)
        {
            Collection.Add(new DniNeuralNetworkLayer(Network.Layers, LayerType.Output, neuronAliases.Length, activationType, param, neuronAliases));
        }

        #region Genetic.

        public DniNeuralNetworkLayers Clone(DniNeuralNetwork clonedNetwork)
        {
            var clonedLayers = new DniNeuralNetworkLayers(clonedNetwork);
            foreach (var layer in Collection)
            {
                var clonedLayer = layer.Clone(clonedLayers);

                clonedLayers.Collection.Add(clonedLayer);
            }

            return clonedLayers;
        }

        public void Mutate(double mutationProbability, double mutationSeverity)
        {
            foreach (var layer in Collection)
            {
                layer.Mutate(mutationProbability, mutationSeverity);
            }
        }

        #endregion
    }
}
