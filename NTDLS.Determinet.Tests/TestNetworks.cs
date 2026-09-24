using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet.Tests
{
    internal static class TestNetworks
    {
        public static DniNeuralNetwork Build(int inputs, int[] hidden, DniActivationType hiddenType, int outputs,
            DniActivationType outputType, bool layerNorm = false, double temperature = 1.0, int? seed = 1)
        {
            var config = new DniConfiguration { LearningRate = 0.01 };
            config.AddInputLayer(inputs);

            foreach (var nodes in hidden)
            {
                var param = new DniNamedParameterCollection();
                if (layerNorm)
                    param.Set(Layer.UseLayerNorm, true);
                config.AddIntermediateLayer(nodes, hiddenType, param);
            }

            var outputParam = new DniNamedParameterCollection();
            if (outputType == DniActivationType.SoftMax)
                outputParam.Set(SoftMax.Temperature, temperature);
            config.AddOutputLayer(outputs, outputType, outputParam);

            var dni = new DniNeuralNetwork(config);

            if (seed == null)
                return dni;

            // Replace the (non-deterministic) initial weights so every test run is reproducible.
            var random = new Random(seed.Value);
            foreach (var synapse in dni.Synapses)
            {
                double std = Math.Sqrt(2.0 / synapse.InputCount);
                for (int i = 0; i < synapse.Weights.Length; i++)
                {
                    double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                    synapse.Weights[i] = std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                }
            }

            return dni;
        }

        /// <summary>
        /// Every trainable parameter array in the network, in a stable order.
        /// </summary>
        public static List<double[]> ParameterArrays(DniNeuralNetwork dni)
        {
            var arrays = new List<double[]>();
            foreach (var synapse in dni.Synapses)
            {
                arrays.Add(synapse.Weights);
                arrays.Add(synapse.Biases);
            }
            foreach (var layer in dni.Layers)
            {
                if (layer.Gamma != null) arrays.Add(layer.Gamma);
                if (layer.Beta != null) arrays.Add(layer.Beta);
            }
            return arrays;
        }

        public static List<double[]> Snapshot(DniNeuralNetwork dni)
            => ParameterArrays(dni).Select(a => (double[])a.Clone()).ToList();

        public static double[] RandomVector(Random random, int length, double scale = 1.0)
            => Enumerable.Range(0, length).Select(_ => (random.NextDouble() * 2 - 1) * scale).ToArray();

        public static double[] OneHot(int length, int index)
        {
            var v = new double[length];
            v[index] = 1;
            return v;
        }

        /// <summary>
        /// Configures plain SGD with learning rate 1 and no decay/clipping, so a Train step moves every parameter by exactly -gradient.
        /// </summary>
        public static void ConfigureForExactGradient(DniNeuralNetwork dni)
        {
            dni.Parameters.Set(Network.LearningRate, 1.0);
            dni.Parameters.Set(Network.WeightDecay, 0.0);
            dni.Parameters.Set(Network.GradientClip, 0.0);
            dni.Parameters.Set(Network.UseAdamOptimization, false);
        }
    }
}
