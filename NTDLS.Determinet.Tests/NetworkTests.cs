using NTDLS.Determinet.Types;
using ProtoBuf;
using System.Globalization;
using System.IO.Compression;
using static NTDLS.Determinet.DniParameters;
using static NTDLS.Determinet.Tests.TestNetworks;

namespace NTDLS.Determinet.Tests
{
    public class NetworkTests
    {
        /// <summary>
        /// Four well-separated Gaussian clusters in 2D (one per class), plus a non-linear XOR-style twist on the inputs.
        /// </summary>
        private static List<(double[] inputs, double[] expected)> ClusterData(int count, int seed)
        {
            var random = new Random(seed);
            double[][] centers = [[1, 1], [-1, 1], [-1, -1], [1, -1]];
            return Enumerable.Range(0, count).Select(i =>
            {
                int c = i % 4;
                double x = centers[c][0] + (random.NextDouble() - 0.5) * 0.6;
                double y = centers[c][1] + (random.NextDouble() - 0.5) * 0.6;
                return (new[] { x, y, x * y }, OneHot(4, c));
            }).ToList();
        }

        private static double Accuracy(DniNeuralNetwork dni, List<(double[] inputs, double[] expected)> data)
            => data.Count(s => dni.Forward(s.inputs).IndexOfMaxValue(out _) == s.expected.IndexOfMaxValue(out _)) / (double)data.Count;

        [Theory]
        [InlineData(false, false)]
        [InlineData(true, false)]
        [InlineData(false, true)]
        [InlineData(true, true)]
        public void Network_learns_a_classification_task(bool useAdam, bool layerNorm)
        {
            var data = ClusterData(400, 1);
            var dni = Build(3, [16, 8], DniActivationType.LeakyReLU, 4, DniActivationType.SoftMax, layerNorm);
            dni.Parameters.Set(Network.UseAdamOptimization, useAdam);
            dni.Parameters.Set(Network.LearningRate, useAdam ? 0.01 : 0.05);

            double initialLoss = data.Average(s => dni.ComputeLoss(s.inputs, s.expected));

            for (int epoch = 0; epoch < 30; epoch++)
            {
                for (int b = 0; b < data.Count; b += 10)
                    dni.TrainBatch(data.Skip(b).Take(10));
            }

            double finalLoss = data.Average(s => dni.ComputeLoss(s.inputs, s.expected));
            Assert.True(finalLoss < initialLoss * 0.2, $"Loss went from {initialLoss} to {finalLoss}.");
            Assert.True(Accuracy(dni, ClusterData(200, 2)) > 0.95);
        }

        [Fact]
        public void Train_reports_loss_before_the_update()
        {
            var dni = Build(3, [5], DniActivationType.Tanh, 4, DniActivationType.SoftMax);
            double[] inputs = [0.1, -0.2, 0.3];
            var expected = OneHot(4, 2);
            double lossBefore = dni.ComputeLoss(inputs, expected);
            Assert.Equal(lossBefore, dni.Train(inputs, expected), 12);
            Assert.Equal(lossBefore, dni.Parameters.Get<double>(Network.ComputedLoss), 12);
        }

        [Fact]
        public void Forward_does_not_modify_the_network_and_is_thread_safe()
        {
            var dni = Build(8, [32, 16], DniActivationType.ReLU, 4, DniActivationType.SoftMax, layerNorm: true);
            var random = new Random(4);
            var inputs = Enumerable.Range(0, 64).Select(_ => RandomVector(random, 8)).ToArray();
            var sequential = inputs.Select(dni.Forward).ToArray();
            var before = Snapshot(dni);

            var parallel = new double[inputs.Length][];
            Parallel.For(0, inputs.Length * 20, i => parallel[i % inputs.Length] = dni.Forward(inputs[i % inputs.Length]));

            for (int i = 0; i < inputs.Length; i++)
                Assert.Equal(sequential[i], parallel[i]);
            Assert.Equal(before, Snapshot(dni));
        }

        [Fact]
        public void Forward_output_is_not_aliased_to_input()
        {
            var dni = Build(3, [], DniActivationType.None, 3, DniActivationType.None);
            double[] inputs = [1, 2, 3];
            var output = dni.Forward(inputs);
            output[0] = 999;
            Assert.Equal(1, inputs[0]);
        }

        [Fact]
        public void Initial_biases_are_zero_and_weights_are_scaled_to_fan_in()
        {
            var dni = Build(400, [200], DniActivationType.ReLU, 10, DniActivationType.SoftMax, seed: null);
            Assert.All(dni.Synapses, s => Assert.All(s.Biases, b => Assert.Equal(0.0, b)));

            var w = dni.Synapses[0].Weights;
            double std = Math.Sqrt(w.Select(v => v * v).Average());
            Assert.InRange(std / Math.Sqrt(2.0 / 400), 0.95, 1.05); // He initialization for ReLU.

            var o = dni.Synapses[1].Weights;
            double stdOut = Math.Sqrt(o.Select(v => v * v).Average());
            Assert.InRange(stdOut / Math.Sqrt(2.0 / (200 + 10)), 0.9, 1.1); // Glorot initialization for SoftMax.
        }

        [Fact]
        public void Save_and_load_round_trip_preserves_outputs_and_adam_state()
        {
            var data = ClusterData(40, 3);
            var dni = Build(3, [8], DniActivationType.ELU, 4, DniActivationType.SoftMax, layerNorm: true);
            dni.Parameters.Set(Network.UseAdamOptimization, true);
            dni.Parameters.Set("Epochs", 12);
            dni.TrainBatch(data.Take(20));

            using var stream = new MemoryStream();
            dni.Save(stream);
            stream.Position = 0;
            var loaded = DniNeuralNetwork.Load(stream);

            foreach (var (inputs, _) in data)
                Assert.Equal(dni.Forward(inputs), loaded.Forward(inputs));
            Assert.Equal(12, loaded.Parameters.Get<int>("Epochs"));

            // Continuing training must be identical, which requires the Adam moments and timestep to have persisted.
            dni.TrainBatch(data.Skip(20));
            loaded.TrainBatch(data.Skip(20));
            Assert.Equal(Snapshot(dni), Snapshot(loaded));
        }

        [Fact]
        public void Legacy_files_load_and_produce_the_same_outputs()
        {
            // Mirror of the pre-2.0 on-disk layout: weights stored flat as [input, output].
            var random = new Random(9);
            int inputs = 3, hidden = 4, outputs = 2;
            var w1 = RandomVector(random, inputs * hidden);
            var b1 = RandomVector(random, hidden);
            var w2 = RandomVector(random, hidden * outputs);
            var b2 = RandomVector(random, outputs);

            var legacy = new LegacyState();
            legacy.Parameters.Set(Network.LearningRate, 0.001);
            legacy.Layers.Add(new LegacyLayer { LayerType = DniLayerType.Input, NodeCount = inputs, ActivationType = DniActivationType.None });
            legacy.Layers.Add(new LegacyLayer { LayerType = DniLayerType.Intermediate, NodeCount = hidden, ActivationType = DniActivationType.Tanh });
            legacy.Layers.Add(new LegacyLayer { LayerType = DniLayerType.Output, NodeCount = outputs, ActivationType = DniActivationType.SoftMax, Labels = ["a", "b"] });
            legacy.Synapses.Add(new LegacySynapse { Rows = inputs, Cols = hidden, FlatWeights = w1, Biases = b1 });
            legacy.Synapses.Add(new LegacySynapse { Rows = hidden, Cols = outputs, FlatWeights = w2, Biases = b2 });

            using var stream = new MemoryStream();
            using (var zip = new DeflateStream(stream, CompressionLevel.Fastest, leaveOpen: true))
                Serializer.Serialize(zip, legacy);
            stream.Position = 0;

            var dni = DniNeuralNetwork.Load(stream);
            double[] x = [0.3, -0.7, 0.9];

            var h = new double[hidden];
            for (int j = 0; j < hidden; j++)
            {
                double sum = b1[j];
                for (int i = 0; i < inputs; i++) sum += x[i] * w1[i * hidden + j];
                h[j] = Math.Tanh(sum);
            }
            var logits = new double[outputs];
            for (int j = 0; j < outputs; j++)
            {
                logits[j] = b2[j];
                for (int i = 0; i < hidden; i++) logits[j] += h[i] * w2[i * outputs + j];
            }
            double max = logits.Max();
            var expected = logits.Select(v => Math.Exp(v - max)).ToArray();
            double total = expected.Sum();

            var actual = dni.Forward(x, out var labels);
            for (int j = 0; j < outputs; j++)
                Assert.Equal(expected[j] / total, actual[j], 12);
            Assert.Equal(["a", "b"], labels.Keys());
            Assert.Equal(0.001, dni.Parameters.Get<double>(Network.LearningRate));
        }

        [ProtoContract]
        private class LegacyState
        {
            [ProtoMember(1)] public DniNamedParameterCollection Parameters { get; set; } = new();
            [ProtoMember(2)] public List<LegacyLayer> Layers { get; set; } = new();
            [ProtoMember(3)] public List<LegacySynapse> Synapses { get; set; } = new();
        }

        [ProtoContract]
        private class LegacyLayer
        {
            [ProtoMember(1)] public DniLayerType LayerType { get; set; }
            [ProtoMember(2)] public int NodeCount { get; set; }
            [ProtoMember(3)] public DniNamedParameterCollection Parameters { get; set; } = new();
            [ProtoMember(4)] public DniActivationType ActivationType { get; set; }
            [ProtoMember(9)] public string[]? Labels { get; set; }
        }

        [ProtoContract]
        private class LegacySynapse
        {
            [ProtoMember(1)] public int Rows { get; set; }
            [ProtoMember(2)] public int Cols { get; set; }
            [ProtoMember(3)] public double[] FlatWeights { get; set; } = [];
            [ProtoMember(4)] public double[] Biases { get; set; } = [];
        }
    }

    public class ParameterCollectionTests
    {
        [Fact]
        public void Values_round_trip_exactly_regardless_of_culture()
        {
            var original = CultureInfo.CurrentCulture;
            try
            {
                CultureInfo.CurrentCulture = new CultureInfo("de-DE");
                var p = new DniNamedParameterCollection();
                p.Set(Network.LearningRate, 1.234567890123456e-20);
                p.Set(Linear.Range, new DniRange(-1.5, 2.25));
                p.Set("Loss", 1234567.125);

                Assert.DoesNotContain(",", p.Values[Network.LearningRate.Key]);

                var copy = new DniNamedParameterCollection();
                foreach (var kv in p.Values)
                    copy.Values[kv.Key] = kv.Value;

                CultureInfo.CurrentCulture = new CultureInfo("en-US");
                Assert.Equal(1.234567890123456e-20, copy.Get<double>(Network.LearningRate));
                Assert.Equal(new DniRange(-1.5, 2.25), copy.Get<DniRange>(Linear.Range));
                Assert.Equal(1234567.125, copy.Get<double>("Loss"));
            }
            finally
            {
                CultureInfo.CurrentCulture = original;
            }
        }

        [Fact]
        public void Legacy_n17_formatted_values_still_parse()
        {
            var p = new DniNamedParameterCollection();
            p.Values[Network.LearningRate.Key] = "0.00050000000000000";
            p.Values["User.BatchLoss"] = "1,234.50000000000000000";
            Assert.Equal(0.0005, p.Get<double>(Network.LearningRate));
            Assert.Equal(1234.5, p.Get<double>("BatchLoss"));
        }

        [Fact]
        public void A_value_can_be_read_as_a_different_numeric_type()
        {
            var p = new DniNamedParameterCollection();
            p.Set("Epochs", 12);
            Assert.Equal(12, p.Get<int>("Epochs"));
            Assert.Equal(12.0, p.Get<double>("Epochs"));
            Assert.Equal(12, p.Get<int>("Epochs"));
        }

        [Fact]
        public void Defaults_apply_until_set_and_after_remove()
        {
            var p = new DniNamedParameterCollection();
            Assert.Equal(0.01, p.Get<double>(LeakyReLU.Alpha));
            Assert.Equal(42.0, p.Get("Missing", 42.0));
            Assert.Throws<KeyNotFoundException>(() => p.Get<double>("Missing"));

            p.Set(LeakyReLU.Alpha, 0.2);
            Assert.Equal(0.2, p.Get<double>(LeakyReLU.Alpha));

            p.Remove(LeakyReLU.Alpha);
            Assert.Equal(0.01, p.Get<double>(LeakyReLU.Alpha));
        }

        [Fact]
        public void Setting_a_parameter_with_the_wrong_type_throws()
        {
            var p = new DniNamedParameterCollection();
            Assert.Throws<ArgumentException>(() => p.Set(Network.LearningRate, true));
        }
    }
}
