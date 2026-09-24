using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;
using static NTDLS.Determinet.Tests.TestNetworks;

namespace NTDLS.Determinet.Tests
{
    /// <summary>
    /// Verifies backpropagation end-to-end: the parameter change produced by a real training step (SGD, lr = 1, no
    /// decay, no clipping) must equal minus the gradient of the loss computed by central finite differences.
    /// </summary>
    public class GradientTests
    {
        public static readonly TheoryData<DniActivationType> HiddenTypes = new(
            Enum.GetValues<DniActivationType>()
                .Where(t => t is not DniActivationType.SoftMax and not DniActivationType.SimpleSoftMax));

        public static readonly TheoryData<DniActivationType, double> OutputTypes = new()
        {
            { DniActivationType.SoftMax, 1.0 },
            { DniActivationType.SoftMax, 2.5 },
            { DniActivationType.SoftMax, 0.4 },
            { DniActivationType.SimpleSoftMax, 1.0 },
            { DniActivationType.Sigmoid, 1.0 },
            { DniActivationType.Tanh, 1.0 },
            { DniActivationType.Identity, 1.0 },
            { DniActivationType.None, 1.0 },
        };

        [Theory]
        [MemberData(nameof(HiddenTypes))]
        public void Hidden_activation_gradients_are_exact(DniActivationType hiddenType)
        {
            var dni = Build(4, [6, 5], hiddenType, 3, DniActivationType.SoftMax);
            AssertGradientsMatch(dni, softMaxTarget: true, seed: (int)hiddenType);
        }

        [Theory]
        [MemberData(nameof(OutputTypes))]
        public void Output_loss_gradients_are_exact(DniActivationType outputType, double temperature)
        {
            var dni = Build(4, [6], DniActivationType.Tanh, 3, outputType, temperature: temperature);
            AssertGradientsMatch(dni, softMaxTarget: outputType is DniActivationType.SoftMax or DniActivationType.SimpleSoftMax, seed: 7);
        }

        [Theory]
        [InlineData(DniActivationType.LeakyReLU)]
        [InlineData(DniActivationType.Tanh)]
        [InlineData(DniActivationType.Sigmoid)]
        [InlineData(DniActivationType.None)]
        public void Layer_norm_gradients_are_exact(DniActivationType hiddenType)
        {
            var dni = Build(4, [6, 5], hiddenType, 3, DniActivationType.SoftMax, layerNorm: true);

            // Move gamma/beta away from their (1, 0) initialization so the check exercises the general case.
            var random = new Random(3);
            foreach (var layer in dni.Layers.Where(l => l.UsesLayerNorm))
            {
                for (int i = 0; i < layer.Gamma!.Length; i++)
                {
                    layer.Gamma[i] = 0.5 + random.NextDouble();
                    layer.Beta![i] = random.NextDouble() - 0.5;
                }
            }

            AssertGradientsMatch(dni, softMaxTarget: true, seed: 11);
        }

        [Fact]
        public void Soft_targets_that_do_not_sum_to_one_are_handled()
        {
            var dni = Build(3, [4], DniActivationType.Tanh, 3, DniActivationType.SoftMax);
            ConfigureForExactGradient(dni);
            var random = new Random(5);
            var inputs = RandomVector(random, 3);
            double[] expected = [0.2, 0.9, 0.3];
            AssertGradientsMatch(dni, inputs, expected);
        }

        [Fact]
        public void TrainBatch_applies_the_mean_gradient_of_its_samples()
        {
            var random = new Random(21);
            var samples = Enumerable.Range(0, 5)
                .Select(i => (inputs: RandomVector(random, 4), expected: OneHot(3, i % 3)))
                .ToList();

            var batchNet = Build(4, [6], DniActivationType.LeakyReLU, 3, DniActivationType.SoftMax);
            ConfigureForExactGradient(batchNet);
            var before = Snapshot(batchNet);

            batchNet.TrainBatch(samples);
            var batchDelta = Delta(before, ParameterArrays(batchNet));

            // Sum of per-sample gradients, each measured from the same starting point.
            var summed = before.Select(a => new double[a.Length]).ToList();
            foreach (var (inputs, expected) in samples)
            {
                var net = Clone(batchNet, before);
                ConfigureForExactGradient(net);
                net.Train(inputs, expected);
                var d = Delta(before, ParameterArrays(net));
                for (int a = 0; a < d.Count; a++)
                    for (int i = 0; i < d[a].Length; i++)
                        summed[a][i] += d[a][i];
            }

            for (int a = 0; a < summed.Count; a++)
                for (int i = 0; i < summed[a].Length; i++)
                    Assert.Equal(summed[a][i] / samples.Count, batchDelta[a][i], 12);
        }

        [Fact]
        public void Gradient_clipping_limits_the_global_norm()
        {
            var dni = Build(4, [6], DniActivationType.Tanh, 3, DniActivationType.Identity);
            ConfigureForExactGradient(dni);
            dni.Parameters.Set(Network.GradientClip, 0.01);

            var before = Snapshot(dni);
            dni.Train([5, -5, 5, -5], [100, -100, 100]);
            var delta = Delta(before, ParameterArrays(dni));

            double norm = Math.Sqrt(delta.Sum(d => d.Sum(v => v * v)));
            Assert.Equal(0.01, norm, 9);
        }

        [Fact]
        public void Mismatched_expected_length_is_rejected()
        {
            var dni = Build(2, [3], DniActivationType.ReLU, 2, DniActivationType.SoftMax);
            Assert.Throws<ArgumentException>(() => dni.Train([1, 2], [1]));
            Assert.Throws<ArgumentException>(() => dni.Train([1, 2], [1, 0, 0]));
            Assert.Throws<ArgumentException>(() => dni.Train([1], [1, 0]));
        }

        #region Helpers.

        private static void AssertGradientsMatch(DniNeuralNetwork dni, bool softMaxTarget, int seed)
        {
            ConfigureForExactGradient(dni);
            var random = new Random(seed);
            int outputs = dni.Layers.Last().NodeCount;
            var inputs = RandomVector(random, dni.Layers[0].NodeCount, 2.0);
            var expected = softMaxTarget ? OneHot(outputs, 1) : RandomVector(random, outputs, 0.9);
            AssertGradientsMatch(dni, inputs, expected);
        }

        private static void AssertGradientsMatch(DniNeuralNetwork dni, double[] inputs, double[] expected)
        {
            const double h = 1e-6;
            var parameters = ParameterArrays(dni);
            var before = Snapshot(dni);

            // Numerical gradient by central differences of the loss. Where the one-sided slopes disagree, the
            // perturbation straddles a kink (e.g. a ReLU input of exactly 0), the loss is not differentiable there,
            // and any subgradient is valid, so that parameter is skipped (NaN).
            double center = dni.ComputeLoss(inputs, expected);
            var numeric = before.Select(a => new double[a.Length]).ToList();
            int total = 0, skipped = 0;
            for (int a = 0; a < parameters.Count; a++)
            {
                for (int i = 0; i < parameters[a].Length; i++)
                {
                    double original = parameters[a][i];
                    parameters[a][i] = original + h;
                    double plus = dni.ComputeLoss(inputs, expected);
                    parameters[a][i] = original - h;
                    double minus = dni.ComputeLoss(inputs, expected);
                    parameters[a][i] = original;

                    double forward = (plus - center) / h, backward = (center - minus) / h;
                    bool kink = Math.Abs(forward - backward) > 1e-3 * Math.Max(1e-3, Math.Abs(forward) + Math.Abs(backward));
                    numeric[a][i] = kink ? double.NaN : (plus - minus) / (2 * h);
                    total++;
                    if (kink) skipped++;
                }
            }
            Assert.True(skipped <= total / 5, $"{skipped} of {total} parameters sit on a kink; the check is not meaningful.");

            // Analytic gradient as applied by a real training step (lr = 1 => delta = -gradient).
            dni.Train(inputs, expected);
            var analytic = Delta(before, ParameterArrays(dni)).Select(d => d.Select(v => -v).ToArray()).ToList();

            for (int a = 0; a < numeric.Count; a++)
            {
                for (int i = 0; i < numeric[a].Length; i++)
                {
                    double n = numeric[a][i], g = analytic[a][i];
                    if (double.IsNaN(n))
                        continue;
                    Assert.True(Math.Abs(n - g) <= 1e-7 + 1e-5 * Math.Max(Math.Abs(n), Math.Abs(g)),
                        $"Parameter array {a}, index {i}: analytic {g}, numeric {n}");
                }
            }
        }

        private static List<double[]> Delta(List<double[]> before, List<double[]> after)
            => before.Select((b, a) => b.Select((v, i) => after[a][i] - v).ToArray()).ToList();

        private static DniNeuralNetwork Clone(DniNeuralNetwork source, List<double[]> parameters)
        {
            using var stream = new MemoryStream();
            source.Save(stream);
            stream.Position = 0;
            var clone = DniNeuralNetwork.Load(stream);
            var arrays = ParameterArrays(clone);
            for (int a = 0; a < arrays.Count; a++)
                Array.Copy(parameters[a], arrays[a], arrays[a].Length);
            return clone;
        }

        #endregion
    }
}
