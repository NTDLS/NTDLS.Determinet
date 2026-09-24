using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

namespace NTDLS.Determinet.Tests
{
    public class ActivationFunctionTests
    {
        public static readonly TheoryData<DniActivationType> ElementWiseTypes = new(
            Enum.GetValues<DniActivationType>()
                .Where(t => t is not DniActivationType.None and not DniActivationType.SoftMax and not DniActivationType.SimpleSoftMax));

        private static IDniActivationFunction Create(DniActivationType type, DniNamedParameterCollection? param = null)
            => new DniLayer(DniLayerType.Output, 1, type, param ?? new(), null).ActivationFunction!;

        private static double Apply(IDniActivationFunction f, double x) => f.Activation([x])[0];

        [Theory]
        [MemberData(nameof(ElementWiseTypes))]
        public void Derivative_matches_finite_difference(DniActivationType type)
        {
            var f = Create(type);
            const double h = 1e-6;

            for (double x = -6.0; x <= 6.0; x += 0.0937) // Step chosen to avoid landing on kinks (0, ±1, ±2.5).
            {
                double numeric = (Apply(f, x + h) - Apply(f, x - h)) / (2 * h);
                double analytic = f.Derivative(x);
                Assert.True(Math.Abs(numeric - analytic) < 1e-6 * Math.Max(1, Math.Abs(numeric)),
                    $"{type} at x={x}: analytic {analytic}, numeric {numeric}");
            }
        }

        [Theory]
        [MemberData(nameof(ElementWiseTypes))]
        public void Outputs_and_derivatives_are_finite_for_extreme_inputs(DniActivationType type)
        {
            var f = Create(type);
            double[] extremes = [-1e6, -1000, -710, -50, 0, 50, 710, 1000, 1e6];

            var outputs = f.Activation(extremes);
            for (int i = 0; i < extremes.Length; i++)
            {
                Assert.True(double.IsFinite(outputs[i]), $"{type}({extremes[i]}) = {outputs[i]}");
                Assert.True(double.IsFinite(f.Derivative(extremes[i])), $"{type}'({extremes[i]}) = {f.Derivative(extremes[i])}");
            }
        }

        [Theory]
        [MemberData(nameof(ElementWiseTypes))]
        public void Activation_does_not_modify_input(DniActivationType type)
        {
            double[] input = [-2, -0.5, 0.25, 3];
            var copy = (double[])input.Clone();
            Create(type).Activation(input);
            Assert.Equal(copy, input);
        }

        [Fact]
        public void Known_values()
        {
            Assert.Equal(0.5, Apply(Create(DniActivationType.Sigmoid), 0), 12);
            Assert.Equal(Math.Log(2), Apply(Create(DniActivationType.SoftPlus), 0), 12);
            Assert.Equal(1000, Apply(Create(DniActivationType.SoftPlus), 1000), 9);
            Assert.Equal(1000, Apply(Create(DniActivationType.Mish), 1000), 9);
            Assert.Equal(-0.01, Apply(Create(DniActivationType.LeakyReLU), -1), 12);
            Assert.Equal(1.0507009873554805 * 2, Apply(Create(DniActivationType.SELU), 2), 12);
        }

        [Fact]
        public void PiecewiseLinear_is_continuous_at_range_edges()
        {
            var param = new DniNamedParameterCollection();
            param.Set(Piecewise.Alpha, 0.1);
            param.Set(Piecewise.Range, new DniRange(-2, 3));
            var f = Create(DniActivationType.PiecewiseLinear, param);

            foreach (double edge in new[] { -2.0, 3.0 })
            {
                Assert.Equal(Apply(f, edge - 1e-9), Apply(f, edge + 1e-9), 6);
            }

            Assert.Equal(3 + 0.1 * 7, Apply(f, 10), 12);
            Assert.Equal(-2 + 0.1 * -8, Apply(f, -10), 12);
            Assert.Equal(1.5, Apply(f, 1.5), 12);
        }

        [Theory]
        [InlineData(DniActivationType.SoftMax)]
        [InlineData(DniActivationType.SimpleSoftMax)]
        public void SoftMax_is_a_stable_probability_distribution(DniActivationType type)
        {
            var f = Create(type);

            foreach (var logits in new double[][] { [1, 2, 3], [1000, 999, -1000], [-1e6, -1e6 + 1, -1e6 + 2], [0] })
            {
                var p = f.Activation(logits);
                Assert.All(p, v => Assert.True(double.IsFinite(v) && v >= 0 && v <= 1));
                Assert.Equal(1.0, p.Sum(), 12);
            }

            // Shift invariance: softmax(x + c) == softmax(x), even for very large logits (no clamping distortion).
            var a = f.Activation([1, 2, 3]);
            var b = f.Activation([1001, 1002, 1003]);
            for (int i = 0; i < a.Length; i++)
                Assert.Equal(a[i], b[i], 12);
        }

        [Fact]
        public void SoftMax_temperature_scales_logits()
        {
            var param = new DniNamedParameterCollection();
            param.Set(SoftMax.Temperature, 2.0);
            var hot = Create(DniActivationType.SoftMax, param).Activation([2, 4, 6]);
            var reference = Create(DniActivationType.SimpleSoftMax).Activation([1, 2, 3]);
            for (int i = 0; i < hot.Length; i++)
                Assert.Equal(reference[i], hot[i], 12);
        }

        [Fact]
        public void SoftMax_rejects_non_positive_temperature()
        {
            var param = new DniNamedParameterCollection();
            param.Set(SoftMax.Temperature, 0.0);
            Assert.Throws<ArgumentOutOfRangeException>(() => Create(DniActivationType.SoftMax, param));
        }

        [Theory]
        [InlineData(DniActivationType.SoftMax)]
        [InlineData(DniActivationType.SimpleSoftMax)]
        public void SoftMax_is_rejected_on_hidden_layers(DniActivationType type)
        {
            var config = new DniConfiguration();
            config.AddInputLayer(2);
            config.AddIntermediateLayer(3, type);
            config.AddOutputLayer(2, DniActivationType.SoftMax);
            Assert.Throws<ArgumentException>(() => new DniNeuralNetwork(config));
        }
    }
}
