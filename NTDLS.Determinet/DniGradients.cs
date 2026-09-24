namespace NTDLS.Determinet
{
    /// <summary>
    /// Accumulates loss gradients for every trainable parameter over one or more samples, prior to an optimizer step.
    /// </summary>
    /// <remarks>A sample's weight gradient is the outer product delta ⊗ input, so instead of materializing a dense,
    /// weight-sized gradient matrix (which costs several full passes over memory), the two factor vectors are kept per
    /// sample. The optimizer expands them on the fly in a single fused pass over the weights, and the gradient norm is
    /// computed from dot products of the factors. Biases and layer-norm parameters are small and stored densely.</remarks>
    internal class DniGradients
    {
        /// <summary>
        /// Per synapse, one (delta, input) pair per accumulated sample. The weight gradient is the sum of delta ⊗ input,
        /// i.e. dW[o, i] = sum over samples of delta[o] * input[i].
        /// </summary>
        public List<(double[] Delta, double[] Input)>[] WeightFactors { get; }

        /// <summary>
        /// Bias gradients per synapse.
        /// </summary>
        public double[][] Biases { get; }

        /// <summary>
        /// Layer-norm gamma gradients per layer (null for layers without layer normalization).
        /// </summary>
        public double[]?[] Gamma { get; }

        /// <summary>
        /// Layer-norm beta gradients per layer (null for layers without layer normalization).
        /// </summary>
        public double[]?[] Beta { get; }

        public DniGradients(DniStateOfBeing state)
        {
            WeightFactors = state.Synapses.Select(_ => new List<(double[], double[])>()).ToArray();
            Biases = state.Synapses.Select(s => new double[s.Biases.Length]).ToArray();
            Gamma = state.Layers.Select(l => l.Gamma == null ? null : new double[l.Gamma.Length]).ToArray();
            Beta = state.Layers.Select(l => l.Beta == null ? null : new double[l.Beta.Length]).ToArray();
        }

        /// <summary>
        /// Whether these buffers still match the shape of <paramref name="state"/>.
        /// </summary>
        public bool Matches(DniStateOfBeing state)
            => Biases.Length == state.Synapses.Count
            && Gamma.Length == state.Layers.Count
            && state.Synapses.Select((s, i) => s.Biases.Length == Biases[i].Length).All(x => x)
            && state.Layers.Select((l, i) => l.Gamma?.Length == Gamma[i]?.Length).All(x => x);

        public void Clear()
        {
            foreach (var factors in WeightFactors)
                factors.Clear();

            foreach (var array in DenseArrays())
                Array.Clear(array);
        }

        /// <summary>
        /// Sum of squares over every gradient value, for global-norm clipping.
        /// </summary>
        /// <remarks>For the weights this uses ||sum_b d_b ⊗ a_b||² = sum_b sum_c (d_b · d_c)(a_b · a_c), which costs
        /// O(samples² * (inputs + outputs)) instead of a pass over every weight.</remarks>
        public double SumOfSquares()
        {
            double total = 0.0;

            foreach (var array in DenseArrays())
                total += DniMath.Dot(array, array);

            foreach (var factors in WeightFactors)
            {
                for (int b = 0; b < factors.Count; b++)
                {
                    for (int c = 0; c <= b; c++)
                    {
                        double deltaDot = DniMath.Dot(factors[b].Delta, factors[c].Delta);
                        if (deltaDot != 0)
                        {
                            total += (b == c ? 1.0 : 2.0) * deltaDot * DniMath.Dot(factors[b].Input, factors[c].Input);
                        }
                    }
                }
            }

            // Round-off in the cross terms can leave a true zero slightly negative. (Math.Max still propagates NaN.)
            return Math.Max(0.0, total);
        }

        private IEnumerable<double[]> DenseArrays()
        {
            foreach (var a in Biases) yield return a;
            foreach (var a in Gamma) if (a != null) yield return a;
            foreach (var a in Beta) if (a != null) yield return a;
        }
    }
}
