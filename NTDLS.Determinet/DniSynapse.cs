namespace NTDLS.Determinet
{
    public class DniSynapse
    {
        public double[,] Weights { get; set; }
        public double[] Biases { get; set; }

        public DniSynapse(double[,] weights, double[] biases)
        {
            Weights = weights;
            Biases = biases;
        }
    }
}
