namespace TestHarness.Library
{
    public class PreparedTrainingSample
    {
        public TrainingSample Sample { get; set; }
        public double[] Bits { get; set; }

        public PreparedTrainingSample(TrainingSample sample, double[] bits)
        {
            Sample = sample;
            Bits = bits;
        }
    }
}
