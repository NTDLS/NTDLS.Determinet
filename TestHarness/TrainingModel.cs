namespace TestHarness
{
    public class TrainingModel(double[] input, double[] expectation)
    {
        public double[] Input { get; set; } = input;
        public double[] Expectation { get; set; } = expectation;
        public int Epoch { get; set; } = 0;
    }
}
