namespace TestHarness.Library
{
    public class TrainingSample(string fileName, string expectedValue, double[] expectation)
    {
        public string FileName { get; set; } = fileName;
        public double[] Expectation { get; set; } = expectation;
        public string ExpectedValue { get; set; } = expectedValue;
        public int Epoch { get; set; } = 0;
    }
}
