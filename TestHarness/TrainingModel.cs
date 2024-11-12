namespace TestHarness
{
    public class TrainingModel(string fileName, double[] expectation)
    {
        public string FileName { get; set; } = fileName;
        public double[] Expectation { get; set; } = expectation;
        public int Epoch { get; set; } = 0;
    }
}
