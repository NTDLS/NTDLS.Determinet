using NTDLS.Determinet.Types;

namespace TestHarness.Library
{
    public class PreparedTrainingSample
    {
        public TrainingSample Sample { get; set; }
        public double[] Bits { get; set; }

        public PreparedTrainingSample(TrainingSample sample)
        {
            Sample = sample;
            Bits = ImageUtility.GetImageGrayscaleBytes(sample.FileName, Constants.ImageWidth, Constants.ImageHeight, new DniRange<int>(-5, 5), new DniRange<int>(-3, 3), new DniRange<int>(4, 8))
                ?? throw new Exception($"Failed to load sample image: {sample.FileName}");
        }
    }
}
