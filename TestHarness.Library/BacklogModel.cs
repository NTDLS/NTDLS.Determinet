using NTDLS.Determinet.Types;

namespace TestHarness.Library
{
    public class BacklogModel
    {
        public TrainingModel Model { get; set; }
        public double[] Bits { get; set; }

        public BacklogModel(TrainingModel model)
        {
            Model = model;
            Bits = ImageUtility.GetImageGrayscaleBytes(model.FileName, Constants.ImageWidth, Constants.ImageHeight, new DniRange<int>(-5, 5), new DniRange<int>(-3, 3), new DniRange<int>(4, 8))
                ?? throw new Exception($"Failed to load image: {model.FileName}");
        }
    }
}
