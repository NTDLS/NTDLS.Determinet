using NTDLS.Determinet;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace TestHarness
{
    internal class Program
    {
        static void Main()
        {
            var trainedModelFilename = "trained.json";

            DniNeuralNetwork dni;
            if (File.Exists(trainedModelFilename))
            {
                dni = DniNeuralNetwork.LoadFromFile(trainedModelFilename)
                    ?? throw new Exception("Failed to load the network from file.");
            }
            else
            {
                dni = TrainAndSave(trainedModelFilename);
            }

            var imageBytes = GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\4\34_00016.png");

            var result = dni.FeedForward(imageBytes.Select(o => (double)o).ToArray());
        }

        static DniNeuralNetwork TrainAndSave(string trainedModelFilename)
        {
            int imageWidth = 28;
            int imageHeight = 28;

            int outputNodes = 10;

            var dni = new DniNeuralNetwork([imageWidth * imageHeight, 128, 10]);


            //Add input layer.
            //dni.Layers.AddInput(ActivationType.LeakyReLU, imageWidth * imageHeight);

            //Add a intermediate "hidden" layer. You can add more if you like.
            //dni.Layers.AddIntermediate(ActivationType.LeakyReLU, 128);

            //Add the output layer.
            //dni.Layers.AddOutput(outputNodes); //One node per digit.

            var digitFolders = Directory.GetDirectories(@"C:\Users\ntdls\Desktop\digit");
            foreach (var digitFolder in digitFolders)
            {
                var imagePaths = Directory.GetFiles(digitFolder, "*.png");

                var outputCharacter = digitFolder[digitFolder.Length - 1]; //Use the name of the folder as the expected output character for training.

                //Create the expected output layer:
                var outputs = new double[outputNodes];
                outputs[outputCharacter - 48] = 1;

                Console.WriteLine($"Training on '{outputCharacter}' with {imagePaths.Length:n0} samples.");

                foreach (var imagePath in imagePaths)
                {
                    var imageBytes = GetImageGrayscaleBytes(imagePath, imageWidth, imageHeight);

                    dni.Train(imageBytes.Select(o => (double)o).ToArray(), outputs);
                }
            }

            Console.WriteLine();

            dni.SaveToFile(trainedModelFilename);

            return dni;
        }

        static byte[] GetImageGrayscaleBytes(string imagePath, int resizeWidth = 28, int resizeHeight = 28)
        {
            var fileBytes = File.ReadAllBytes(imagePath);
            using var img = Image.Load<Rgba32>(new MemoryStream(fileBytes));

            // Define initial bounds
            int left = img.Width, right = 0, top = img.Height, bottom = 0;

            // Loop through pixels to find non-white areas
            for (int y = 0; y < img.Height; y++)
            {
                for (int x = 0; x < img.Width; x++)
                {
                    Rgba32 pixel = img[x, y];
                    if (pixel.R < 255 || pixel.G < 255 || pixel.B < 255) // Adjust for your background color
                    {
                        if (x < left) left = x;
                        if (x > right) right = x;
                        if (y < top) top = y;
                        if (y > bottom) bottom = y;
                    }
                }
            }

            // Calculate the bounding box
            int width = right - left + 1;
            int height = bottom - top + 1;
            var bounds = new Rectangle(left, top, width, height);

            // Crop to the bounding box
            img.Mutate(x => x.Crop(bounds));

            using var resizedImg = img.Clone(context => context.Resize(resizeWidth, resizeHeight));

            var pixelData = new byte[resizeWidth * resizeHeight];

            int index = 0;
            for (int y = 0; y < resizedImg.Height; y++)
            {
                for (int x = 0; x < resizedImg.Width; x++)
                {
                    var pixel = resizedImg[x, y];
                    byte grayValue = (byte)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
                    pixelData[index++] = grayValue;
                }
            }

            return pixelData;
        }
    }
}
