using NTDLS.Determinet;
using NTDLS.Determinet.Types;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace TestHarness
{
    internal class Program
    {
        const int _imageWidth = 28;
        const int _imageHeight = 28;
        const int _outputNodes = 10; //10 values because we are working with digits with 10 possible outputs.

        static void Main()
        {
            var trainedModelFilename = "trained.json";

            File.Delete(trainedModelFilename);

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

            int outputNode = 0;
            double confidence = 0;

            dni.FeedForward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\0\30_00010.png"), out outputNode, out confidence);
            Console.WriteLine($"Expected: 0: Result: {outputNode:n0}, confidence: {confidence:n4}");
            dni.FeedForward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\1\31_00010.png"), out outputNode, out confidence);
            Console.WriteLine($"Expected: 1: Result: {outputNode:n0}, confidence: {confidence:n4}");
            dni.FeedForward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\2\32_00010.png"), out outputNode, out confidence);
            Console.WriteLine($"Expected: 2: Result: {outputNode:n0}, confidence: {confidence:n4}");
            dni.FeedForward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\3\33_00010.png"), out outputNode, out confidence);
            Console.WriteLine($"Expected: 3: Result: {outputNode:n0}, confidence: {confidence:n4}");
            dni.FeedForward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\4\34_00010.png"), out outputNode, out confidence);
            Console.WriteLine($"Expected: 4: Result: {outputNode:n0}, confidence: {confidence:n4}");
            dni.FeedForward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\5\35_00010.png"), out outputNode, out confidence);
            Console.WriteLine($"Expected: 5: Result: {outputNode:n0}, confidence: {confidence:n4}");
            dni.FeedForward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\6\36_00010.png"), out outputNode, out confidence);
            Console.WriteLine($"Expected: 6: Result: {outputNode:n0}, confidence: {confidence:n4}");
            dni.FeedForward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\7\37_00010.png"), out outputNode, out confidence);
            Console.WriteLine($"Expected: 7: Result: {outputNode:n0}, confidence: {confidence:n4}");
            dni.FeedForward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\8\38_00010.png"), out outputNode, out confidence);
            Console.WriteLine($"Expected: 8: Result: {outputNode:n0}, confidence: {confidence:n4}");
            dni.FeedForward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\9\39_00010.png"), out outputNode, out confidence);
            Console.WriteLine($"Expected: 9: Result: {outputNode:n0}, confidence: {confidence:n4}");
        }

        /// <summary>
        /// Gets a random training model from the list that has not yet been fed-in for the current training epoch.
        /// Increments the training epoch for the random model.
        /// </summary>
        private static TrainingModel? GetRandomTrainingModel(List<TrainingModel> models, int epoch)
        {
            var modelsThisEpoch = models.Where(o => o.Epoch == epoch).ToList();

            if (modelsThisEpoch.Count == 0)
            {
                return null;
            }

            int randomIndex = DniUtility.Random.Next(modelsThisEpoch.Count);
            var randomModel = modelsThisEpoch[randomIndex];

            randomModel.Epoch++;

            return randomModel;
        }

        private static List<TrainingModel> LoadTrainingModels(string path)
        {
            var trainingModels = new List<TrainingModel>();

            var digitFolders = Directory.GetDirectories(path);
            foreach (var digitFolder in digitFolders)
            {
                var imagePaths = Directory.GetFiles(digitFolder, "*.png");

                //Use the name of the folder as the expected output character for training.
                var outputCharacter = digitFolder[digitFolder.Length - 1];

                //Create the expected output layer:
                var outputs = new double[_outputNodes];
                outputs[outputCharacter - 48] = 1;

                foreach (var imagePath in imagePaths)
                {
                    var imageBits = GetImageGrayscaleBytes(imagePath, _imageWidth, _imageHeight);
                    trainingModels.Add(new(imageBits, outputs));
                }
            }

            return trainingModels;
        }

        static DniNeuralNetwork TrainAndSave(string trainedModelFilename)
        {
            var configuration = new DniConfiguration();
            configuration.AddInputLayer(_imageWidth * _imageHeight);
            configuration.AddHiddenLayer(280, DniActivationType.Sigmoid);
            configuration.AddOutputLayer(_outputNodes, DniActivationType.Sigmoid);

            var dni = new DniNeuralNetwork(configuration);

            //Add input layer.
            //dni.Layers.AddInput(ActivationType.LeakyReLU, imageWidth * imageHeight);

            //Add a intermediate "hidden" layer. You can add more if you like.
            //dni.Layers.AddIntermediate(ActivationType.LeakyReLU, 128);

            //Add the output layer.
            //dni.Layers.AddOutput(outputNodes); //One node per digit.

            var trainingModels = LoadTrainingModels(@"C:\Users\ntdls\Desktop\digit");

            for (int epoch = 0; epoch < 10; epoch++)
            {
                Console.WriteLine($"Epoch {epoch:n0}.");

                TrainingModel? model;
                while ((model = GetRandomTrainingModel(trainingModels, epoch)) != null)
                {
                    dni.Train(model.Input, model.Expectation);
                }
            }

            Console.WriteLine();

            dni.SaveToFile(trainedModelFilename);

            return dni;
        }

        static double[] GetImageGrayscaleBytes(string imagePath, int resizeWidth = 28, int resizeHeight = 28)
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

            double[] pixelData = new double[resizeWidth * resizeHeight];

            int index = 0;
            for (int y = 0; y < resizedImg.Height; y++)
            {
                for (int x = 0; x < resizedImg.Width; x++)
                {
                    var pixel = resizedImg[x, y];
                    byte grayValue = (byte)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
                    pixelData[index++] = grayValue / 255.0; //Scale the pixel value to 0-1.
                }
            }

            return pixelData;
        }
    }
}
