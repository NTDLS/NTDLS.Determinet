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

            //File.Delete(trainedModelFilename);

            DniNeuralNetwork dni;
            if (File.Exists(trainedModelFilename))
            {
                dni = TrainAndSave(trainedModelFilename);

                dni = DniNeuralNetwork.LoadFromFile(trainedModelFilename)
                    ?? throw new Exception("Failed to load the network from file.");
            }
            else
            {
                dni = TrainAndSave(trainedModelFilename);
            }

            int outputNode = 0;
            double confidence = 0;

            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\0\30_00010.png")), out confidence);
            Console.WriteLine($"Expected: 0: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\1\31_00010.png")), out confidence);
            Console.WriteLine($"Expected: 1: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\2\32_00010.png")), out confidence);
            Console.WriteLine($"Expected: 2: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\3\33_00010.png")), out confidence);
            Console.WriteLine($"Expected: 3: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\4\34_00010.png")), out confidence);
            Console.WriteLine($"Expected: 4: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\5\35_00010.png")), out confidence);
            Console.WriteLine($"Expected: 5: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\6\36_00010.png")), out confidence);
            Console.WriteLine($"Expected: 6: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\7\37_00010.png")), out confidence);
            Console.WriteLine($"Expected: 7: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\8\38_00010.png")), out confidence);
            Console.WriteLine($"Expected: 8: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\Users\ntdls\Desktop\digit\9\39_00010.png")), out confidence);
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
                    trainingModels.Add(new(imagePath, outputs));
                }
            }

            return trainingModels;
        }

        static DniNeuralNetwork TrainAndSave(string trainedModelFilename)
        {
            var configuration = new DniConfiguration();
            configuration.AddInputLayer(_imageWidth * _imageHeight);
            configuration.AddIntermediateLayer(280, DniActivationType.LeakyReLU);

            /*//Example of adding parameters for a layer activation function:
            var piecewiseLinearParam = new DniNamedFunctionParameters();
            piecewiseLinearParam.Set("alpha", 1);
            piecewiseLinearParam.Set("range", new DniRange(-10, 10));
            configuration.AddIntermediateLayer(280, DniActivationType.PiecewiseLinear, piecewiseLinearParam);
            */

            configuration.AddOutputLayer(_outputNodes, DniActivationType.SoftMax);

            var dni = new DniNeuralNetwork(configuration);

            Console.WriteLine($"Loading image paths...");
            var trainingModels = LoadTrainingModels(@"C:\Users\ntdls\Desktop\digit");

            double initialLearningRate = 0.006;
            double learningRate = initialLearningRate;
            int patience = 3; // Number of epochs to wait before reducing learning rate
            double decayFactor = 0.5; // Factor to reduce learning rate
            int trainingEpochs = 5;
            double previousEpochLoss = double.MaxValue;
            int patienceCounter = 0;
            double epochLoss = double.PositiveInfinity;

            for (int epoch = 0; epoch < trainingEpochs; epoch++)
            {
                Console.WriteLine($"Epoch {epoch + 1}/{trainingEpochs} - Loss: {epochLoss:n8} - Learning Rate: {learningRate:n8}");

                dni.LearningRate = learningRate;

                TrainingModel? model;
                epochLoss = 0;
                while ((model = GetRandomTrainingModel(trainingModels, epoch)) != null)
                {
                    var imageBits = GetImageGrayscaleBytes(model.FileName, _imageWidth, _imageHeight);
                    epochLoss += dni.Train(imageBits, model.Expectation);
                }

                epochLoss /= trainingModels.Count;

                if (epochLoss > previousEpochLoss) // Check if loss has increased.
                {
                    if (patienceCounter++ >= patience)
                    {
                        // Reduce learning rate if loss has not improved for 'patience' epochs
                        learningRate *= decayFactor;
                        patienceCounter = 0; // Reset patience counter after reducing learning rate
                        Console.WriteLine($"Reduced learning rate to {learningRate:n4}.");
                    }
                }
                else
                {
                    patienceCounter = 0; // Reset counter if loss decreases
                }

                previousEpochLoss = epochLoss;
            }

            Console.WriteLine();

            dni.SaveToFile(trainedModelFilename);

            return dni;
        }


        static double[] GetImageGrayscaleBytes(string imagePath, int resizeWidth = 28, int resizeHeight = 28, int rotationAngleRange = 15)
        {
            var fileBytes = File.ReadAllBytes(imagePath);

            // Load the image in RGB format and convert to RGBA
            using var img = Image.Load<Rgba32>(new MemoryStream(fileBytes));

            // Determine larger canvas size (double the original size to prevent cropping during rotation)
            int largerSize = Math.Max(img.Width, img.Height) * 2;

            // Create a new transparent image with the larger canvas size
            using var canvas = new Image<Rgba32>(largerSize, largerSize, Color.Transparent);

            // Center the original image on this new transparent canvas
            int offsetX = (largerSize - img.Width) / 2;
            int offsetY = (largerSize - img.Height) / 2;
            canvas.Mutate(x => x.DrawImage(img, new Point(offsetX, offsetY), 1.0f));

            // Apply random rotation to the whole canvas
            int randomAngle = DniUtility.Random.Next(-rotationAngleRange, rotationAngleRange);
            canvas.Mutate(x => x.Rotate(randomAngle));

            // Define the bounds for cropping out non-transparent areas
            int left = canvas.Width, right = 0, top = canvas.Height, bottom = 0;
            for (int y = 0; y < canvas.Height; y++)
            {
                for (int x = 0; x < canvas.Width; x++)
                {
                    Rgba32 pixel = canvas[x, y];
                    if (pixel.A > 0) // Find non-transparent pixels
                    {
                        if (x < left) left = x;
                        if (x > right) right = x;
                        if (y < top) top = y;
                        if (y > bottom) bottom = y;
                    }
                }
            }

            // Crop to the bounding box of the rotated image content
            int width = right - left + 1;
            int height = bottom - top + 1;
            var bounds = new Rectangle(left, top, width, height);
            canvas.Mutate(x => x.Crop(bounds));

            // Define initial bounds for cropping non-white areas
            left = canvas.Width;
            right = 0;
            top = canvas.Height;
            bottom = 0;

            // Loop through pixels to find non-white areas
            for (int y = 0; y < canvas.Height; y++)
            {
                for (int x = 0; x < canvas.Width; x++)
                {
                    Rgba32 pixel = canvas[x, y];
                    if (pixel.A > 0 && (pixel.R < 255 || pixel.G < 255 || pixel.B < 255)) // Adjust for your background color
                    {
                        if (x < left) left = x;
                        if (x > right) right = x;
                        if (y < top) top = y;
                        if (y > bottom) bottom = y;
                    }
                }
            }

            // Calculate the bounding box
            width = right - left + 1;
            height = bottom - top + 1;
            bounds = new Rectangle(left, top, width, height);

            // Crop to the bounding box
            canvas.Mutate(x => x.Crop(bounds));

            // Resize to target dimensions
            using var finalResizedImg = canvas.Clone(context => context.Resize(resizeWidth, resizeHeight));

            //finalResizedImg.SaveAsPng(@$"C:\Users\ntdls\Desktop\Test\{Path.GetFileName(imagePath)}");

            // Convert to grayscale and normalize
            double[] pixelData = new double[resizeWidth * resizeHeight];
            int index = 0;
            for (int y = 0; y < finalResizedImg.Height; y++)
            {
                for (int x = 0; x < finalResizedImg.Width; x++)
                {
                    var pixel = finalResizedImg[x, y];
                    byte grayValue = (byte)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
                    pixelData[index++] = grayValue / 255.0; // Scale the pixel value to 0-1.
                }
            }

            return pixelData;
        }
    }
}
