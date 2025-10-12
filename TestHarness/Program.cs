using NTDLS.Determinet;
using NTDLS.Determinet.Types;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using static NTDLS.Determinet.DniParameters;

namespace TestHarness
{
    internal class Program
    {
        const double _initialLearningRate = 0.005;  // Starting learning rate for training.
        const double _convergence = 0.000000001;    // Threshold for considering the training has converged.
        const int _cooldown = 5;                    // epochs to wait after each learning rate decay.
        const int _patience = 5;                    // Number of epochs to wait before reducing learning rate once cost starts increasing or reaches a plateau.
        const double _decayFactor = 0.8;            // Factor to reduce learning rate
        const int _trainingEpochs = 250;            // Total number of training epochs
        const int _imageWidth = 64;                 // Downscale for faster processing with minimal quality loss.
        const int _imageHeight = 64;                // Downscale for faster processing with minimal quality loss.
        const int _outputNodes = 62;                // 10 digits + 26 lowercase + 26 uppercase letters
        const double _minDelta = 0.001;             // minimum improvement threshold
        const int _earlyStopPatience = 10;          // epochs with no improvement before stopping
        const int _earlyStopMinEpochs = 10;         // minimum epochs before we even consider stopping
        const double _minLearningRate = 0.000001;
        const double _slop = 0.000000000001;


        static void Main()
        {
            var trainedModelFilename = "trained.dni";

            //File.Delete(trainedModelFilename);

            var dni = TrainAndSave(trainedModelFilename);

            /*
            int outputNode = 0;
            double confidence = 0;

            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\NTDLS\NTDLS.Determinet\Training Characters\0 (1).png", _imageWidth, _imageHeight)), out confidence);
            Console.WriteLine($"Expected: 0: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\NTDLS\NTDLS.Determinet\Training Characters\1 (1).png", _imageWidth, _imageHeight)), out confidence);
            Console.WriteLine($"Expected: 1: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\NTDLS\NTDLS.Determinet\Training Characters\2 (1).png", _imageWidth, _imageHeight)), out confidence);
            Console.WriteLine($"Expected: 2: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\NTDLS\NTDLS.Determinet\Training Characters\3 (1).png", _imageWidth, _imageHeight)), out confidence);
            Console.WriteLine($"Expected: 3: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\NTDLS\NTDLS.Determinet\Training Characters\4 (1).png", _imageWidth, _imageHeight)), out confidence);
            Console.WriteLine($"Expected: 4: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\NTDLS\NTDLS.Determinet\Training Characters\5 (1).png", _imageWidth, _imageHeight)), out confidence);
            Console.WriteLine($"Expected: 5: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\NTDLS\NTDLS.Determinet\Training Characters\6 (1).png", _imageWidth, _imageHeight)), out confidence);
            Console.WriteLine($"Expected: 6: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\NTDLS\NTDLS.Determinet\Training Characters\7 (1).png", _imageWidth, _imageHeight)), out confidence);
            Console.WriteLine($"Expected: 7: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\NTDLS\NTDLS.Determinet\Training Characters\8 (1).png", _imageWidth, _imageHeight)), out confidence);
            Console.WriteLine($"Expected: 8: Result: {outputNode:n0}, confidence: {confidence:n4}");
            outputNode = DniUtility.GetIndexOfMaxValue(dni.Forward(GetImageGrayscaleBytes(@"C:\NTDLS\NTDLS.Determinet\Training Characters\9 (1).png", _imageWidth, _imageHeight)), out confidence);
            Console.WriteLine($"Expected: 9: Result: {outputNode:n0}, confidence: {confidence:n4}");
            */
        }

        private static List<TrainingModel> LoadTrainingModels(string path)
        {
            var trainingModels = new List<TrainingModel>();

            var imagePaths = Directory.EnumerateFiles(path, "*.png");

            var distinctCharacters = new char[] {
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'a', 'b', 'c', 'd', 'e', 'f','g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                'A', 'B', 'C', 'D', 'E', 'F','G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
            };

            int charIndex = 0;

            var layerExpectations = new Dictionary<char, double[]>();
            foreach (var character in distinctCharacters)
            {
                var outputs = new double[_outputNodes];
                outputs[charIndex++] = 1;
                layerExpectations[character] = outputs;
            }

            foreach (var imagePath in imagePaths)
            {
                var fileName = Path.GetFileName(imagePath);
                //Use the first character of the image filename as the expected output character:
                var layerExpectation = layerExpectations[fileName[0]];
                trainingModels.Add(new(imagePath, layerExpectation));
            }

            return trainingModels;
        }


        private static readonly Lock _lockGetRandomTrainingModel = new();
        /// <summary>
        /// Gets a random training model from the list that has not yet been fed-in for the current training epoch.
        /// Increments the training epoch for the random model.
        /// </summary>
        private static bool TryGetRandomTrainingModel(List<TrainingModel> models, int epoch, [NotNullWhen(true)] out TrainingModel? randomModel)
        {
            lock (_lockGetRandomTrainingModel)
            {
                var modelsThisEpoch = models.Where(o => o.Epoch == epoch).ToList();

                if (modelsThisEpoch.Count == 0)
                {
                    randomModel = null;
                    return false;
                }

                int randomIndex = DniUtility.Random.Next(modelsThisEpoch.Count);
                randomModel = modelsThisEpoch[randomIndex];

                //Once we have consumed a model for this epoch, increment its epoch so we don't use it again this epoch:
                randomModel.Epoch++;

                return true;
            }
        }

        class BacklogModel
        {
            public TrainingModel Model { get; set; }
            public double[] Bits { get; set; }

            public BacklogModel(TrainingModel model)
            {
                Model = model;
                Bits = GetImageGrayscaleBytes(model.FileName, _imageWidth, _imageHeight)
                    ?? throw new Exception($"Failed to load image: {model.FileName}");
            }
        }

        static DniNeuralNetwork TrainAndSave(string trainedModelFilename)
        {
            DniNeuralNetwork dni;

            if (File.Exists(trainedModelFilename))
            {
                dni = DniNeuralNetwork.LoadFromFile(trainedModelFilename)
                    ?? throw new Exception("Failed to load the network from file.");
            }
            else
            {
                var configuration = new DniConfiguration();

                /*
                | Layer    | Role                                     | Notes                                                                              |
                | -------- | ---------------------------------------- | ---------------------------------------------------------------------------------- |
                | Input    | `_imageWidth * _imageHeight` (130x130)   | Flattened grayscale image. Normalize pixel values to [0, 1] or z-score per sample. |
                | Hidden 1 | 512 → LeakyReLU                          | Enough capacity to learn non-linear stroke patterns.                               |
                | Hidden 2 | 256 → LeakyReLU                          | Tapering keeps parameter count reasonable; aids generalization.                    |
                | Hidden 3 | 128 → LeakyReLU                          | Distills features before classification.                                           |
                | Output   | 62 → Softmax                             | One neuron per symbol (0-9, A-Z, a-z).                                             |
                */

                configuration.AddInputLayer(_imageWidth * _imageHeight);

                var leakyReLUParam = new DniNamedFunctionParameters();
                //leakyReLUParam.Set(LayerParameters.UseBatchNorm, true);
                //layerParam.Set(DniParameters.LayerParameters.BatchNormMomentum, 0.9);

                //MLPs: 2–3 hidden layers, 128–512 units each, tapering (512, 256, 128).
                configuration.AddIntermediateLayer(768, DniActivationType.LeakyReLU);
                configuration.AddIntermediateLayer(512, DniActivationType.LeakyReLU, leakyReLUParam);
                configuration.AddIntermediateLayer(256, DniActivationType.LeakyReLU);
                configuration.AddIntermediateLayer(128, DniActivationType.LeakyReLU);

                /*//Example of adding parameters for a layer activation function:
                var piecewiseLinearParam = new DniNamedFunctionParameters();
                piecewiseLinearParam.Set("alpha", 1);
                piecewiseLinearParam.Set("range", new DniRange(-10, 10));
                configuration.AddIntermediateLayer(280, DniActivationType.PiecewiseLinear, piecewiseLinearParam);
                */

                var softMaxParam = new DniNamedFunctionParameters();
                //softMaxParam.Set(LayerParameters.SoftMaxTemperature, 5.5);
                configuration.AddOutputLayer(_outputNodes, DniActivationType.SoftMax, softMaxParam);

                dni = new DniNeuralNetwork(configuration);
            }

            Console.WriteLine($"Loading image paths...");
            var trainingModels = LoadTrainingModels(@"C:\NTDLS\NTDLS.Determinet\Training Characters");

            double learningRate = dni.LearningRate > 0 ? dni.LearningRate : _initialLearningRate;

            double previousEpochLoss = double.MaxValue;
            double bestLoss = double.MaxValue;
            int epochsSinceImprovement = 0;
            int cooldownCounter = _cooldown;

            Console.WriteLine($"Starting backlog threads...");
            ConcurrentStack<BacklogModel> backlogModels = new();

            for (int epoch = 0; epoch < _trainingEpochs; epoch++)
            {
                dni.LearningRate = learningRate;

                double epochLoss = 0;
                int samplesProcessed = 0;
                int threadCount = Environment.ProcessorCount;
                int threadsCompleted = 0;

                for (int i = 0; i < threadCount; i++)
                {
                    Task.Run(() =>
                    {
                        while (TryGetRandomTrainingModel(trainingModels, epoch, out var model))
                        {
                            backlogModels.Push(new BacklogModel(model));

                            while (backlogModels.Count >= 100)
                            {
                                Thread.Sleep(10); // Let the backlog get processed a bit before adding more.
                            }
                        }

                        Interlocked.Increment(ref threadsCompleted);
                    });
                }


                if (epoch == 0)
                {
                    Console.WriteLine($"Training...");
                }

                while (backlogModels.TryPop(out var backlogModel) || threadsCompleted != threadCount)
                {
                    if (backlogModel == null)
                    {
                        Thread.Sleep(10);
                    }
                    else
                    {
                        var loss = dni.Train(backlogModel.Bits, backlogModel.Model.Expectation);
                        epochLoss += loss;
                        //Console.WriteLine($"{loss:n10}");
                        samplesProcessed++;
                    }
                }

                /*
                 Epoch |  Expected Avg. Loss | Interpretation
                 1     |  ~4.6               | Random baseline
                 5     |  ~3–3.5             | Network learning broad class features
                 10    |  ~2                 | Clear improvement
                 20+   |  <1.0               | Network reliably distinguishing characters
                */

                epochLoss /= Math.Max(1, samplesProcessed);

                Console.WriteLine($"Epoch {epoch + 1}/{_trainingEpochs} - Loss: {epochLoss:n8} - Learning Rate: {learningRate:n10}");

                if (epochLoss < bestLoss - _minDelta)
                {
                    //We save every time we get a new best loss.
                    dni.SaveToFile(trainedModelFilename);

                    bestLoss = epochLoss;
                    epochsSinceImprovement = 0;
                }
                else
                {
                    epochsSinceImprovement++;
                }

                #region Learning Rate Scheduler.

                if (cooldownCounter > 0)
                {
                    cooldownCounter--;
                }
                else
                {
                    if (epochsSinceImprovement >= _patience && learningRate > _minLearningRate + _slop)
                    {
                        var newLR = learningRate * _decayFactor;
                        learningRate = Math.Max(_minLearningRate, newLR);
                        cooldownCounter = _cooldown;
                        Console.WriteLine($"[LR Scheduler] Plateau (best={bestLoss:n4}, current={epochLoss:n4}). Reducing LR -> {learningRate:n6}");
                    }
                }

                #endregion

                #region Early Stopping.

                // early stop check
                if (epoch >= _earlyStopMinEpochs && (epochsSinceImprovement >= _earlyStopPatience || epochLoss < _convergence))
                {
                    Console.WriteLine(
                        $"[Convergence] No improvement for {epochsSinceImprovement} epochs. " +
                        $"Best loss: {bestLoss:n4}. Stopping at epoch {epoch + 1}.");
                    break;
                }

                #endregion

                previousEpochLoss = epochLoss;
            }

            return dni;
        }

        private static double[]? GetImageGrayscaleBytes(string imagePath, int resizeWidth, int resizeHeight, int rotationAngleRange = 5)
        {
            var fileBytes = File.ReadAllBytes(imagePath);

            // Load the image in RGB format and convert to RGBA.
            using var img = Image.Load<Rgba32>(new MemoryStream(fileBytes));

            int width = img.Width;
            int height = img.Height;

            // Detect bounds of non-white pixels
            int threshold = 250;
            int left = width, right = 0, top = height, bottom = 0;

            img.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < width; x++)
                    {
                        Rgba32 p = row[x];
                        if (p.R < threshold || p.G < threshold || p.B < threshold)
                        {
                            if (x < left) left = x;
                            if (x > right) right = x;
                            if (y < top) top = y;
                            if (y > bottom) bottom = y;
                        }
                    }
                }
            });

            // No ink detected — blank image
            if (right <= left || bottom <= top)
                return null;

            // Add margin and clamp to image edges
            int margin = 5;
            left = Math.Max(0, left - margin);
            top = Math.Max(0, top - margin);
            right = Math.Min(width - 1, right + margin);
            bottom = Math.Min(height - 1, bottom + margin);

            int cropWidth = right - left + 1;
            int cropHeight = bottom - top + 1;

            // Crop region of interest
            var bounds = new Rectangle(left, top, cropWidth, cropHeight);
            using var cropped = img.Clone(ctx => ctx.Crop(bounds));

            // Create a square white canvas (to center drawing)
            int squareSize = Math.Max(cropWidth + (margin * 2), cropHeight + (margin * 2));
            using var squareCanvas = new Image<Rgba32>(squareSize, squareSize, Color.White);

            int offsetX = (squareSize - cropWidth) / 2;
            int offsetY = (squareSize - cropHeight) / 2;
            squareCanvas.Mutate(ctx => ctx.DrawImage(cropped, new Point(offsetX, offsetY), 1f));

            int angle = DniUtility.Random.Next(-rotationAngleRange, rotationAngleRange);
            // Small random rotation (for consistency with training)
            using var rotated = squareCanvas.Clone(ctx => ctx.Rotate(angle));

            // flatten the transparency onto a white background
            using var flattened = new Image<Rgba32>(rotated.Width, rotated.Height, Color.White);
            int shiftX = DniUtility.Random.Next(-3, 3);
            int shiftY = DniUtility.Random.Next(-3, 3);
            //Draw rotated image onto white background with a small random shift in position:
            flattened.Mutate(ctx => ctx.DrawImage(rotated, new Point(shiftX, shiftY), 1f));

            squareCanvas.Mutate(ctx => ctx.GaussianBlur(DniUtility.Random.Next(4, 8)));

            using var resized = flattened.Clone(ctx => ctx.Resize(resizeWidth, resizeHeight));

            // Convert to grayscale and normalize [0..1]
            var pixels = new double[resizeWidth * resizeHeight];
            int index = 0;

            resized.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < resizeHeight; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < resizeWidth; x++)
                    {
                        Rgba32 p = row[x];
                        double gray = (0.299 * p.R + 0.587 * p.G + 0.114 * p.B) / 255.0;
                        pixels[index++] = gray;
                    }
                }
            });

            //resized.Save($"C:\\NTDLS\\NTDLS.Determinet\\debug\\{Path.GetFileNameWithoutExtension(imagePath)}.png");

            return pixels;
        }

        /*
        static double[] GetImageGrayscaleBytes(string imagePath, int resizeWidth, int resizeHeight, int rotationAngleRange = 5)
        {
            try
            {
                //Using images from: https://www.nist.gov/srd/nist-special-database-19

                var fileBytes = File.ReadAllBytes(imagePath);

                // Load the image in RGB format and convert to RGBA.
                using var img = Image.Load<Rgba32>(new MemoryStream(fileBytes));

                // Determine larger canvas size (double the original size to prevent cropping during rotation).
                int largerSize = Math.Max(img.Width, img.Height) * 2;

                // Create a new transparent image with the larger canvas size.
                using var canvas = new Image<Rgba32>(largerSize, largerSize, Color.White);

                // Center the original image on this new transparent canvas.
                int offsetX = (largerSize - img.Width) / 2;
                int offsetY = (largerSize - img.Height) / 2;
                canvas.Mutate(x => x.DrawImage(img, new Point(offsetX, offsetY), 1.0f));

                // Apply random rotation to the whole canvas.
                int randomAngle = DniUtility.Random.Next(-rotationAngleRange, rotationAngleRange);
                canvas.Mutate(x => x.Rotate(randomAngle));

                // Define the bounds for cropping out non-transparent areas.
                int left = canvas.Width, right = 0, top = canvas.Height, bottom = 0;
                for (int y = 0; y < canvas.Height; y++)
                {
                    for (int x = 0; x < canvas.Width; x++)
                    {
                        Rgba32 pixel = canvas[x, y];
                        if (pixel.A > 0) // Find non-transparent pixels.
                        {
                            if (x < left) left = x;
                            if (x > right) right = x;
                            if (y < top) top = y;
                            if (y > bottom) bottom = y;
                        }
                    }
                }

                // Crop to the bounding box of the rotated image content.
                int width = right - left + 1;
                int height = bottom - top + 1;
                var bounds = new Rectangle(left, top, width, height);
                canvas.Mutate(x => x.Crop(bounds));

                // Define initial bounds for cropping non-white areas.
                left = canvas.Width;
                right = 0;
                top = canvas.Height;
                bottom = 0;

                // Loop through pixels to find non-white areas.
                for (int y = 0; y < canvas.Height; y++)
                {
                    for (int x = 0; x < canvas.Width; x++)
                    {
                        Rgba32 pixel = canvas[x, y];
                        if (pixel.A > 0 && (pixel.R < 255 || pixel.G < 255 || pixel.B < 255))
                        {
                            if (x < left) left = x;
                            if (x > right) right = x;
                            if (y < top) top = y;
                            if (y > bottom) bottom = y;
                        }
                    }
                }

                // Calculate the bounding box.
                width = right - left + 1;
                height = bottom - top + 1;
                bounds = new Rectangle(left, top, width, height);

                // Crop to the bounding box.
                canvas.Mutate(x => x.Crop(bounds));

                // Resize to target dimensions.
                using var finalResizedImg = canvas.Clone(context => context.Resize(resizeWidth, resizeHeight));

                // Convert to grayscale and normalize.
                double[] pixelData = new double[resizeWidth * resizeHeight];
                int index = 0;
                for (int y = 0; y < finalResizedImg.Height; y++)
                {
                    for (int x = 0; x < finalResizedImg.Width; x++)
                    {
                        var pixel = finalResizedImg[x, y];
                        byte grayValue = (byte)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
                        pixelData[index++] = grayValue / 255.0; // Scale the pixel value to 0-1 (normalize).
                    }
                }

                //finalResizedImg.Save($"debug_{Path.GetFileNameWithoutExtension(imagePath)}.png");

                return pixelData;
            }
            catch
            {
                throw;
            }
        }
        */
    }
}
