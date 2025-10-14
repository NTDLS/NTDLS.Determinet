using NTDLS.Determinet;
using NTDLS.Determinet.Types;
using TestHarness.Library;
using static NTDLS.Determinet.DniParameters;

namespace TestHarness.Train
{
    internal class Program
    {
        const string sampleImagePath = "..\\..\\..\\..\\Sample Images\\Training";

        private static readonly char[] _networkOutputLabels = [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'a', 'b', 'c', 'd', 'e', 'f','g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                'A', 'B', 'C', 'D', 'E', 'F','G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
            ];

        const double _initialLearningRate = 0.0005;  // Starting learning rate for training.
        const double _convergence = 0.000000001;    // Threshold for considering the training has converged.
        const int _cooldown = 5;                    // epochs to wait after each learning rate decay.
        const int _patience = 3;                    // Number of epochs to wait before reducing learning rate once cost starts increasing or reaches a plateau.
        const double _decayFactor = 0.8;            // Factor to reduce learning rate
        const int _trainingEpochs = 250;            // Total number of training epochs
        const double _minDelta = 0.001;             // minimum improvement threshold
        const int _earlyStopPatience = 10;          // epochs with no improvement before stopping
        const int _earlyStopMinEpochs = 10;         // minimum epochs before we even consider stopping
        const double _minLearningRate = 0.000001;
        const double _slop = 0.000000000001;

        static void Main(string[] args)
        {
            var trainedModelPath = "..\\..\\..\\..\\Trained Models";


            DniNeuralNetwork dni;

            var existing = Path.Combine(trainedModelPath, "CharacterRecognition_Best.dni");
            if (File.Exists(existing))
            {
                dni = DniNeuralNetwork.LoadFromFile(existing)
                    ?? throw new Exception("Failed to load the network from file.");
            }
            else
            {
                var configuration = new DniConfiguration()
                {
                    LearningRate = _initialLearningRate
                };

                /*
                | Layer    | Role                                     | Notes                                                                              |
                | -------- | ---------------------------------------- | ---------------------------------------------------------------------------------- |
                | Input    | `_imageWidth * _imageHeight` (130x130)   | Flattened grayscale image. Normalize pixel values to [0, 1] or z-score per sample. |
                | Hidden 1 | 512 → LeakyReLU                          | Enough capacity to learn non-linear stroke patterns.                               |
                | Hidden 2 | 256 → LeakyReLU                          | Tapering keeps parameter count reasonable; aids generalization.                    |
                | Hidden 3 | 128 → LeakyReLU                          | Distills features before classification.                                           |
                | Output   | 62 → Softmax                             | One neuron per symbol (0-9, A-Z, a-z).                                             |
                */

                configuration.AddInputLayer(Constants.ImageWidth * Constants.ImageHeight);

                var leakyReLUParam = new DniNamedParameterCollection();
                //leakyReLUParam.Set(Layer.UseBatchNorm, true);
                //leakyReLUParam.Set(Layer.BatchNormMomentum, 0.9);

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

                var softMaxParam = new DniNamedParameterCollection();
                //softMaxParam.Set(SoftMax.Temperature, 5.5);
                var outputLabels = _networkOutputLabels.Select(label => label.ToString()).ToArray();
                configuration.AddOutputLayer(_networkOutputLabels.Length, DniActivationType.SoftMax, softMaxParam, outputLabels);

                dni = new DniNeuralNetwork(configuration);
            }

            // Allow setting initial learning rate from command line for experimentation:
            if (args.Length > 0 && string.IsNullOrWhiteSpace(args[0]) == false && double.TryParse(args[0], out var overrideLearningRate))
            {
                dni.Parameters.Set(Network.LearningRate, overrideLearningRate);
            }

            TrainAndSave(dni, trainedModelPath);

            //dni.Forward([]);

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

        static DniNeuralNetwork TrainAndSave(DniNeuralNetwork dni, string trainedModelPath)
        {
            Console.WriteLine($"Loading image paths...");
            var backgroundLoader = new BackgroundLoader(dni, sampleImagePath,
                Constants.ImageWidth, Constants.ImageHeight, new DniRange<int>(-5, 5), new DniRange<int>(-3, 3), new DniRange<int>(1, 4), new DniRange<double>(0.5, 1));

            var learningRate = dni.Parameters.Get(Network.LearningRate, _initialLearningRate);

            double bestLoss = double.MaxValue;
            int epochsSinceImprovement = 0;
            int cooldownCounter = _cooldown;

            Console.WriteLine($"Starting backlog threads...");

            for (int epoch = 0; epoch < _trainingEpochs; epoch++)
            {
                backgroundLoader.BeginPopulation(epoch);

                dni.Parameters.Set(Network.LearningRate, learningRate);

                double epochLoss = 0;
                double samplesProcessed = 0;

                if (epoch == 0)
                {
                    Console.WriteLine($"Training with learning Rate: {learningRate:n10}");
                }

                while (backgroundLoader.Pop(out var preparedSample))
                {
                    var loss = dni.Train(preparedSample.Bits, preparedSample.Sample.Expectation);
                    epochLoss += loss;
                    //Console.WriteLine($"{loss:n10}");
                    samplesProcessed++;

                    Console.Write($"{samplesProcessed:n0} of {backgroundLoader.Count:n0} ({((samplesProcessed / backgroundLoader.Count) * 100.0):n1}%)\r");
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

                dni.Parameters.Set("BatchLoss", epochLoss);

                //Save checkpoints.
                var timestamp = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
                dni.SaveToFile(Path.Combine(trainedModelPath, $"CharacterRecognition_{timestamp}_{epochLoss:n8}.dni"));

                if (epochLoss < bestLoss - _minDelta)
                {
                    //We save every time we get a new best loss.
                    dni.SaveToFile(Path.Combine(trainedModelPath, "CharacterRecognition_Best.dni"));

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
            }

            return dni;
        }
    }
}
