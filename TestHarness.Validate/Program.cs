using NTDLS.Determinet;
using NTDLS.Determinet.Types;
using TestHarness.Library;
using static NTDLS.Determinet.DniParameters;

namespace TestHarness.Validate
{
    internal class Program
    {
        const string _sampleImagePath = "..\\..\\..\\..\\Sample Images\\Validation";
        const string _trainedModelPath = "..\\..\\..\\..\\Trained Models";

        static void Main(string[] args)
        {
            if (args.Length > 0)
            {
                var givenFileName = args[0];
                var existing = Path.Combine(_trainedModelPath, givenFileName);
                if (!File.Exists(existing))
                {
                    Console.WriteLine($"The trained model file does not exist: {existing}");
                    return;
                }

                MeasureModel(existing);
                return;
            }
            string historyLogFile = Path.Combine(_trainedModelPath, "HistoryLog.csv");
            var alreadyProcessed = Array.Empty<string>();
            if (File.Exists(historyLogFile))
            {
                alreadyProcessed = File.ReadAllLines(historyLogFile);
            }

            Directory.EnumerateFiles(_trainedModelPath, "*.dni")
                .OrderByDescending(f => f)
                .Where(f => !f.EndsWith("Best.dni", StringComparison.InvariantCultureIgnoreCase)
                            && !alreadyProcessed.Contains(Path.GetFileName(f)))
                .ToList()
                .ForEach(f => MeasureModel(f));
        }

        private static void MeasureModel(string trainedModelFile)
        {
            string logFile = Path.Combine(_trainedModelPath, "ValidationLog.csv");
            string historyLogFile = Path.Combine(_trainedModelPath, "HistoryLog.csv");


            var dni = DniNeuralNetwork.LoadFromFile(trainedModelFile)
                ?? throw new Exception("Failed to load the network from file.");

            var backgroundLoader = new BackgroundLoader(dni, _sampleImagePath,
                Constants.ImageWidth, Constants.ImageHeight, DniRange<int>.Zero, DniRange<int>.Zero, new DniRange<float>(0.5f, 0.5f), DniRange<double>.One);

            var learningRate = dni.Parameters.Get<double>(Network.LearningRate);
            var reportedLoss = dni.Parameters.Get<double>("BatchLoss");
            var reportedEpochs = dni.Parameters.Get<int>("Epochs", 0);

            Console.WriteLine($"Loaded model : {trainedModelFile}");
            Console.WriteLine($"Learning Rate: {learningRate:n8}");
            Console.WriteLine($"Reported Loss: {reportedLoss:n8}");

            backgroundLoader.BeginPopulation(0);

            double confidenceThreshold = 0.8;
            double samplesProcessed = 0;
            double correct = 0;
            double incorrect = 0;
            double noConfidence = 0;
            var confusion = new Dictionary<string, (int correct, int total)>();

            double weightedAccuracySum = 0;
            double weightedConfidenceSum = 0;

            while (backgroundLoader.Pop(out var preparedSample))
            {
                dni.Forward(preparedSample.Bits, out var outputLabelValues);

                var expected = preparedSample.Sample.ExpectedValue;
                var predicted = outputLabelValues.Max();

                bool isCorrect = string.Equals(expected, predicted.Key, StringComparison.InvariantCultureIgnoreCase);
                weightedAccuracySum += predicted.Value * (isCorrect ? 1.0 : 0.0);
                weightedConfidenceSum += predicted.Value;

                if (predicted.Value > confidenceThreshold)
                {
                    if (!confusion.ContainsKey(expected))
                        confusion[expected] = (0, 0);

                    confusion[expected] = (
                        correct: confusion[expected].correct + (string.Equals(expected, predicted.Key, StringComparison.InvariantCultureIgnoreCase) ? 1 : 0),
                        total: confusion[expected].total + 1
                    );

                    if (string.Equals(expected, predicted.Key, StringComparison.InvariantCultureIgnoreCase))
                    {
                        correct++;
                    }
                    else
                    {
                        incorrect++;
                    }
                }
                else
                {
                    noConfidence++;
                }

                samplesProcessed++;

                Console.Write($"{samplesProcessed:n0} of {backgroundLoader.Count:n0} ({((samplesProcessed / backgroundLoader.Count) * 100.0):n1}%)\r");
            }

            double weightedAccuracy = weightedConfidenceSum > 0 ? weightedAccuracySum / weightedConfidenceSum : 0;
            double confidentSamples = correct + incorrect;

            Console.WriteLine("\n=== Validation Summary ===");
            Console.WriteLine($"Total Samples:           {samplesProcessed:n0}");
            Console.WriteLine($"Correct:                 {correct:n0}");
            Console.WriteLine($"Incorrect:               {incorrect:n0}");
            Console.WriteLine($"Accuracy:                {(correct / samplesProcessed * 100.0):F2}%");
            //This tells us how accurate the model is when it is confident in its prediction, this is the number we care about
            //  because it means that when the model is confident, it is usually correct - and it "knows when it doesn't know".
            Console.WriteLine($"Accuracy (Confident):    {(confidentSamples > 0 ? correct / confidentSamples * 100.0 : 0):F2}%");
            Console.WriteLine($"No-Confidence Samples:   {noConfidence} ({noConfidence / samplesProcessed * 100.0:F2}%)");
            Console.WriteLine($"Weighted Accuracy:       {weightedAccuracy * 100.0:F2}%");

            if (!File.Exists(logFile)) // Build CSV header only if the file doesn't exist
            {
                File.AppendAllText(logFile,
                    "Epoch,Name,Timestamp,LearningRate,ReportedLoss,TotalSamples,Correct,Incorrect,Accuracy,AccuracyConfident,NoConfidence,WeightedAccuracy\n");
            }

            // Append this epoch’s results
            string csvLine = string.Join(",",
                reportedEpochs.ToString(),
                Path.GetFileName(trainedModelFile),
                DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                learningRate.ToString(),
                reportedLoss.ToString(),
                samplesProcessed.ToString(),
                correct.ToString(),
                incorrect.ToString(),
                (correct / samplesProcessed * 100.0).ToString(),
                (confidentSamples > 0 ? correct / confidentSamples * 100.0 : 0).ToString(),
                (noConfidence / samplesProcessed * 100.0).ToString(),
                (weightedAccuracy * 100.0).ToString()
            );

            File.AppendAllText(logFile, csvLine + Environment.NewLine);
            File.AppendAllText(historyLogFile, Path.GetFileName(trainedModelFile) + Environment.NewLine);

            /*
            // Detailed per-character accuracy:
            Console.WriteLine("\n=== Per-Character Accuracy ===");
            foreach (var kv in confusion.OrderBy(k => k.Key))
            {
                double acc = kv.Value.total > 0 ? (double)kv.Value.correct / kv.Value.total * 100.0 : 0;
                Console.WriteLine($"{kv.Key}: {acc:F1}% ({kv.Value.correct}/{kv.Value.total})");
            }
            */

            //Console.WriteLine("Press Enter to exit.");
            //Console.ReadLine();
        }
    }
}
