using NTDLS.Determinet;
using NTDLS.Determinet.Types;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using TestHarness.Library;

namespace TestHarness.Validate
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var sampleImagePath = "..\\..\\..\\..\\Sample Images\\Validation";
            var trainedModelPath = "..\\..\\..\\..\\Trained Models";

            string file = "CharacterRecognition_Best.dni";
            if (args.Length > 0)
            {
                file = args[0];
            }

            var existing = Path.Combine(trainedModelPath, file);
            if (!File.Exists(existing))
            {
                Console.WriteLine($"The trained model file does not exist: {existing}");
                return;
            }

            var dni = DniNeuralNetwork.LoadFromFile(existing)
                ?? throw new Exception("Failed to load the network from file.");

            var backgroundLoader = new BackgroundLoader(dni, sampleImagePath,
                Constants.ImageWidth, Constants.ImageHeight, DniRange<int>.Zero, DniRange<int>.Zero, new DniRange<int>(4, 4), DniRange<double>.One);

            backgroundLoader.BeginPopulation(0);

            double confidenceThreshold = 0.8;
            double samplesProcessed = 0;
            double correct = 0;
            double incorrect = 0;
            double noConfidence = 0;
            var confusion = new Dictionary<string, (int correct, int total)>();

            while (backgroundLoader.Pop(out var preparedSample))
            {
                dni.Forward(preparedSample.Bits, out var outputLabelValues);

                var expected = preparedSample.Sample.ExpectedValue;
                var predicted = outputLabelValues.Max();

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

            double confidentSamples = correct + incorrect;

            Console.WriteLine("\n=== Validation Summary ===");
            Console.WriteLine($"Total Samples: {samplesProcessed}");
            Console.WriteLine($"Correct:       {correct}");
            Console.WriteLine($"Incorrect:     {incorrect}");
            Console.WriteLine($"Accuracy:      {(correct / samplesProcessed * 100.0):F2}%");
            Console.WriteLine($"Accuracy (All):          {(correct / samplesProcessed * 100.0):F2}%");
            //This tells us how accurate the model is when it is confident in its prediction, this is the number we care about
            //  because it means that when the model is confident, it is usually correct - and it "knows when it doesn't know".
            Console.WriteLine($"Accuracy (Confident):    {(confidentSamples > 0 ? correct / confidentSamples * 100.0 : 0):F2}%");
            Console.WriteLine($"No-Confidence Samples:   {noConfidence} ({noConfidence / samplesProcessed * 100.0:F2}%)");

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
