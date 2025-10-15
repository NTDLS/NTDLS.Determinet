using NTDLS.Determinet;
using NTDLS.Determinet.Types;
using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;

namespace TestHarness.Library
{
    public class BackgroundLoader
    {
        private readonly Lock _lockGetRandomTrainingSample = new();
        private int _threadsCompleted = 0;
        private readonly int _threadCount = Environment.ProcessorCount;
        private List<TrainingSample> _trainingSample = new();
        private readonly ConcurrentStack<PreparedTrainingSample> _stack = new();

        private int _resizeWidth;
        private int _resizeHeight;
        private DniRange<int>? _angleVariance;
        private DniRange<int>? _shiftVariance;
        private DniRange<float>? _blurVariance;
        private DniRange<double>? _scaleVariance;

        public int Count => _trainingSample.Count;

        public bool IsComplete => _threadsCompleted == _threadCount;

        public BackgroundLoader(DniNeuralNetwork dni, string path, int resizeWidth, int resizeHeight,
            DniRange<int>? angleVariance, DniRange<int>? shiftVariance, DniRange<float>? blurVariance, DniRange<double>? scaleVariance)
        {
            _resizeWidth = resizeWidth;
            _resizeHeight = resizeHeight;
            _angleVariance = angleVariance;
            _shiftVariance = shiftVariance;
            _blurVariance = blurVariance;
            _scaleVariance = scaleVariance;

            if (dni.OutputLabels == null || dni.OutputLabels.Length == 0)
                throw new InvalidOperationException("DNI Neural Network must have OutputLabels defined to load training sample.");

            var imagePaths = Directory.EnumerateFiles(path, "*.png", new EnumerationOptions() { RecurseSubdirectories = true });

            int charIndex = 0;

            var layerExpectations = new Dictionary<char, double[]>();
            foreach (var character in dni.OutputLabels)
            {
                var outputs = new double[dni.OutputLabels.Length];
                outputs[charIndex++] = 1;
                // Use the first character of the label as the key to find the expected output for that character:
                layerExpectations[character[0]] = outputs;
            }

            foreach (var imagePath in imagePaths)
            {
                var fileName = Path.GetFileName(imagePath);
                //Use the first character of the image filename as the expected output character:
                var layerExpectation = layerExpectations[fileName[0]];
                _trainingSample.Add(new(imagePath, fileName[0].ToString(), layerExpectation));
            }
        }

        public bool Pop([NotNullWhen(true)] out PreparedTrainingSample? outPrepared)
        {
            while (_stack.TryPop(out var prepared) || !IsComplete)
            {
                if (prepared == null)
                {
                    Thread.Sleep(10);
                }
                else
                {
                    outPrepared = prepared;
                    return true;
                }
            }

            outPrepared = null;
            return false;
        }

        public void BeginPopulation(int epoch)
        {
            _threadsCompleted = 0;
            for (int i = 0; i < _threadCount; i++)
            {
                Task.Run(() =>
                {
                    while (TryGetRandomTrainingSample(epoch, out var sample))
                    {
                        var bits = ImageUtility.GetImageGrayscaleBytes(sample.FileName, _resizeWidth,
                            _resizeHeight, _angleVariance, _shiftVariance, _blurVariance, _scaleVariance)
                        ?? throw new Exception($"Failed to load sample image: {sample.FileName}");

                        _stack.Push(new PreparedTrainingSample(sample, bits));

                        while (_stack.Count >= 100)
                        {
                            Thread.Sleep(10); // Let the backlog get processed a bit before adding more.
                        }
                    }

                    Interlocked.Increment(ref _threadsCompleted);
                });
            }
        }

        /// <summary>
        /// Gets a random training model from the list that has not yet been fed-in for the current training epoch.
        /// Increments the training epoch for the random model.
        /// </summary>
        private bool TryGetRandomTrainingSample(int epoch, [NotNullWhen(true)] out TrainingSample? randomSample)
        {
            lock (_lockGetRandomTrainingSample)
            {
                var samplesThisEpoch = _trainingSample.Where(o => o.Epoch == epoch).ToList();

                if (samplesThisEpoch.Count == 0)
                {
                    randomSample = null;
                    return false;
                }

                int randomIndex = DniUtility.Random.Next(samplesThisEpoch.Count);
                randomSample = samplesThisEpoch[randomIndex];

                //Once we have consumed a sample for this epoch, increment its epoch so we don't use it again this epoch:
                randomSample.Epoch++;

                return true;
            }
        }
    }
}
