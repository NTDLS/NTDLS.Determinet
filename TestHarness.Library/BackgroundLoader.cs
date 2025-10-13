using NTDLS.Determinet;
using System.Diagnostics.CodeAnalysis;

namespace TestHarness.Library
{
    public static class BackgroundLoader
    {
        public static List<TrainingModel> LoadTrainingModels(string path)
        {
            var trainingModels = new List<TrainingModel>();

            var imagePaths = Directory.EnumerateFiles(path, "*.png", new EnumerationOptions() { RecurseSubdirectories = true });


            int charIndex = 0;

            var layerExpectations = new Dictionary<char, double[]>();
            foreach (var character in Constants.NetworkOutputLabels)
            {
                var outputs = new double[Constants.NetworkOutputLabels.Length];
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
        public static bool TryGetRandomTrainingModel(List<TrainingModel> models, int epoch, [NotNullWhen(true)] out TrainingModel? randomModel)
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
    }
}
