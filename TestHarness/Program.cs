using NTDLS.Determinet;
using NTDLS.Determinet.Types;

namespace TestHarness
{
    internal class Program
    {
        public enum AIInputs
        {
            DistanceFromObstacle,
            AngleToObstacleInDecimalDegrees
        }

        public enum AIOutputs
        {
            MoveAway,
            AdjustSpeed
        }

        static void Main()
        {
            TrainAndSaveModel("./TestHarness.json");

            //Note that if you want to use the model in different threads, you will need to
            //  make a clone since the values of the input nodes are altered when making decisions.
            //  Fortunately, this can be easily accomplished with a call to Clone();
            var network = LoadSavedModel("./TestHarness.json");

            var decidingFactors = GatherInputs(); //Get decision inputs.

            var decisions = network.FeedForward(decidingFactors); //Make decisions.

            //Handle the speed decisions.
            var shouldAdjustSpeed = decisions.Get(AIOutputs.AdjustSpeed);
            if (shouldAdjustSpeed > 0.8)
            {
                //Adjust speed up.
            }
            else if (shouldAdjustSpeed < 0.2)
            {
                //Adjust speed down.
            }

            //Handle the heading direction.
            var shouldMoveAway = decisions.Get(AIOutputs.MoveAway);
            if (shouldMoveAway > 0.9)
            {
                //Change heading. Maybe just turn around completely?
            }
        }

        /// <summary>
        /// Get the input values we need to make a decision.
        /// </summary>
        /// <returns></returns>
        static DniNamedInterfaceParameters GatherInputs()
        {
            //Here we are just using some dummy values, in this hypothetical situation
            //  we would be getting the distance from a wall and the angle to it.

            double idealMaxDistance = 1000;
            double distanceFromObstacle = 500;

            double percentageOfCloseness = (distanceFromObstacle / idealMaxDistance);
            double angleToObstacleInDecimalDegrees = 0.8;

            var aiParams = new DniNamedInterfaceParameters();
            aiParams.Set(AIInputs.DistanceFromObstacle, percentageOfCloseness);
            aiParams.Set(AIInputs.AngleToObstacleInDecimalDegrees, angleToObstacleInDecimalDegrees);
            return aiParams;
        }

        static DniNeuralNetwork LoadSavedModel(string fileName)
        {
            return DniNeuralNetwork.LoadFromFile(fileName)
                ?? throw new Exception("Failed to load the network from file.");
        }

        static void TrainAndSaveModel(string fileName)
        {
            var Network = new DniNeuralNetwork
            {
                LearningRate = 0.01
            };

            //Add input layer
            Network.Layers.AddInput(ActivationType.LeakyReLU,
                [
                AIInputs.DistanceFromObstacle,
                AIInputs.AngleToObstacleInDecimalDegrees
                ]);

            //Add a intermediate "hidden" layer. You can add more if you like.
            Network.Layers.AddIntermediate(ActivationType.Sigmoid, 8);

            //Add the output layer.
            Network.Layers.AddOutput(
                [
                AIOutputs.MoveAway,
                AIOutputs.AdjustSpeed
                ]);

            //Train the model with some input scenarios. Look at TrainingScenario() and TrainingDecision()
            //  to see that these ominous looking numbers are actually just named inputs. Its pretty simple really.
            for (int epoch = 0; epoch < 5000; epoch++)
            {
                //Very close to observed object, slow way down and get away
                Network.BackPropagate(TrainingScenario(0, 0), TrainingDecision(1, 0));
                Network.BackPropagate(TrainingScenario(0, -1), TrainingDecision(1, 0));
                Network.BackPropagate(TrainingScenario(0, 1), TrainingDecision(1, 0));
                Network.BackPropagate(TrainingScenario(0, 0.5), TrainingDecision(1, 0));
                Network.BackPropagate(TrainingScenario(0, -0.5), TrainingDecision(1, 0));

                //Pretty close to observed object, slow down a bit and get away.
                Network.BackPropagate(TrainingScenario(0.25, 0), TrainingDecision(1, 0.2));
                Network.BackPropagate(TrainingScenario(0.25, -1), TrainingDecision(1, 0.2));
                Network.BackPropagate(TrainingScenario(0.25, 1), TrainingDecision(1, 0.2));
                Network.BackPropagate(TrainingScenario(0.25, 0.5), TrainingDecision(1, 0.2));
                Network.BackPropagate(TrainingScenario(0.25, -0.5), TrainingDecision(1, 0.2));

                //Very far from observed object, speed up and maintain heading.
                Network.BackPropagate(TrainingScenario(1, 0), TrainingDecision(0, 1));
                Network.BackPropagate(TrainingScenario(1, -1), TrainingDecision(0, 1));
                Network.BackPropagate(TrainingScenario(1, 1), TrainingDecision(0, 1));
                Network.BackPropagate(TrainingScenario(1, 0.5), TrainingDecision(0, 1));
                Network.BackPropagate(TrainingScenario(1, -0.5), TrainingDecision(0, 1));

                //Pretty far from observed object, maintain heading but don't change speed.
                Network.BackPropagate(TrainingScenario(0.75, 0), TrainingDecision(0, 0.5));
                Network.BackPropagate(TrainingScenario(0.75, -1), TrainingDecision(0, 0.5));
                Network.BackPropagate(TrainingScenario(0.75, 1), TrainingDecision(0, 0.5));
                Network.BackPropagate(TrainingScenario(0.75, 0.5), TrainingDecision(0, 0.5));
                Network.BackPropagate(TrainingScenario(0.75, -0.5), TrainingDecision(0, 0.5));
            }

            static DniNamedInterfaceParameters TrainingScenario(double distanceFromObstacle, double angleToObstacleInDecimalDegrees)
            {
                var param = new DniNamedInterfaceParameters();
                param.Set(AIInputs.DistanceFromObstacle, distanceFromObstacle);
                param.Set(AIInputs.AngleToObstacleInDecimalDegrees, angleToObstacleInDecimalDegrees);
                return param;
            }

            static DniNamedInterfaceParameters TrainingDecision(double moveAway, double adjustSpeed)
            {
                var param = new DniNamedInterfaceParameters();

                param.Set(AIOutputs.MoveAway, moveAway);
                param.Set(AIOutputs.AdjustSpeed, adjustSpeed);
                return param;
            }

            //Save the network to a file. This is only done here for examples sake.
            Network.Save(fileName);
        }

    }
}
