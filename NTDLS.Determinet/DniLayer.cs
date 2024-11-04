using Newtonsoft.Json;
using NTDLS.Determinet.ActivationFunctions;
using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    public class DniLayer
    {
        [JsonProperty]
        private readonly int _inputSize;
        [JsonProperty]
        private readonly int _outputSize;
        [JsonProperty]
        private double[,] _weights;
        [JsonProperty]
        private double[] _biases;

        [JsonProperty]
        private readonly DniActivationType _activationType;

        [JsonProperty]
        private readonly DniNamedFunctionParameters? _activationParameter;
        private IDniActivationFunction _activationFunction;

        public DniLayer()
        {
            _weights = new double[0, 0];
            _biases = Array.Empty<double>();
        }

        public DniLayer(int inputSize, int outputSize, DniActivationType activationType, DniNamedFunctionParameters? activationParameter = null)
        {
            _activationType = activationType;
            _activationParameter = activationParameter;
            _inputSize = inputSize;
            _outputSize = outputSize;
            _weights = InitializeMatrix(inputSize, outputSize);
            _biases = InitializeArray(outputSize);

            InstantiateActivationFunction();

            if (_activationFunction == null)
            {
                throw new NotImplementedException($"Unknown activation type: [{activationType}].");
            }
        }

        internal void InstantiateActivationFunction()
        {
            _activationFunction = _activationType switch
            {
                DniActivationType.Identity => new DniIdentityFunction(_activationParameter),
                DniActivationType.ReLU => new DniReLUFunction(),
                DniActivationType.Linear => new DniLinearFunction(_activationParameter),
                DniActivationType.Sigmoid => new DniSigmoidFunction(),
                DniActivationType.Tanh => new DniTanhFunction(),
                DniActivationType.LeakyReLU => new DniLeakyReLUFunction(),
                _ => throw new NotImplementedException($"Unknown activation type: [{_activationType}].")
            };
        }

        private double[,] InitializeMatrix(int rows, int cols)
        {
            var rand = new Random();
            var matrix = new double[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = rand.NextDouble() - 0.5;
                }
            }
            return matrix;
        }

        private double[] InitializeArray(int length)
        {
            var rand = new Random();
            var array = new double[length];
            for (int i = 0; i < length; i++)
            {
                array[i] = rand.NextDouble() - 0.5;
            }
            return array;
        }

        public double[] FeedForward(double[] inputs)
        {
            double[] outputs = new double[_outputSize];

            for (int i = 0; i < _outputSize; i++)
            {
                outputs[i] = _biases[i];
                for (int j = 0; j < _inputSize; j++)
                {
                    outputs[i] += inputs[j] * _weights[j, i];
                }
                outputs[i] = _activationFunction.Activation(outputs[i]);
            }

            return outputs;
        }

        public double[] CalculateOutputLayerError(double[] outputActivations, double[] expectedOutputs)
        {
            double[] outputErrors = new double[_outputSize];
            for (int i = 0; i < _outputSize; i++)
            {
                outputErrors[i] = (expectedOutputs[i] - outputActivations[i]) * _activationFunction.Derivative(outputActivations[i]);
            }
            return outputErrors;
        }

        public double[] Backpropagate(double[] errors, double[] previousActivations, double learningRate)
        {
            double[] previousLayerErrors = new double[_inputSize];

            // Update weights and biases
            for (int i = 0; i < _outputSize; i++)
            {
                _biases[i] += learningRate * errors[i];
                for (int j = 0; j < _inputSize; j++)
                {
                    _weights[j, i] += learningRate * errors[i] * previousActivations[j];
                    previousLayerErrors[j] += errors[i] * _weights[j, i];
                }
            }

            // Apply derivative to previous layer errors
            for (int i = 0; i < _inputSize; i++)
            {
                previousLayerErrors[i] *= _activationFunction.Derivative(previousActivations[i]);
            }

            return previousLayerErrors;
        }
    }
}
