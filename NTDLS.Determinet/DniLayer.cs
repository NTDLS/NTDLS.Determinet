using Newtonsoft.Json;
using NTDLS.Determinet.ActivationFunctions;
using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet
{
    internal class DniLayer
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
        DniLayerType _layerType;

        [JsonProperty]
        private readonly DniActivationType _activationType;

        [JsonProperty]
        private readonly DniNamedFunctionParameters? _activationParameter;
        private IDniActivationFunction _activationFunction;

        /// <summary>
        /// Used only for deserialization.
        /// </summary>
        internal DniLayer()
        {
            _weights = new double[0, 0];
            _biases = Array.Empty<double>();
            _activationFunction = new DniSigmoidFunction();
        }

        internal DniLayer(DniLayerType layerType, int inputSize, int outputSize, DniActivationType activationType, DniNamedFunctionParameters? activationParameter = null)
        {
            _layerType = layerType;
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
            var matrix = new double[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = DniUtility.Random.NextDouble() - 0.5;
                }
            }
            return matrix;
        }

        private double[] InitializeArray(int length)
        {
            var array = new double[length];
            for (int i = 0; i < length; i++)
            {
                array[i] = DniUtility.Random.NextDouble() - 0.5;
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

        public double[] CalculateOutputLayerError(double[] outputActivations, double[] trueLabel)
        {
            if (_activationFunction is IDniActivationSingleValue activationSingleValue)
            {
                double[] outputErrors = new double[_outputSize];
                for (int i = 0; i < _outputSize; i++)
                {
                    outputErrors[i] = (trueLabel[i] - outputActivations[i]) * activationSingleValue.Derivative(outputActivations[i], trueLabel);
                }
                return outputErrors;
            }
            else if (_activationFunction is IDniActivationMultiValue activationMultiValue)
            {
                return activationMultiValue.Derivative(outputActivations, trueLabel);
            }
            throw new NotImplementedException("Activation type is not implemented.");
        }

        public double[] Backpropagate(double[] errors, double[] previousActivations, double learningRate, double[] trueLabel)
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

            // Apply the appropriate derivative function
            if (_activationFunction is IDniActivationSingleValue activationSingleValue)
            {
                for (int i = 0; i < _inputSize; i++)
                {
                    previousLayerErrors[i] *= activationSingleValue.Derivative(previousActivations[i], trueLabel);
                }
            }
            else if (_activationFunction is IDniActivationMultiValue activationMultiValue)
            {
                var derivatives = activationMultiValue.Derivative(previousActivations, trueLabel);
                for (int i = 0; i < _inputSize; i++)
                {
                    previousLayerErrors[i] *= derivatives[i];
                }
            }

            return previousLayerErrors;
        }
    }
}
