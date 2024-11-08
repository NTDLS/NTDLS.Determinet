﻿using NTDLS.Determinet.ActivationFunctions.Interfaces;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    ///  The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly
    ///  if it is positive, otherwise, it will output zero. It has become the default activation function for many types of neural networks
    ///  because a model that uses it is easier to train and often achieves better performance.
    /// </summary>
    [Serializable]
    public class DniReLUFunction : IDniActivationSingleValue
    {
        public double Activation(double x)
        {
            if (x > 0)
            {
                return x;
            }
            return 0;
        }

        public double Derivative(double x, double[] trueLabel)
        {
            if (x > 0)
            {
                return 1;
            }
            return 0;
        }
    }
}
