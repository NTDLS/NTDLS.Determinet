﻿using NTDLS.Determinet.ActivationFunctions.Interfaces;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// The hyperbolic tangent function, or tanh for short, is a similar shaped nonlinear activation function that outputs values between -1.0 and 1.0.
    /// In the later 1990s and through the 2000s, the tanh function was preferred over the sigmoid activation function as models that used it were
    /// easier to train and often had better predictive performance.
    /// </summary>
    [Serializable]
    public class DniTanhFunction : IDniActivationSingleValue
    {
        public double Activation(double x)
        {
            return (double)Math.Tanh(x);
        }

        public double Derivative(double x, double[] trueLabel)
        {
            return 1 - (x * x);
        }
    }
}
