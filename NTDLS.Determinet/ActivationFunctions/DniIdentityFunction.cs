﻿using NTDLS.Determinet.ActivationFunctions.Interfaces;
using NTDLS.Determinet.Types;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// An identity function is a function that returns the same value as its input. In machine learning, an identity
    /// function is often used as a default activation function for a layer of a neural network. When used as an activation
    /// function, an identity function simply passes the input through the layer unchanged.
    /// </summary>
    public class DniIdentityFunction : IDniActivationFunction
    {
        public DniIdentityFunction(DniNamedFunctionParameters? param)
        {
        }

        public double[] Activation(double[] nodes) => nodes;
        public double Derivative(double x) => 1;
    }
}
