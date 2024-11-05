using NTDLS.Determinet.ActivationFunctions.Interfaces;

namespace NTDLS.Determinet.ActivationFunctions
{
    /// <summary>
    /// The softmax activation function is often used in the output layer of neural networks for multi-class classification tasks.
    /// It converts a vector of raw scores(logits) into a probability distribution over multiple classes.
    /// </summary>
    public class DniSoftMaxFunction: IDniActivationMultiValue
    {
        public double[] Activation(double[] previousLayer)
        {
            double max = previousLayer.Max();  // For numerical stability
            double[] exps = previousLayer.Select(i => Math.Exp(i - max)).ToArray();
            double sumExps = exps.Sum();
            return exps.Select(e => e / sumExps).ToArray();
        }

        public double[,] FullJacobianMatrixDerivative(double[] softmaxOutput)
        {
            int length = softmaxOutput.Length;
            double[,] jacobian = new double[length, length];

            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    if (i == j)
                    {
                        jacobian[i, j] = softmaxOutput[i] * (1 - softmaxOutput[i]);
                    }
                    else
                    {
                        jacobian[i, j] = -softmaxOutput[i] * softmaxOutput[j];
                    }
                }
            }
            return jacobian;
        }

        public double[] Derivative(double[] softmaxOutput, double[] trueLabel)
        {
            int length = softmaxOutput.Length;
            double[] gradient = new double[length];

            for (int i = 0; i < length; i++)
            {
                gradient[i] = softmaxOutput[i] - trueLabel[i];
            }
            return gradient;
        }
    }
}
