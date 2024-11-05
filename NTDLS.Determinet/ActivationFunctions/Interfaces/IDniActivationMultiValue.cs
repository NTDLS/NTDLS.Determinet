namespace NTDLS.Determinet.ActivationFunctions.Interfaces
{
    public interface IDniActivationMultiValue
    {
        double[] Activation(double[] x);
        double[] Derivative(double[] x, double[] trueLabel);
    }
}
