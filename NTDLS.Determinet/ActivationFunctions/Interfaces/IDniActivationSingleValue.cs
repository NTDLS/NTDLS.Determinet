namespace NTDLS.Determinet.ActivationFunctions.Interfaces
{
    public interface IDniActivationSingleValue: IDniActivationFunction
    {
        double Activation(double x);
        double Derivative(double x, double[] trueLabel);
    }
}
