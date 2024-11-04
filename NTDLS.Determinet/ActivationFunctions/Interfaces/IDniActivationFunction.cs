namespace NTDLS.Determinet.ActivationFunctions.Interfaces
{
    public interface IDniActivationFunction
    {
        double Activation(double x);
        double Derivative(double x);
    }
}
