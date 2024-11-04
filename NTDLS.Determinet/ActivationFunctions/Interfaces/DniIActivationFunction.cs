namespace NTDLS.Determinet.ActivationFunctions.Interfaces
{
    public interface DniIActivationFunction : DniIFunction
    {
        double Activation(double x);
        double Derivative(double x);
    }
}
