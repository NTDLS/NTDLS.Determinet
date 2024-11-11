namespace NTDLS.Determinet.ActivationFunctions.Interfaces
{
    public interface IDniActivationFunction
    {
        double[] Activation(double[] nodes);
        double Derivative(double x);
    }
}
