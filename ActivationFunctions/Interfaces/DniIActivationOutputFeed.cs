namespace NTDLS.Determinet.ActivationFunctions.Interfaces
{
    public interface DniIActivationOutputFeed : DniIFunction
    {
        double[] Activation(double[] previousLayer);
    }
}
