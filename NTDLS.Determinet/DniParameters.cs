namespace NTDLS.Determinet
{
    public class DniParameters
    {
        public class LayerParameters
        {
            /// <summary>
            /// Represents the configuration key for enabling or disabling batch normalization.
            /// </summary>
            public const string UseBatchNorm = "UseBatchNorm";
            /// <summary>
            /// Represents the configuration key for enabling the use of Kaiming initialization in batch normalization layers
            /// which will auto-scale the gamma parameter based on the layer's node count.
            /// </summary>
            public const string BatchNormUseKaiming = "BatchNormUseKaiming";
            /// <summary>
            /// Represents the name of the gamma parameter used in batch normalization layers.
            /// </summary>
            public const string BatchNormGamma = "BatchNormGamma";
            /// <summary>
            /// Represents the key used to identify the beta parameter in a batch normalization operation.
            /// </summary>
            public const string BatchNormBeta = "BatchNormBeta";
        }
    }
}
