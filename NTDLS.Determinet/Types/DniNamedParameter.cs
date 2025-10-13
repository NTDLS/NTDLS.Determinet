namespace NTDLS.Determinet.Types
{
    /// <summary>
    ///     
    /// </summary>
    /// <param name="key"></param>
    /// <param name="dataType"></param>
    /// <param name="defaultValue"></param>
    public class DniNamedParameter(string key, Type dataType, object defaultValue)
    {
        /// <summary>
        /// Gets the unique identifier associated with this instance.
        /// </summary>
        public string Key { get; } = key;
        /// <summary>
        /// Gets the data type associated with the current instance.
        /// </summary>
        public Type DataType { get; } = dataType;
        /// <summary>
        /// Gets or sets the default value associated with this instance.
        /// </summary>
        public object DefaultValue { get; set; } = defaultValue;
    }
}
