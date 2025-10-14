namespace NTDLS.Determinet.Types
{
    /// <summary>
    /// Represents a named parameter with an associated key, data type, and default value.
    /// </summary>
    /// <remarks>This class is typically used to define parameters with metadata, such as their unique
    /// identifier, expected data type, and an optional default value. Instances of this class are immutable except for
    /// the <see cref="DefaultValue"/> property, which can be updated after initialization.</remarks>
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
