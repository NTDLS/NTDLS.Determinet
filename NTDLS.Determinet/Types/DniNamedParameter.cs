namespace NTDLS.Determinet.Types
{
    public class DniNamedParameter(string key, Type dataType, object defaultValue)
    {
        public string Key { get; } = key;
        public Type DataType { get; } = dataType;
        public object DefaultValue { get; set; } = defaultValue;
    }
}
