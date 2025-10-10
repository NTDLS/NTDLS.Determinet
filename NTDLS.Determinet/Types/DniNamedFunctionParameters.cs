namespace NTDLS.Determinet.Types
{
    /// <summary>
    /// Used to pass optional parameters to activation functions.
    /// </summary>
    public class DniNamedFunctionParameters
    {
        public readonly Dictionary<string, object> Values = new(StringComparer.InvariantCultureIgnoreCase);

        public void Set(string key, double value)
            => Values[key] = value;

        public void Set(string key, DniRange value)
            => Values[key] = value;

        public T Get<T>(string key)
            => (T)Values[key];

        public KeyValuePair<string, object> Get(int index)
            => Values.ElementAt(index);

        public object[] ToArray()
        {
            var values = new object[Values.Count];
            var keys = Values.Keys.ToList();
            for (int i = 0; i < keys.Count; i++)
            {
                values[i] = Values[keys[i]];
            }
            return values;
        }

        public T Get<T>(string key, T defaultValue)
        {
            if (Values.TryGetValue(key, out object? value) && value != null)
            {
                return (T)value;
            }
            return defaultValue;
        }
    }
}
