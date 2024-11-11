namespace NTDLS.Determinet.Types
{
    /// <summary>
    /// Used to pass optional parameters to activation functions.
    /// </summary>
    public class DniNamedFunctionParameters
    {
        public readonly Dictionary<string, object> Values = new();

        public void Set(object key, double value)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();

            if (Values.ContainsKey(stringKey))
            {
                Values[stringKey] = value;
            }
            else
            {
                Values.Add(stringKey, value);
            }
        }

        public void Set(object key, DniRange value)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();

            if (Values.ContainsKey(stringKey))
            {
                Values[stringKey] = value;
            }
            else
            {
                Values.Add(stringKey, value);
            }
        }

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

        public T Get<T>(object key)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();

            return (T)Values[stringKey];
        }

        public T Get<T>(object key, T defaultValue)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();

            if (Values.ContainsKey(stringKey))
            {
                return (T)Values[stringKey];
            }
            return defaultValue;
        }

        public KeyValuePair<string, object> Get(int index)
        {
            return Values.ElementAt(index);
        }

        public object Get(object key, object defaultValue)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();

            if (Values.TryGetValue(stringKey, out object? value))
            {
                return value;
            }
            return defaultValue;
        }
    }
}
