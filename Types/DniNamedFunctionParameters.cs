using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NTDLS.Determinet.Types
{
    /// <summary>
    /// Used to pass optional parameters to activation functions.
    /// </summary>
    [Serializable]
    public class DniNamedFunctionParameters
    {
        [JsonProperty]
        private readonly Dictionary<string, object> _dictonary = new();

        public void Set(object key, object value)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();

            if (_dictonary.ContainsKey(stringKey))
            {
                _dictonary[stringKey] = value;
            }
            else
            {
                _dictonary.Add(stringKey, value);
            }
        }

        public object[] ToArray()
        {
            var values = new object[_dictonary.Count];
            var keys = _dictonary.Keys.ToList();
            for (int i = 0; i < keys.Count; i++)
            {
                values[i] = _dictonary[keys[i]];
            }
            return values;
        }

        public T Get<T>(object key)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();

            return (T)_dictonary[stringKey];
        }

        public T Get<T>(object key, T defaultValue)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();

            if (_dictonary.ContainsKey(stringKey))
            {
                return (T)_dictonary[stringKey];
            }
            return defaultValue;
        }

        public KeyValuePair<string, object> Get(int index)
        {
            return _dictonary.ElementAt(index);
        }

        public object Get(object key, object defaultValue)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();

            if (_dictonary.TryGetValue(stringKey, out object? value))
            {
                return value;
            }
            return defaultValue;
        }
    }
}
