using Newtonsoft.Json;

namespace NTDLS.Determinet.Types
{
    /// <summary>
    /// When the input and/or output nodes are aliased, this class is used to "interface" with the neural network input and output
    /// by allowing you to supply named values instead of using neuron ordinals.
    /// </summary>
    [Serializable]
    public class DniNamedInterfaceParameters
    {
        [JsonProperty]
        private readonly Dictionary<string, double> _dictonary = new();

        public void Set(object key, double value)
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

        /// <summary>
        /// Sets the input value if the given value is less than the existing value or if the key does not yet exist.
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value"></param>
        public void SetIfLess(object key, double value)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();

            if (_dictonary.ContainsKey(stringKey) == false)
            {
                _dictonary.Add(stringKey, value);
            }
            else
            {
                var existingValue = _dictonary[stringKey];

                if (value < existingValue)
                {
                    _dictonary[stringKey] = value;
                }
            }
        }

        /// <summary>
        /// Sets the input value if the given value is greater than the existing value or if the key does not yet exist.
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value"></param>
        public void SetIfGreater(object key, double value)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();

            if (_dictonary.ContainsKey(stringKey) == false)
            {
                _dictonary.Add(stringKey, value);
            }
            else
            {
                var existingValue = _dictonary[stringKey];

                if (value > existingValue)
                {
                    _dictonary[stringKey] = value;
                }
            }
        }

        public double[] ToArray()
        {
            var values = new double[_dictonary.Count];
            var keys = _dictonary.Keys.ToList();
            for (int i = 0; i < keys.Count; i++)
            {
                values[i] = _dictonary[keys[i]];
            }
            return values;
        }

        public double Get(object key)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();
            return _dictonary[stringKey];
        }

        public KeyValuePair<string, double> Get(int index)
        {
            return _dictonary.ElementAt(index);
        }

        public double Get(object key, double defaultValue)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();

            if (_dictonary.TryGetValue(stringKey, out double value))
            {
                return value;
            }
            return defaultValue;
        }
    }
}
