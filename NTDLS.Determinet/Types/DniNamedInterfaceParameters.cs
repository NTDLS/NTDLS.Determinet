using ProtoBuf;

namespace NTDLS.Determinet.Types
{
    /// <summary>
    /// When the input and/or output nodes are aliased, this class is used to "interface" with the neural network input and output
    /// by allowing you to supply named values instead of using neuron ordinals.
    /// </summary>
    [ProtoContract]
    public class DniNamedInterfaceParameters
    {
        [ProtoMember(1)]
        public Dictionary<string, double> Lookup { get; private set; } = new();

        public void Set(object key, double value)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();
            if (Lookup.ContainsKey(stringKey))
            {
                Lookup[stringKey] = value;
            }
            else
            {
                Lookup.Add(stringKey, value);
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

            if (Lookup.ContainsKey(stringKey) == false)
            {
                Lookup.Add(stringKey, value);
            }
            else
            {
                var existingValue = Lookup[stringKey];

                if (value < existingValue)
                {
                    Lookup[stringKey] = value;
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

            if (Lookup.ContainsKey(stringKey) == false)
            {
                Lookup.Add(stringKey, value);
            }
            else
            {
                var existingValue = Lookup[stringKey];

                if (value > existingValue)
                {
                    Lookup[stringKey] = value;
                }
            }
        }

        public double[] ToArray()
        {
            var values = new double[Lookup.Count];
            var keys = Lookup.Keys.ToList();
            for (int i = 0; i < keys.Count; i++)
            {
                values[i] = Lookup[keys[i]];
            }
            return values;
        }

        public double Get(object key)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();
            return Lookup[stringKey];
        }

        public KeyValuePair<string, double> Get(int index)
        {
            return Lookup.ElementAt(index);
        }

        public double Get(object key, double defaultValue)
        {
            string stringKey = (key?.ToString() ?? string.Empty).ToLower();

            if (Lookup.TryGetValue(stringKey, out double value))
            {
                return value;
            }
            return defaultValue;
        }
    }
}
