using ProtoBuf;
using System.Diagnostics.CodeAnalysis;

namespace NTDLS.Determinet.Types
{
    /// <summary>
    /// When the input and/or output nodes are aliased, this class is used to "interface" with the neural network input and output
    /// by allowing you to supply named values instead of using neuron ordinals.
    /// </summary>
    [ProtoContract]
    public class DniNamedLabelValues
    {
        [ProtoMember(1)] public Dictionary<string, double> Lookup { get; private set; } = new();

        public KeyValuePair<string, double>[] Values
            => Lookup.ToArray();

        public void Set(string key, double value)
            => Lookup[key] = value;

        public KeyValuePair<string, double> Max()
        {
            var fff = Lookup.OrderByDescending(o => o.Value).Take(1);

            return Lookup.MaxBy(kv => kv.Value);
        }

        public KeyValuePair<string, double> Min()
            => Lookup.MinBy(kv => kv.Value);

        /// <summary>
        /// Sets the input value if the given value is less than the existing value or if the key does not yet exist.
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value"></param>
        public void SetIfLess(string key, double value)
        {
            if (Lookup.ContainsKey(key) == false)
            {
                Lookup.Add(key, value);
            }
            else
            {
                var existingValue = Lookup[key];

                if (value < existingValue)
                {
                    Lookup[key] = value;
                }
            }
        }

        /// <summary>
        /// Sets the input value if the given value is greater than the existing value or if the key does not yet exist.
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value"></param>
        public void SetIfGreater(string key, double value)
        {
            if (Lookup.ContainsKey(key) == false)
            {
                Lookup.Add(key, value);
            }
            else
            {
                var existingValue = Lookup[key];

                if (value > existingValue)
                {
                    Lookup[key] = value;
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

        public double Get(string key)
            => Lookup[key];

        public KeyValuePair<string, double> Get(int index)
            => Lookup.ElementAt(index);

        public double Get(string key, double defaultValue)
        {
            if (Lookup.TryGetValue(key, out double value))
            {
                return value;
            }
            return defaultValue;
        }

        public bool TryGetValue(string key, [NotNullWhen(true)] out double outValue)
            => Lookup.TryGetValue(key, out outValue);
    }
}
