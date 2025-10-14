using ProtoBuf;
using System.Diagnostics.CodeAnalysis;

namespace NTDLS.Determinet.Types
{
    /// <summary>
    /// When the input and/or output nodes are aliased, this class is used to "interface" with the
    /// neural network by allowing you to supply named values instead of using neuron ordinals.
    /// </summary>
    [ProtoContract]
    public class DniNamedLabelValues
    {
        /// <summary>
        /// Gets the dictionary used to store key-value pairs, where the key is a string and the value is a double.
        /// </summary>
        [ProtoMember(1)] public Dictionary<string, double> Lookup { get; private set; } = new();

        /// <summary>
        /// Gets an array of key-value pairs representing the current lookup values.
        /// </summary>
        public KeyValuePair<string, double>[] Values
            => Lookup.ToArray();

        /// <summary>
        /// Sets the specified key to the given value in the lookup.
        /// </summary>
        /// <remarks>If the key already exists in the lookup, its value will be updated. If the key does
        /// not exist, it will be added.</remarks>
        /// <param name="key">The key to associate with the specified value. Cannot be <see langword="null"/>.</param>
        /// <param name="value">The value to assign to the specified key.</param>
        public void Set(string key, double value)
            => Lookup[key] = value;

        /// <summary>
        /// Returns the key-value pair with the maximum value in the collection.
        /// </summary>
        /// <remarks>This method uses the value of each key-value pair to determine the maximum.  If
        /// multiple pairs have the same maximum value, the first one encountered is returned.</remarks>
        /// <returns>A <see cref="KeyValuePair{TKey, TValue}"/> representing the key and the maximum value in the collection. If
        /// the collection is empty, the behavior depends on the underlying implementation of <see cref="Lookup"/>.</returns>
        public KeyValuePair<string, double> Max()
            => Lookup.MaxBy(kv => kv.Value);

        /// <summary>
        /// Finds the key-value pair with the smallest value in the collection.
        /// </summary>
        /// <remarks>This method uses the value of each key-value pair to determine the minimum. If
        /// multiple key-value pairs have the same minimum value, the first one encountered is returned.</remarks>
        /// <returns>A <see cref="KeyValuePair{TKey, TValue}"/> representing the key-value pair with the smallest value. If the
        /// collection is empty, the behavior is undefined.</returns>
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

        /// <summary>
        /// Converts the current collection of key-value pairs to an array of values.
        /// </summary>
        /// <returns>An array of <see cref="double"/> values representing the values in the collection. The order of the values
        /// in the array corresponds to the order of the keys in the collection.</returns>
        public double[] ToArray()
            => Values.Select(kv => kv.Value).ToArray();

        /// <summary>
        /// Retrieves all keys from the collection.
        /// </summary>
        /// <returns>An array of strings containing the keys. The array will be empty if the collection contains no elements.</returns>
        public string[] Keys()
            => Values.Select(kv => kv.Key).ToArray();

        /// <summary>
        /// Retrieves the value associated with the specified key.
        /// </summary>
        /// <param name="key">The key whose associated value is to be retrieved. Cannot be null.</param>
        /// <returns>The value associated with the specified key.</returns>
        public double Get(string key)
            => Lookup[key];

        /// <summary>
        /// Retrieves the key-value pair at the specified index in the collection.
        /// </summary>
        /// <param name="index">The zero-based index of the key-value pair to retrieve.</param>
        /// <returns>A <see cref="KeyValuePair{TKey, TValue}"/> representing the key and value at the specified index.</returns>
        public KeyValuePair<string, double> Get(int index)
            => Lookup.ElementAt(index);

        /// <summary>
        /// Retrieves the value associated with the specified key, or returns a default value if the key is not found.
        /// </summary>
        /// <param name="key">The key whose associated value is to be retrieved. Cannot be <see langword="null"/>.</param>
        /// <param name="defaultValue">The value to return if the specified key is not found.</param>
        /// <returns>The value associated with the specified key if it exists; otherwise, <paramref name="defaultValue"/>.</returns>
        public double Get(string key, double defaultValue)
        {
            if (Lookup.TryGetValue(key, out double value))
            {
                return value;
            }
            return defaultValue;
        }

        /// <summary>
        /// Attempts to retrieve the value associated with the specified key.
        /// </summary>
        /// <param name="key">The key whose associated value is to be retrieved.</param>
        /// <param name="outValue">When this method returns, contains the value associated with the specified key, if the key is found;
        /// otherwise, the default value for the type of the value parameter. This parameter is passed uninitialized.</param>
        /// <returns><see langword="true"/> if the key was found and the value was successfully retrieved; otherwise, <see
        /// langword="false"/>.</returns>
        public bool TryGetValue(string key, [NotNullWhen(true)] out double outValue)
            => Lookup.TryGetValue(key, out outValue);
    }
}
