using Microsoft.Extensions.Caching.Memory;
using ProtoBuf;

namespace NTDLS.Determinet.Types
{
    /// <summary>
    /// Used to pass optional parameters to activation functions.
    /// </summary>
    [ProtoContract]
    public class DniNamedParameterCollection
    {
        /// <summary>
        /// Represents an in-memory cache used to store and retrieve data efficiently within the application.
        /// </summary>
        /// <remarks>This field is initialized with default <see cref="MemoryCacheOptions"/> and is
        /// intended for internal use only. It provides a mechanism for caching data in memory to improve performance by
        /// reducing the need for repeated expensive operations, such as database queries or external service
        /// calls.</remarks>
        private readonly MemoryCache _cache = new(new MemoryCacheOptions());

        [ProtoMember(1)] public Dictionary<string, string?> Values { get; private set; } = new(StringComparer.InvariantCultureIgnoreCase);

        /// <summary>
        /// Sets the specified parameter to the given value.
        /// </summary>
        /// <remarks>The value is stored as a string formatted to 17 decimal places. This method also
        /// updates an internal cache with the specified value.</remarks>
        /// <param name="param">The named parameter to set. The <see cref="DniNamedParameter.Key"/> property is used as the identifier.</param>
        /// <param name="value">The value to associate with the specified parameter.</param>
        public void Set(DniNamedParameter param, double value)
        {
            if (param.DataType != typeof(double))
                throw new ArgumentException($"Parameter '{param.Key}' is of type '{param.DataType.Name}', not 'double'.");

            Values[param.Key] = value.ToString("n17");
            _cache.Set(param.Key, value);
        }

        /// <summary>
        /// Sets the specified parameter to the given boolean value.
        /// </summary>
        /// <remarks>The method updates the internal storage with the string representation of the value
        /// and caches the boolean value for the specified parameter key.</remarks>
        /// <param name="param">The named parameter to set. The <see cref="DniNamedParameter.Key"/> property is used as the identifier.</param>
        /// <param name="value">The boolean value to assign to the parameter.</param>
        public void Set(DniNamedParameter param, bool value)
        {
            if (param.DataType != typeof(bool))
                throw new ArgumentException($"Parameter '{param.Key}' is of type '{param.DataType.Name}', not 'bool'.");

            Values[param.Key] = value.ToString();
            _cache.Set(param.Key, value);
        }

        /// <summary>
        /// Sets the specified parameter to the given floating-point value.
        /// </summary>
        /// <remarks>The value is stored with a precision of up to 17 significant digits. The method also
        /// updates an internal cache with the specified value.</remarks>
        /// <param name="param">The named parameter whose value is to be set. The <see cref="DniNamedParameter.Key"/> property is used as
        /// the identifier.</param>
        /// <param name="value">The floating-point value to associate with the specified parameter.</param>
        public void Set(DniNamedParameter param, float value)
        {
            if (param.DataType != typeof(float))
                throw new ArgumentException($"Parameter '{param.Key}' is of type '{param.DataType.Name}', not 'float'.");

            Values[param.Key] = value.ToString("n17");
            _cache.Set(param.Key, value);
        }

        /// <summary>
        /// Sets the specified parameter to the given integer value.
        /// </summary>
        /// <remarks>This method updates the internal collection and cache with the provided value. The
        /// parameter's key is used to identify the value in both storage mechanisms.</remarks>
        /// <param name="param">The named parameter whose value is to be set. The <see cref="DniNamedParameter.Key"/> property is used as
        /// the key.</param>
        /// <param name="value">The integer value to associate with the specified parameter.</param>
        public void Set(DniNamedParameter param, int value)
        {
            if (param.DataType != typeof(int))
                throw new ArgumentException($"Parameter '{param.Key}' is of type '{param.DataType.Name}', not 'int'.");

            Values[param.Key] = value.ToString("");
            _cache.Set(param.Key, value);
        }

        /// <summary>
        /// Sets the specified parameter to the given range value.
        /// </summary>
        /// <remarks>This method updates the internal collection with the string representation of the
        /// range value  and caches the range value for the specified parameter key.</remarks>
        /// <param name="param">The named parameter whose value is to be set. The <see cref="DniNamedParameter.Key"/> property is used as
        /// the key.</param>
        /// <param name="value">The range value to associate with the specified parameter.</param>
        public void Set(DniNamedParameter param, DniRange value)
        {
            Values[param.Key] = value.ToString();
            _cache.Set(param.Key, value);
        }

        /// <summary>
        /// Sets a user-defined parameter value.
        /// </summary>
        public void Set(string key, double value)
        {
            key = $"User.{key}";
            Values[key] = value.ToString("n17");
            _cache.Set(key, value);
        }

        /// <summary>
        /// Sets a user-defined parameter value.
        /// </summary>
        public void Set(string key, bool value)
        {
            key = $"User.{key}";
            Values[key] = value.ToString();
            _cache.Set(key, value);
        }

        /// <summary>
        /// Sets a user-defined parameter value.
        /// </summary>
        public void Set(string key, float value)
        {
            key = $"User.{key}";
            Values[key] = value.ToString("n17");
            _cache.Set(key, value);
        }

        /// <summary>
        /// Sets a user-defined parameter value.
        /// </summary>
        public void Set(string key, int value)
        {
            key = $"User.{key}";
            Values[key] = value.ToString("");
            _cache.Set(key, value);
        }

        /// <summary>
        /// Sets a user-defined parameter value.
        /// </summary>
        public void Set(string key, DniRange value)
        {
            key = $"User.{key}";
            Values[key] = value.ToString();
            _cache.Set(key, value);
        }

        /// <summary>
        /// Removes the specified parameter from the collection.
        /// </summary>
        /// <remarks>This method removes the parameter from both the internal collection and the cache. 
        /// If the specified parameter does not exist in the collection, no action is taken.</remarks>
        /// <param name="param">The parameter to remove, identified by its key. The key must not be null.</param>
        public void Remove(DniNamedParameter param)
        {
            Values.Remove(param.Key);
            _cache.Remove(param.Key);
        }

        /// <summary>
        /// Removes the specified key and its associated value from the collection.
        /// </summary>
        /// <remarks>This method removes the key and its associated value from both the primary collection
        /// and the cache. If the key does not exist in the collection, no action is taken.</remarks>
        /// <param name="key">The key of the item to remove. The key is prefixed with "User." internally. Cannot be <see langword="null"/> or empty.</param>
        public void Remove(string key)
        {
            key = $"User.{key}";

            Values.Remove(key);
            _cache.Remove(key);
        }

        /// <summary>
        /// Retrieves a value of the specified type from the cache or computes it based on the provided parameter.
        /// </summary>
        /// <typeparam name="T">The type of the value to retrieve or compute.</typeparam>
        /// <param name="param">The parameter containing the key, data type, and default value used to retrieve or compute the value.</param>
        /// <returns>The value associated with the specified key, converted to the specified type. If the key is not found, the
        /// default value is used.</returns>
        /// <exception cref="KeyNotFoundException">Thrown if the key is not found, the conversion fails, or no default value is provided for the specified key.</exception>
        public T Get<T>(DniNamedParameter param)
        {
            var result = _cache.GetOrCreate(param.Key, entry =>
            {
                if (Values.TryGetValue(param.Key, out var s) && s != null)
                {
                    if (param.DataType == typeof(DniRange))
                    {
                        return (T)(object)DniRange.Parse(s);
                    }

                    return (T)Convert.ChangeType(s, param.DataType);
                }
                return (T)Convert.ChangeType(param.DefaultValue, param.DataType);
            }) ?? throw new KeyNotFoundException($"Key not found, conversion failed, or no default is present for '{param.Key}'.");

            return result;
        }

        /// <summary>
        /// Retrieves a value of the specified type from the cache or returns the provided default value.
        /// </summary>
        /// <remarks>The method attempts to retrieve the value associated with the key specified in
        /// <paramref name="param"/> from the cache.  If the value is not found or cannot be converted to the specified
        /// type <typeparamref name="T"/>, the <paramref name="defaultValue"/> is returned.</remarks>
        /// <typeparam name="T">The type of the value to retrieve.</typeparam>
        /// <param name="param">The named parameter that specifies the key and expected data type of the value.</param>
        /// <param name="defaultValue">The default value to return if the key is not found or the value cannot be converted to the specified type.</param>
        /// <returns>The value associated with the specified key, converted to the specified type, or the provided default value
        /// if the key is not found or the conversion fails.</returns>
        public T? Get<T>(DniNamedParameter param, T defaultValue)
        {
            var result = _cache.GetOrCreate(param.Key, entry =>
            {
                if (Values.TryGetValue(param.Key, out var s) && s != null)
                {
                    if (param.DataType == typeof(DniRange))
                    {
                        return (T)(object)DniRange.Parse(s);
                    }

                    return (T)Convert.ChangeType(s, param.DataType);
                }
                return defaultValue;
            });

            return result ?? defaultValue;
        }

        /// <summary>
        /// Retrieves a value from the cache or computes it if not present, converting it to the specified type.
        /// </summary>
        /// <typeparam name="T">The type to which the cached value should be converted.</typeparam>
        /// <param name="key">The key associated with the value to retrieve. The key is prefixed with "User." internally.</param>
        /// <returns>The value associated with the specified key, converted to the specified type.</returns>
        /// <exception cref="KeyNotFoundException">Thrown if the key does not exist in the cache, the conversion to the specified type fails,  or no default
        /// value is provided for the key.</exception>
        public T Get<T>(string key)
        {
            key = $"User.{key}";

            var result = _cache.GetOrCreate(key, entry =>
            {
                if (Values.TryGetValue(key, out var s) && s != null)
                {
                    if (typeof(T) == typeof(DniRange))
                    {
                        return (T)(object)DniRange.Parse(s);
                    }

                    return (T)Convert.ChangeType(s, typeof(T));
                }
                throw new KeyNotFoundException($"Key not found, conversion failed, or no default is present for '{key}'.");
            }) ?? throw new KeyNotFoundException($"Key not found, conversion failed, or no default is present for '{key}'.");

            return result;
        }

        /// <summary>
        /// Retrieves a value of the specified type from the cache, or returns a default value if the key is not found.
        /// </summary>
        /// <remarks>The method attempts to retrieve the value from the cache using the provided key. If
        /// the value is found, it is converted to the specified type <typeparamref name="T"/>. If the value cannot be
        /// found or converted, the <paramref name="defaultValue"/> is returned.</remarks>
        /// <typeparam name="T">The type of the value to retrieve.</typeparam>
        /// <param name="key">The key associated with the value to retrieve. The key is automatically prefixed with "User.".</param>
        /// <param name="defaultValue">The default value to return if the key is not found or the value cannot be converted to the specified type.</param>
        /// <returns>The value associated with the specified key, converted to the specified type, or the <paramref
        /// name="defaultValue"/> if the key is not found or the conversion fails.</returns>
        public T? Get<T>(string key, T defaultValue)
        {
            key = $"User.{key}";

            var result = _cache.GetOrCreate(key, entry =>
            {
                if (Values.TryGetValue(key, out var s) && s != null)
                {
                    if (typeof(T) == typeof(DniRange))
                    {
                        return (T)(object)DniRange.Parse(s);
                    }

                    return (T)Convert.ChangeType(s, typeof(T));
                }
                return defaultValue;
            });

            return result ?? defaultValue;
        }

        /// <summary>
        /// Converts the collection of values to an array.
        /// </summary>
        /// <returns>An array of objects containing the values from the collection. The array may contain null values if the
        /// collection includes null entries.</returns>
        public object?[] ToArray()
            => Values.Select(kv => kv.Value).ToArray();

        /// <summary>
        /// Retrieves all keys from the collection.
        /// </summary>
        /// <returns>An array of strings containing the keys. The array will be empty if the collection contains no elements.</returns>
        public string[] Keys()
            => Values.Select(kv => kv.Key).ToArray();
    }
}
