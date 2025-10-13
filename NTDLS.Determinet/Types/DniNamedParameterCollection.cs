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
            if(param.DataType != typeof(double))
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
            if(param.DataType != typeof(bool))
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
            if(param.DataType != typeof(float))
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
            if(param.DataType != typeof(int))
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

        public void Remove(DniNamedParameter param)
        {
            Values.Remove(param.Key);
            _cache.Remove(param.Key);
        }

        public void Remove(string key)
        {
            Values.Remove(key);
            _cache.Remove(key);
        }

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

        public object?[] ToArray()
        {
            var values = new object?[Values.Count];
            var keys = Values.Keys.ToList();
            for (int i = 0; i < keys.Count; i++)
            {
                values[i] = Values[keys[i]];
            }
            return values;
        }
    }
}
