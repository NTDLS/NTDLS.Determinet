using ProtoBuf;
using System.Collections.Concurrent;
using System.Globalization;

namespace NTDLS.Determinet.Types
{
    /// <summary>
    /// Used to pass optional parameters to activation functions, layers and the network.
    /// </summary>
    /// <remarks>Values are persisted as culture-invariant, round-trippable strings so that a model saved on one machine
    /// loads identically on any other, and parsed values are cached so hot-path lookups do not re-parse.</remarks>
    [ProtoContract]
    public class DniNamedParameterCollection
    {
        private const string UserPrefix = "User.";

        /// <summary>
        /// Parsed values keyed by parameter key. Invalidated whenever <see cref="Values"/> changes.
        /// </summary>
        private readonly ConcurrentDictionary<string, object> _cache = new(StringComparer.InvariantCultureIgnoreCase);

        /// <summary>
        /// Gets the collection of key-value pairs, where the keys are case-insensitive strings, and the values are
        /// the invariant-culture string representations of the parameter values.
        /// </summary>
        [ProtoMember(1)] public Dictionary<string, string?> Values { get; private set; } = new(StringComparer.InvariantCultureIgnoreCase);

        #region Set.

        /// <summary>
        /// Sets the specified parameter to the given value.
        /// </summary>
        public void Set(DniNamedParameter param, double value)
            => SetValue(EnsureType(param, typeof(double)), value, value.ToString("R", CultureInfo.InvariantCulture));

        /// <summary>
        /// Sets the specified parameter to the given value.
        /// </summary>
        public void Set(DniNamedParameter param, bool value)
            => SetValue(EnsureType(param, typeof(bool)), value, value.ToString(CultureInfo.InvariantCulture));

        /// <summary>
        /// Sets the specified parameter to the given value.
        /// </summary>
        public void Set(DniNamedParameter param, float value)
            => SetValue(EnsureType(param, typeof(float)), value, value.ToString("R", CultureInfo.InvariantCulture));

        /// <summary>
        /// Sets the specified parameter to the given value.
        /// </summary>
        public void Set(DniNamedParameter param, int value)
            => SetValue(EnsureType(param, typeof(int)), value, value.ToString(CultureInfo.InvariantCulture));

        /// <summary>
        /// Sets the specified parameter to the given value.
        /// </summary>
        public void Set(DniNamedParameter param, DniRange value)
            => SetValue(EnsureType(param, typeof(DniRange)), value, value.ToString());

        /// <summary>
        /// Sets a user-defined value. User keys are namespaced so they cannot collide with built-in parameters.
        /// </summary>
        public void Set(string key, double value)
            => SetValue(UserPrefix + key, value, value.ToString("R", CultureInfo.InvariantCulture));

        /// <summary>
        /// Sets a user-defined value. User keys are namespaced so they cannot collide with built-in parameters.
        /// </summary>
        public void Set(string key, bool value)
            => SetValue(UserPrefix + key, value, value.ToString(CultureInfo.InvariantCulture));

        /// <summary>
        /// Sets a user-defined value. User keys are namespaced so they cannot collide with built-in parameters.
        /// </summary>
        public void Set(string key, float value)
            => SetValue(UserPrefix + key, value, value.ToString("R", CultureInfo.InvariantCulture));

        /// <summary>
        /// Sets a user-defined value. User keys are namespaced so they cannot collide with built-in parameters.
        /// </summary>
        public void Set(string key, int value)
            => SetValue(UserPrefix + key, value, value.ToString(CultureInfo.InvariantCulture));

        /// <summary>
        /// Sets a user-defined value. User keys are namespaced so they cannot collide with built-in parameters.
        /// </summary>
        public void Set(string key, DniRange value)
            => SetValue(UserPrefix + key, value, value.ToString());

        #endregion

        #region Remove.

        /// <summary>
        /// Removes the specified parameter so that its default value applies again.
        /// </summary>
        public void Remove(DniNamedParameter param)
        {
            Values.Remove(param.Key);
            _cache.TryRemove(param.Key, out _);
        }

        /// <summary>
        /// Removes the specified user-defined value.
        /// </summary>
        public void Remove(string key)
        {
            key = UserPrefix + key;
            Values.Remove(key);
            _cache.TryRemove(key, out _);
        }

        #endregion

        #region Get.

        /// <summary>
        /// Gets the value of the specified parameter, or the parameter's default value if it has not been set.
        /// </summary>
        public T Get<T>(DniNamedParameter param)
        {
            if (TryGetValue<T>(param.Key, out var value))
                return value;

            return ConvertTo<T>(param.DefaultValue);
        }

        /// <summary>
        /// Gets the value of the specified parameter, or <paramref name="defaultValue"/> if it has not been set.
        /// </summary>
        public T Get<T>(DniNamedParameter param, T defaultValue)
            => TryGetValue<T>(param.Key, out var value) ? value : defaultValue;

        /// <summary>
        /// Gets the specified user-defined value.
        /// </summary>
        /// <exception cref="KeyNotFoundException">Thrown when the key has not been set.</exception>
        public T Get<T>(string key)
            => TryGetValue<T>(UserPrefix + key, out var value) ? value
                : throw new KeyNotFoundException($"Key not found: '{key}'.");

        /// <summary>
        /// Gets the specified user-defined value, or <paramref name="defaultValue"/> if it has not been set.
        /// </summary>
        public T Get<T>(string key, T defaultValue)
            => TryGetValue<T>(UserPrefix + key, out var value) ? value : defaultValue;

        #endregion

        /// <summary>
        /// Gets the raw string values.
        /// </summary>
        public object?[] ToArray()
            => Values.Select(kv => kv.Value).ToArray();

        /// <summary>
        /// Gets the stored keys.
        /// </summary>
        public string[] Keys()
            => Values.Select(kv => kv.Key).ToArray();

        #region Internals.

        private static string EnsureType(DniNamedParameter param, Type type)
        {
            if (param.DataType != type)
                throw new ArgumentException($"Parameter '{param.Key}' is of type '{param.DataType.Name}', not '{type.Name}'.");
            return param.Key;
        }

        private void SetValue(string key, object value, string text)
        {
            Values[key] = text;
            _cache[key] = value;
        }

        private bool TryGetValue<T>(string key, out T value)
        {
            if (_cache.TryGetValue(key, out var cached))
            {
                if (cached is T typed)
                {
                    value = typed;
                    return true;
                }
            }
            else if (Values.TryGetValue(key, out var text) && text != null)
            {
                cached = text;
            }
            else
            {
                value = default!;
                return false;
            }

            // Either the stored text has not been parsed yet, or it was cached as a different type
            // (e.g. set as int, read as double). Convert and cache the requested type.
            value = ConvertTo<T>(cached);
            _cache[key] = value!;
            return true;
        }

        private static T ConvertTo<T>(object value)
        {
            if (value is T typed)
                return typed;

            var target = Nullable.GetUnderlyingType(typeof(T)) ?? typeof(T);

            if (value is string text)
            {
                if (target == typeof(DniRange))
                    return (T)(object)DniRange.Parse(text);

                if (target == typeof(bool))
                    return (T)(object)bool.Parse(text);

                if (target == typeof(string))
                    return (T)(object)text;

                // AllowThousands keeps files written by older versions (which used culture-formatted "n17") loadable.
                // Parsing through double also lets integral types read values such as "12.00000".
                var number = double.Parse(text, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture);
                value = number;
                if (target == typeof(double))
                    return (T)value;
            }

            return (T)Convert.ChangeType(value, target, CultureInfo.InvariantCulture);
        }

        #endregion
    }
}
