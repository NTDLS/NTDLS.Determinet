using Microsoft.Extensions.Caching.Memory;
using ProtoBuf;

namespace NTDLS.Determinet.Types
{
    /// <summary>
    /// Used to pass optional parameters to activation functions.
    /// </summary>
    [ProtoContract]
    public class DniNamedFunctionParameters
    {
        private static readonly MemoryCache _cache = new(new MemoryCacheOptions());

        [ProtoMember(1)] public Dictionary<string, string?> Values { get; private set; } = new(StringComparer.InvariantCultureIgnoreCase);

        public void Set(string key, double value)
        {
            Values[key] = value.ToString("n17");
            _cache.Set(key, value);
        }

        public void Set(string key, bool value)
        {
            Values[key] = value.ToString();
            _cache.Set(key, value);
        }

        public void Set(string key, float value)
        {
            Values[key] = value.ToString("n17");
            _cache.Set(key, value);
        }

        public void Set(string key, int value)
        {
            Values[key] = value.ToString("");
            _cache.Set(key, value);
        }

        public void Set(string key, DniRange value)
        {
            Values[key] = $"$[{value.Min},{value.Max}]";
            _cache.Set(key, value);
        }

        public void Remove(string key)
        {
            Values.Remove(key);
            _cache.Remove(key);
        }

        public T Get<T>(string key)
        {
            var result = _cache.GetOrCreate(key, entry =>
                {
                    if (Values.TryGetValue(key, out var s) && s != null)
                    {
                        // Special handling for DniRange
                        if (s.StartsWith("$[") && s.EndsWith("]") && typeof(T) == typeof(DniRange))
                        {
                            var parts = s[2..^1].Split(',');
                            if (parts.Length == 2 && double.TryParse(parts[0], out var min) && double.TryParse(parts[1], out var max))
                            {
                                return (T)(object)new DniRange(min, max);
                            }
                            throw new InvalidCastException($"Cannot convert value '{s}' to DniRange.");
                        }

                        return (T)Convert.ChangeType(s, typeof(T));
                    }
                    throw new KeyNotFoundException($"Key '{key}' not found in parameters.");
                }) ?? throw new KeyNotFoundException($"Key '{key}' not found in parameters.");

            return result;
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

        public T Get<T>(string key, T defaultValue)
        {
            var result = _cache.GetOrCreate(key, entry =>
            {
                if (Values.TryGetValue(key, out var s) && s != null)
                {
                    // Special handling for DniRange
                    if (s.StartsWith("$[") && s.EndsWith("]") && typeof(T) == typeof(DniRange))
                    {
                        var parts = s[2..^1].Split(',');
                        if (parts.Length == 2 && double.TryParse(parts[0], out var min) && double.TryParse(parts[1], out var max))
                        {
                            return (T)(object)new DniRange(min, max);
                        }
                        return defaultValue;
                    }

                    return (T)Convert.ChangeType(s, typeof(T));
                }
                return defaultValue;
            });

            return result ?? defaultValue;
        }
    }
}
