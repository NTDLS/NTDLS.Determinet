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
        private static readonly MemoryCache _cache = new(new MemoryCacheOptions());

        [ProtoMember(1)] public Dictionary<string, string?> Values { get; private set; } = new(StringComparer.InvariantCultureIgnoreCase);

        public void Set(DniNamedParameter param, double value)
        {
            Values[param.Key] = value.ToString("n17");
            _cache.Set(param.Key, value);
        }

        public void Set(DniNamedParameter param, bool value)
        {
            Values[param.Key] = value.ToString();
            _cache.Set(param.Key, value);
        }

        public void Set(DniNamedParameter param, float value)
        {
            Values[param.Key] = value.ToString("n17");
            _cache.Set(param.Key, value);
        }

        public void Set(DniNamedParameter param, int value)
        {
            Values[param.Key] = value.ToString("");
            _cache.Set(param.Key, value);
        }

        public void Set(DniNamedParameter param, DniRange value)
        {
            Values[param.Key] = value.ToString();
            _cache.Set(param.Key, value);
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
