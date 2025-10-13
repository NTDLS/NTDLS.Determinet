namespace NTDLS.Determinet.Types
{
    [Serializable]
    public struct DniRange
    {
        private double min, max;

        public double Min
        {
            get { return min; }
            set { min = value; }
        }

        public double Max
        {
            get { return max; }
            set { max = value; }
        }

        public readonly double Length => max - min;
        public readonly double[] ToArray() => new[] { min, max };
        public static implicit operator double[](DniRange range) => range.ToArray();

        public DniRange(double min, double max)
        {
            this.min = min;
            this.max = max;
        }

        public override string ToString() => $"$[{min},{max}]";

        public static DniRange Parse(string s)
        {
            if (s.StartsWith("$[") && s.EndsWith("]"))
            {
                var parts = s[2..^1].Split(',');
                if (parts.Length == 2 && double.TryParse(parts[0], out var min) && double.TryParse(parts[1], out var max))
                {
                    return new DniRange(min, max);
                }
            }
            throw new FormatException("Invalid DniRange format.");
        }

        public static bool TryParse(string s, out DniRange range)
        {
            if (s.StartsWith("$[") && s.EndsWith("]"))
            {
                var parts = s[2..^1].Split(',');
                if (parts.Length == 2 && double.TryParse(parts[0], out var min) && double.TryParse(parts[1], out var max))
                {
                    range = new DniRange(min, max);
                    return true;
                }
            }
            range = default;
            return false;
        }
    }
}
