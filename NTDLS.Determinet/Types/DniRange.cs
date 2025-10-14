using ProtoBuf;

namespace NTDLS.Determinet.Types
{
    /// <summary>
    /// Represents a range defined by a minimum and maximum value, with utility methods for parsing, conversion, and
    /// range operations.
    /// </summary>
    /// <remarks>The <see cref="DniRange"/> struct is used to define a numeric range with a minimum and
    /// maximum value. It provides methods for converting the range to an array, parsing string representations of
    /// ranges, and calculating the range length. The struct also includes predefined ranges, such as <see
    /// cref="Zero"/>, for convenience.</remarks>
    [Serializable]
    public struct DniRange
    {
        /// <summary>
        /// Represents a range where both the start and end values are set to zero.
        /// </summary>
        /// <remarks>This static member provides a predefined range with zero values for both boundaries. 
        /// It can be used as a default or placeholder range in scenarios where no specific range is required.</remarks>
        public static readonly DniRange Zero = new(0, 0);

        /// <summary>
        /// Represents a predefined DNI range with both the start and end values set to 0.
        /// </summary>
        /// <remarks>This static field provides a constant instance of a <see cref="DniRange"/> where the
        /// range is effectively empty,  as both the start and end values are the same.</remarks>
        public static readonly DniRange One = new(1, 1);

        /// <summary>
        /// Gets or sets the minimum value.
        /// </summary>
        [ProtoMember(1)] public double Min { get; set; }

        /// <summary>
        /// Gets or sets the maximum allowable value.
        /// </summary>
        [ProtoMember(2)] public double Max { get; set; }

        /// <summary>
        /// Gets the length of the range, calculated as the difference between the maximum and minimum values.
        /// </summary>
        public readonly double Length => Max - Min;
        /// <summary>
        /// Converts the current range to an array containing the minimum and maximum values.
        /// </summary>
        /// <returns>An array of two elements, where the first element is the minimum value and the second element is the maximum
        /// value.</returns>
        public readonly double[] ToArray() => [Min, Max];
        /// <summary>
        /// Returns a string representation of the range defined by the <see cref="Min"/> and <see cref="Max"/> values.
        /// </summary>
        /// <returns>A string in the format "$[Min,Max]" where <c>Min</c> and <c>Max</c> represent the minimum and maximum values
        /// of the range, respectively.</returns>
        public override string ToString() => $"$[{Min},{Max}]";

        /// <summary>
        /// Converts a <see cref="DniRange"/> instance to an array of <see langword="double"/> values.
        /// </summary>
        /// <param name="range">The <see cref="DniRange"/> instance to convert.</param>
        public static implicit operator double[](DniRange range) => range.ToArray();

        /// <summary>
        /// Initializes a new instance of the <see cref="DniRange"/> class with the specified minimum and maximum
        /// values.
        /// </summary>
        /// <param name="min">The minimum value of the range. Must be less than or equal to <paramref name="max"/>.</param>
        /// <param name="max">The maximum value of the range. Must be greater than or equal to <paramref name="min"/>.</param>
        /// <exception cref="ArgumentException">Thrown if <paramref name="min"/> is greater than <paramref name="max"/>.</exception>
        public DniRange(double min, double max)
        {
            if (min > max)
                throw new ArgumentException("min cannot be greater than max.");
            Min = min;
            Max = max;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DniRange"/> class.
        /// </summary>
        public DniRange()
        {
        }

        /// <summary>
        /// Parses the specified string representation of a DNI range and returns an equivalent <see cref="DniRange"/>
        /// object.
        /// </summary>
        /// <param name="s">A string in the format <c>$[min,max]</c>, where <c>min</c> and <c>max</c> are numeric values representing
        /// the range.</param>
        /// <returns>A <see cref="DniRange"/> object that represents the parsed range.</returns>
        /// <exception cref="FormatException">Thrown if the input string is not in the correct format or if the numeric values cannot be parsed.</exception>
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

        /// <summary>
        /// Attempts to parse the specified string representation of a range into a <see cref="DniRange"/> object.
        /// </summary>
        /// <remarks>The input string must follow the format <c>"$[min,max]"</c>, where <c>min</c> and
        /// <c>max</c> are valid numeric values. If the string does not conform to this format or if the numeric values
        /// cannot be parsed, the method returns <see langword="false"/>.</remarks>
        /// <param name="s">The string to parse. The string must start with <c>"$["</c>, end with <c>"]"</c>, and contain two
        /// comma-separated numeric values.</param>
        /// <param name="range">When this method returns, contains the <see cref="DniRange"/> object parsed from the string, if the parsing
        /// succeeded; otherwise, the default value of <see cref="DniRange"/>.</param>
        /// <returns><see langword="true"/> if the string was successfully parsed into a <see cref="DniRange"/> object;
        /// otherwise, <see langword="false"/>.</returns>
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
