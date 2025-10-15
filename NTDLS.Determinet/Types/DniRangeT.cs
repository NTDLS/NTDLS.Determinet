using System.Numerics;

namespace NTDLS.Determinet.Types
{
    /// <summary>
    /// Represents a range defined by a minimum and maximum value of a numeric type.
    /// </summary>
    /// <remarks>This structure is designed to work with any numeric type that implements the <see
    /// cref="INumber{T}"/> interface. It provides properties to access the minimum and maximum values, as well as
    /// utility methods to calculate the range length and convert the range to an array.</remarks>
    /// <typeparam name="T">The numeric type of the range boundaries. Must implement <see cref="INumber{T}"/>.</typeparam>
    public struct DniRange<T>
        where T : INumber<T>
    {
        /// <summary>
        /// Represents a range where both the start and end values are set to zero.
        /// </summary>
        /// <remarks>This static member provides a predefined range with zero values for both boundaries. 
        /// It can be used as a default or placeholder range in scenarios where no specific range is required.</remarks>
        public static readonly DniRange<T> Zero = new(T.Zero, T.Zero);

        /// <summary>
        /// Represents a predefined range where both the start and end values are set to the minimum unit value of the
        /// type <typeparamref name="T"/>.
        /// </summary>
        /// <remarks>This static member provides a convenient way to reference a range where the start and
        /// end are both equal to the unit value of the type.</remarks>
        public static readonly DniRange<T> One = new(T.One, T.One);

        /// <summary>
        /// Gets or sets the minimum value in the collection.
        /// </summary>
        public T Min { get; set; }

        /// <summary>
        /// Gets or sets the maximum value allowed for the range.
        /// </summary>
        public T Max { get; set; }

        /// <summary>
        /// Gets the length of the range, calculated as the difference between the maximum and minimum values.
        /// </summary>
        public readonly T Length => Max - Min;

        /// <summary>
        /// Converts the range to an array containing the minimum and maximum values.
        /// </summary>
        /// <returns>An array of two elements, where the first element is the minimum value and the second element is the maximum
        /// value of the range.</returns>
        public readonly T[] ToArray() => [Min, Max];

        /// <summary>
        /// Initializes a new instance of the <see cref="DniRange{T}"/> class with the specified minimum and maximum
        /// values.
        /// </summary>
        /// <param name="min">The minimum value of the range.</param>
        /// <param name="max">The maximum value of the range. Must be greater than or equal to <paramref name="min"/>.</param>
        /// <exception cref="ArgumentException">Thrown if <paramref name="min"/> is greater than <paramref name="max"/>.</exception>
        public DniRange(T min, T max)
        {
            if (min > max)
                throw new ArgumentException("min cannot be greater than max.");
            Min = min;
            Max = max;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DniRange{T}"/> class with the minimum and maximum values set to
        /// their default values.
        /// </summary>
        /// <remarks>The default values for <see cref="Min"/> and <see cref="Max"/> are determined by the
        /// default value of the generic type <typeparamref name="T"/>.</remarks>
        public DniRange()
        {
            Min = T.Zero;
            Max = T.Zero;
        }
    }
}
