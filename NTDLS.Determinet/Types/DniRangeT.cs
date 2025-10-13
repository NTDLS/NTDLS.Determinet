using System.Numerics;

namespace NTDLS.Determinet.Types
{
    [Serializable]
    public struct DniRange<T>(T min, T max)
        where T : INumber<T>
    {
        public static readonly DniRange<T> Zero = new(T.Zero, T.Zero);

        private T min = min, max = max;

        public T Min
        {
            get { return min; }
            set { min = value; }
        }

        public T Max
        {
            get { return max; }
            set { max = value; }
        }

        public readonly T Length => max - min;
        public readonly T[] ToArray() => [min, max];
    }
}
