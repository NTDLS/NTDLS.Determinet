using ImageMagick;
using ImageMagick.Drawing;

namespace GenImages
{
    internal class Program
    {

        static void Main()
        {
            var fontFiles = Directory.GetFiles("C:\\NTDLS\\NTDLS.Determinet\\Fonts");

            string chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

            string outRoot = @"C:\NTDLS\NTDLS.Determinet\Sample Images\Training";
            uint size = 130;

            Directory.CreateDirectory(outRoot);

            foreach (char ch in chars)
            {
                int index = 0;
                foreach (var fontFile in fontFiles)
                {
                    double fontSize = 90;

                    using var img = new MagickImage(MagickColors.White, size, size);

                    // Gravity.Center centers the glyph's text box on the canvas.
                    new Drawables()
                        .Font(fontFile)
                        .FontPointSize(fontSize)
                        .FillColor(MagickColors.Black)
                        .Gravity(Gravity.Center)
                        .Text(0, 0, ch.ToString())
                        .Draw(img);

                    img.GaussianBlur(0, 0.5);

                    var fontName = Path.GetFileNameWithoutExtension(fontFile);
                    img.Write(Path.Combine(outRoot, $"{ch} {index++:000} {SanitizePathName(fontName)}.png"), MagickFormat.Png);
                }

                Console.WriteLine($"Generated {ch}");
            }

            Console.WriteLine("Done.");
        }

        public static string SanitizePathName(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
                return string.Empty;

            // Combine invalid file + path chars, plus some reserved ones like ':' or '?'
            var invalidChars = Path.GetInvalidFileNameChars()
                .Concat(Path.GetInvalidPathChars())
                .Distinct()
                .ToArray();

            foreach (char c in invalidChars)
            {
                input = input.Replace(c.ToString(), string.Empty);
            }

            return input.Trim();
        }
    }
}
