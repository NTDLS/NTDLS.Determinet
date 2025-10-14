using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace GenImages
{
    internal class Program
    {

        static void Main()
        {
            var fontFiles = Directory.GetFiles("C:\\NTDLS\\NTDLS.Determinet\\Fonts");

            string chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

            string outRoot = @"C:\NTDLS\NTDLS.Determinet\Sample Images\Training";
            int size = 130;

            var fc = new FontCollection();
            var families = fontFiles.Select(f => fc.Add(f)).ToArray();

            Directory.CreateDirectory(outRoot);

            foreach (char ch in chars)
            {
                int index = 0;
                foreach (var fam in families)
                {
                    float fontSize = 90;
                    var font = fam.CreateFont(fontSize, FontStyle.Regular);

                    using var img = new Image<Rgba32>(size, size, Color.White);

                    // Measure text to center it manually
                    var textOptionsForMeasure = new TextOptions(font);
                    var measured = TextMeasurer.MeasureSize(ch.ToString(), textOptionsForMeasure);

                    var origin = new PointF(
                        ((size / 2) - (measured.Width / 2f)),
                        ((size / 2) - (measured.Height / 2f))
                    );

                    img.Mutate(ctx =>
                    {
                        var drawOpts = new DrawingOptions
                        {
                            GraphicsOptions = new GraphicsOptions { Antialias = true }
                        };

                        ctx.DrawText(drawOpts, ch.ToString(), font, Color.Black, origin);

                        ctx.GaussianBlur(0.5f);
                    });

                    img.SaveAsPng(System.IO.Path.Combine(outRoot, $"{ch} {index++:000} {SanitizePathName(fam.Name)}.png"));
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
            var invalidChars = System.IO.Path.GetInvalidFileNameChars()
                .Concat(System.IO.Path.GetInvalidPathChars())
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