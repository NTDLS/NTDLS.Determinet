using NTDLS.Determinet;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace OCR
{
    /// <summary>
    /// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /// This is just a POC, this is NOT complete and does NOT work!!!!
    /// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /// </summary>
    internal class Program
    {
        private static char[] distinctCharacters = new char[] {
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'a', 'b', 'c', 'd', 'e', 'f','g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                'A', 'B', 'C', 'D', 'E', 'F','G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
            };

        const int _imageWidth = 64;              // Downscale for faster processing with minimal quality loss.
        const int _imageHeight = 64;             // Downscale for faster processing with minimal quality loss.

        static void Main()
        {
            var dni = DniNeuralNetwork.LoadFromFile(@"C:\NTDLS\NTDLS.Determinet\TestHarness\bin\Release\net9.0\trained.dni")
                ?? throw new Exception("Failed to load the network from file.");

            var inputPath = "C:\\NTDLS\\NTDLS.Determinet\\OCR\\screenshot.png";
            using var image = Image.Load<Rgba32>(inputPath);

            // Convert to grayscale
            image.Mutate(x => x.Grayscale());

            // Binarize (black text, white background)
            float threshold = 0.8f;
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    var pixel = image[x, y];
                    float brightness = pixel.R / 255f;
                    image[x, y] = brightness < threshold ? Color.Black : Color.White;
                }
            }

            // --- STEP 1: Find horizontal line segments ---
            var lineBounds = new List<(int Y, int Height)>();
            bool inLine = false;
            int startY = 0;

            for (int y = 0; y < image.Height; y++)
            {
                bool hasBlackPixel = false;
                for (int x = 0; x < image.Width; x++)
                {
                    if (image[x, y].R < 128)
                    {
                        hasBlackPixel = true;
                        break;
                    }
                }

                if (hasBlackPixel && !inLine)
                {
                    inLine = true;
                    startY = y;
                }
                else if (!hasBlackPixel && inLine)
                {
                    inLine = false;
                    int height = y - startY;
                    if (height > 3) lineBounds.Add((startY, height));
                }
            }

            if (inLine)
                lineBounds.Add((startY, image.Height - startY));

            Directory.CreateDirectory("chars");

            int lineIndex = 0;
            foreach (var (y, height) in lineBounds)
            {
                using var lineImg = image.Clone(ctx => ctx.Crop(new Rectangle(0, y, image.Width, height)));

                // --- STEP 2: Find character segments within the line ---
                var charBounds = new List<(int X, int Width)>();
                bool inChar = false;
                int startX = 0;

                for (int x = 0; x < lineImg.Width; x++)
                {
                    bool hasBlackPixel = false;
                    for (int yy = 0; yy < lineImg.Height; yy++)
                    {
                        if (lineImg[x, yy].R < 128)
                        {
                            hasBlackPixel = true;
                            break;
                        }
                    }

                    if (hasBlackPixel && !inChar)
                    {
                        inChar = true;
                        startX = x;
                    }
                    else if (!hasBlackPixel && inChar)
                    {
                        inChar = false;
                        int width = x - startX;
                        if (width > 1) charBounds.Add((startX, width));
                    }
                }
                if (inChar)
                    charBounds.Add((startX, lineImg.Width - startX));

                int charIndex = 0;
                foreach (var (x, width) in charBounds)
                {
                    var rect = new Rectangle(x, 0, width, lineImg.Height);
                    using var charImg = lineImg.Clone(ctx => ctx.Crop(rect));
                    //charImg.Save($"C:\\NTDLS\\NTDLS.Determinet\\OCR\\debug_out\\line{lineIndex:00}_char{charIndex:000}.png");


                    var inputBits = GetImageGrayscaleBytes(charImg, _imageWidth, _imageHeight);

                    if (inputBits != null)
                    {
                        try
                        {
                            var outputs = dni.Forward(inputBits);
                            var prediction = DniUtility.IndexOfMaxValue(outputs, out var confidence);
                            Console.Write($"{distinctCharacters[prediction]}");
                        }
                        catch
                        {
                        }
                    }

                    charIndex++;
                }

                lineIndex++;
            }

            Console.WriteLine($"Extracted characters from {lineIndex} lines.");
        }

        static int dbgIndex = 0;

        private static double[]? GetImageGrayscaleBytes(Image<Rgba32> img, int resizeWidth, int resizeHeight)
        {
            int width = img.Width;
            int height = img.Height;

            // Detect bounds of non-white pixels
            int threshold = 250;
            int left = width, right = 0, top = height, bottom = 0;

            img.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < width; x++)
                    {
                        Rgba32 p = row[x];
                        if (p.R < threshold || p.G < threshold || p.B < threshold)
                        {
                            if (x < left) left = x;
                            if (x > right) right = x;
                            if (y < top) top = y;
                            if (y > bottom) bottom = y;
                        }
                    }
                }
            });

            // No ink detected — blank image
            if (right <= left || bottom <= top)
                return null;

            // Add margin and clamp to image edges
            int margin = 5;
            left = Math.Max(0, left - margin);
            top = Math.Max(0, top - margin);
            right = Math.Min(width - 1, right + margin);
            bottom = Math.Min(height - 1, bottom + margin);

            int cropWidth = right - left + 1;
            int cropHeight = bottom - top + 1;

            // Crop region of interest
            var bounds = new Rectangle(left, top, cropWidth, cropHeight);
            using var cropped = img.Clone(ctx => ctx.Crop(bounds));

            // Create a square white canvas (to center drawing)
            int squareSize = Math.Max(cropWidth + (margin * 2), cropHeight + (margin * 2));
            using var squareCanvas = new Image<Rgba32>(squareSize, squareSize, Color.White);

            int offsetX = (squareSize - cropWidth) / 2;
            int offsetY = (squareSize - cropHeight) / 2;
            squareCanvas.Mutate(ctx => ctx.DrawImage(cropped, new Point(offsetX, offsetY), 1f));

            using var resized = squareCanvas.Clone(ctx => ctx.Resize(resizeWidth, resizeHeight));

            resized.Save($"C:\\NTDLS\\NTDLS.Determinet\\OCR\\debug_out\\line{dbgIndex++}.png");

            // Convert to grayscale and normalize [0..1]
            var pixels = new double[resizeWidth * resizeHeight];
            int index = 0;

            resized.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < resizeHeight; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < resizeWidth; x++)
                    {
                        Rgba32 p = row[x];
                        double gray = (0.299 * p.R + 0.587 * p.G + 0.114 * p.B) / 255.0;
                        pixels[index++] = gray;
                    }
                }
            });

            return pixels;
        }
    }
}
