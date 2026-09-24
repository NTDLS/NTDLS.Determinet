using ImageMagick;
using NTDLS.Determinet;

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
            using var image = new MagickImage(inputPath);

            // Binarize (black text, white background): grayscale, then anything below 80% brightness becomes black.
            image.BackgroundColor = MagickColors.White;
            image.Alpha(AlphaOption.Remove);
            image.Grayscale();
            image.Threshold(new Percentage(80));

            int imageWidth = (int)image.Width;
            int imageHeight = (int)image.Height;
            var gray = ReadGray(image);
            bool IsInk(int x, int y) => gray[y * imageWidth + x] < 128;

            // --- STEP 1: Find horizontal line segments ---
            var lineBounds = new List<(int Y, int Height)>();
            bool inLine = false;
            int startY = 0;

            for (int y = 0; y < imageHeight; y++)
            {
                bool hasBlackPixel = false;
                for (int x = 0; x < imageWidth; x++)
                {
                    if (IsInk(x, y))
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
                lineBounds.Add((startY, imageHeight - startY));

            Directory.CreateDirectory("chars");

            int lineIndex = 0;
            foreach (var (y, height) in lineBounds)
            {
                // --- STEP 2: Find character segments within the line ---
                var charBounds = new List<(int X, int Width)>();
                bool inChar = false;
                int startX = 0;

                for (int x = 0; x < imageWidth; x++)
                {
                    bool hasBlackPixel = false;
                    for (int yy = y; yy < y + height; yy++)
                    {
                        if (IsInk(x, yy))
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
                    charBounds.Add((startX, imageWidth - startX));

                int charIndex = 0;
                foreach (var (x, width) in charBounds)
                {
                    using var charImg = image.CloneArea(new MagickGeometry(x, y, (uint)width, (uint)height));
                    //charImg.Write($"C:\\NTDLS\\NTDLS.Determinet\\OCR\\debug_out\\line{lineIndex:00}_char{charIndex:000}.png");

                    var inputBits = GetImageGrayscaleBytes(charImg, _imageWidth, _imageHeight);

                    if (inputBits != null)
                    {
                        try
                        {
                            var outputs = dni.Forward(inputBits);
                            var prediction = outputs.IndexOfMaxValue(out var confidence);
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

        private static double[]? GetImageGrayscaleBytes(IMagickImage<byte> img, int resizeWidth, int resizeHeight)
        {
            int width = (int)img.Width;
            int height = (int)img.Height;

            // Detect bounds of non-white pixels
            int threshold = 250;
            int left = width, right = 0, top = height, bottom = 0;

            var gray = ReadGray(img);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (gray[y * width + x] < threshold)
                    {
                        if (x < left) left = x;
                        if (x > right) right = x;
                        if (y < top) top = y;
                        if (y > bottom) bottom = y;
                    }
                }
            }

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
            using var cropped = img.CloneArea(new MagickGeometry(left, top, (uint)cropWidth, (uint)cropHeight));

            // Create a square white canvas (to center drawing)
            int squareSize = Math.Max(cropWidth + (margin * 2), cropHeight + (margin * 2));
            using var squareCanvas = new MagickImage(MagickColors.White, (uint)squareSize, (uint)squareSize);

            int offsetX = (squareSize - cropWidth) / 2;
            int offsetY = (squareSize - cropHeight) / 2;
            squareCanvas.Composite(cropped, offsetX, offsetY, CompositeOperator.Over);

            squareCanvas.FilterType = FilterType.Catrom;
            squareCanvas.Resize(new MagickGeometry((uint)resizeWidth, (uint)resizeHeight) { IgnoreAspectRatio = true });

            squareCanvas.Write($"C:\\NTDLS\\NTDLS.Determinet\\OCR\\debug_out\\line{dbgIndex++}.png");

            // Normalize grayscale to [0..1]
            var resizedGray = ReadGray(squareCanvas);
            return resizedGray.Select(v => v / 255.0).ToArray();
        }

        /// <summary>
        /// Reads the image as one 8-bit luma value per pixel, row-major.
        /// </summary>
        private static byte[] ReadGray(IMagickImage<byte> image)
        {
            using var pixels = image.GetPixelsUnsafe();
            var rgb = pixels.ToByteArray(PixelMapping.RGB)
                ?? throw new InvalidOperationException("Failed to read image pixels.");

            var gray = new byte[rgb.Length / 3];
            for (int i = 0; i < gray.Length; i++)
                gray[i] = (byte)Math.Round(0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]);
            return gray;
        }
    }
}
