using NTDLS.Determinet;
using NTDLS.Determinet.Types;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace TestHarness.Library
{
    public class ImageUtility
    {
        public delegate void PreviewImageHandler(Image<Rgba32> img);

        public static double[]? GetImageGrayscaleBytes(string imagePath, int resizeWidth, int resizeHeight,
            DniRange<int>? angleVariance, DniRange<int>? shiftVariance, DniRange<int>? blurVariance, PreviewImageHandler? previewImageHandler = null)
        {
            var imageBytes = File.ReadAllBytes(imagePath);
            return GetImageGrayscaleBytes(imageBytes, resizeWidth, resizeHeight, angleVariance, shiftVariance, blurVariance, previewImageHandler);
        }

        public static double[]? GetImageGrayscaleBytes(byte[] imageBytes, int resizeWidth, int resizeHeight,
            DniRange<int>? angleVariance, DniRange<int>? shiftVariance, DniRange<int>? blurVariance, PreviewImageHandler? previewImageHandler = null)
        {
            // Load the image in RGB format and convert to RGBA.
            using var img = Image.Load<Rgba32>(new MemoryStream(imageBytes));
            return GetImageGrayscaleBytes(img, resizeWidth, resizeHeight, angleVariance, shiftVariance, blurVariance, previewImageHandler);
        }

        public static double[]? GetImageGrayscaleBytes(Image<Rgba32> img, int resizeWidth, int resizeHeight,
            DniRange<int>? angleVariance, DniRange<int>? shiftVariance, DniRange<int>? blurVariance, PreviewImageHandler? previewImageHandler = null)
        {
            angleVariance ??= new DniRange<int>(0, 0);
            shiftVariance ??= new DniRange<int>(0, 0);
            blurVariance ??= new DniRange<int>(0, 0);

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

            int angle = DniUtility.Random.Next(angleVariance.Value.Min, angleVariance.Value.Max);
            // Small random rotation (for consistency with training)
            using var rotated = squareCanvas.Clone(ctx => ctx.Rotate(angle));

            // flatten the transparency onto a white background
            using var flattened = new Image<Rgba32>(rotated.Width, rotated.Height, Color.White);
            int shiftX = DniUtility.Random.Next(shiftVariance.Value.Min, shiftVariance.Value.Max);
            int shiftY = DniUtility.Random.Next(shiftVariance.Value.Min, shiftVariance.Value.Max);
            //Draw rotated image onto white background with a small random shift in position:
            flattened.Mutate(ctx => ctx.DrawImage(rotated, new Point(shiftX, shiftY), 1f));

            flattened.Mutate(ctx => ctx.GaussianBlur(DniUtility.Random.Next(blurVariance.Value.Min, blurVariance.Value.Max)));

            using var resized = flattened.Clone(ctx => ctx.Resize(resizeWidth, resizeHeight));

            previewImageHandler?.Invoke(resized);

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

            //resized.Save($"C:\\NTDLS\\NTDLS.Determinet\\debug\\{Path.GetFileNameWithoutExtension(imagePath)}.png");

            return pixels;
        }
    }
}
