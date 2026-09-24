using ImageMagick;
using NTDLS.Determinet;
using NTDLS.Determinet.Types;
using System.Drawing;

namespace TestHarness.Library
{
    public class ImageUtility
    {
        public delegate void PreviewImageHandler(IMagickImage<byte> img, int randomAngle, Point randomShift, float randomBlur, Point randomScale);

        public static double[]? GetImageGrayscaleBytes(string imagePath, int resizeWidth, int resizeHeight, DniRange<int>? angleVariance,
            DniRange<int>? shiftVariance, DniRange<float>? blurVariance, DniRange<double>? scaleVariance, PreviewImageHandler? previewImageHandler = null)
        {
            var imageBytes = File.ReadAllBytes(imagePath);

            /*
            blurVariance = new DniRange<float>(0.5f, 1.5f);
            previewImageHandler = ((img, randomAngle, randomShift, randomBlur, randomScale) =>
            {
                var name = Path.GetFileNameWithoutExtension(imagePath);
                img.Write($"C:\\NTDLS\\NTDLS.Determinet\\DebugImages\\{name}_A{randomAngle}_SH{randomShift.X},{randomShift.Y}_B{randomBlur}_SC{randomScale.X},{randomScale.Y}.png");
            });
            */

            return GetImageGrayscaleBytes(imageBytes, resizeWidth, resizeHeight, angleVariance, shiftVariance, blurVariance, scaleVariance, previewImageHandler);
        }

        public static double[]? GetImageGrayscaleBytes(byte[] imageBytes, int resizeWidth, int resizeHeight, DniRange<int>? angleVariance,
            DniRange<int>? shiftVariance, DniRange<float>? blurVariance, DniRange<double>? scaleVariance, PreviewImageHandler? previewImageHandler = null)
        {
            using var img = new MagickImage(imageBytes);
            return GetImageGrayscaleBytes(img, resizeWidth, resizeHeight, angleVariance, shiftVariance, blurVariance, scaleVariance, previewImageHandler);
        }

        /// <summary>
        /// Crops the ink, centers it on a square white canvas, applies the requested random augmentation, resizes, and
        /// returns the grayscale pixels normalized to [0..1]. Returns null for a blank image. <paramref name="source"/> is not modified.
        /// </summary>
        public static double[]? GetImageGrayscaleBytes(IMagickImage<byte> source, int resizeWidth, int resizeHeight, DniRange<int>? angleVariance,
            DniRange<int>? shiftVariance, DniRange<float>? blurVariance, DniRange<double>? scaleVariance, PreviewImageHandler? previewImageHandler = null)
        {
            angleVariance ??= new DniRange<int>(0, 0);
            shiftVariance ??= new DniRange<int>(0, 0);
            blurVariance ??= new DniRange<float>(0, 0);
            scaleVariance ??= DniRange<double>.One;

            // Work on an opaque RGB copy: transparent areas become white rather than being read as (black) ink.
            using var img = source.Clone();
            img.BackgroundColor = MagickColors.White;
            img.Alpha(AlphaOption.Remove);

            int width = (int)img.Width;
            int height = (int)img.Height;

            // Detect bounds of non-white pixels
            int threshold = 250;
            int left = width, right = 0, top = height, bottom = 0;

            var rgb = ReadRgb(img);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int i = (y * width + x) * 3;
                    if (rgb[i] < threshold || rgb[i + 1] < threshold || rgb[i + 2] < threshold)
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
            using var cropped = img.Clone();
            cropped.Crop(new MagickGeometry(left, top, (uint)cropWidth, (uint)cropHeight));
            cropped.ResetPage();

            // Create a square white canvas(to center drawing)
            int squareSize = Math.Max(cropWidth + (margin * 2), cropHeight + (margin * 2));
            using var squareCanvas = new MagickImage(MagickColors.White, (uint)squareSize, (uint)squareSize);

            // Apply random scaling
            double scale = DniUtility.NextDouble(scaleVariance.Value.Min, scaleVariance.Value.Max);
            int scaledWidth = Math.Max(1, (int)(cropWidth * scale));
            int scaledHeight = Math.Max(1, (int)(cropHeight * scale));

            using var scaled = cropped.Clone();
            Resize(scaled, scaledWidth, scaledHeight);

            // Center scaled drawing
            int offsetX = (squareSize - scaledWidth) / 2;
            int offsetY = (squareSize - scaledHeight) / 2;
            squareCanvas.Composite(scaled, offsetX, offsetY, CompositeOperator.Over);

            // Apply rotation (the canvas grows to fit, and the exposed corners are filled with white)
            int randomAngle = DniUtility.Random.Next(angleVariance.Value.Min, angleVariance.Value.Max);
            using var rotated = squareCanvas.Clone();
            if (randomAngle != 0)
            {
                rotated.BackgroundColor = MagickColors.White;
                rotated.Rotate(randomAngle);
                rotated.ResetPage();
            }

            int shiftX = DniUtility.Random.Next(shiftVariance.Value.Min, shiftVariance.Value.Max);
            int shiftY = DniUtility.Random.Next(shiftVariance.Value.Min, shiftVariance.Value.Max);
            //Draw rotated image onto white background with a small random shift in position:
            using var flattened = new MagickImage(MagickColors.White, rotated.Width, rotated.Height);
            flattened.Composite(rotated, shiftX, shiftY, CompositeOperator.Over);

            float randomBlur = DniUtility.NextFloat(blurVariance.Value.Min, blurVariance.Value.Max);

            if (randomBlur > 0)
            {
                flattened.GaussianBlur(0, randomBlur);
            }

            using var resized = flattened.Clone();
            Resize(resized, resizeWidth, resizeHeight);

            previewImageHandler?.Invoke(resized, randomAngle, new Point(shiftX, shiftY), randomBlur, new Point(scaledWidth, scaledHeight));

            // Convert to grayscale and normalize [0..1]
            var pixels = new double[resizeWidth * resizeHeight];
            var resizedRgb = ReadRgb(resized);

            for (int index = 0; index < pixels.Length; index++)
            {
                int i = index * 3;
                pixels[index] = (0.299 * resizedRgb[i] + 0.587 * resizedRgb[i + 1] + 0.114 * resizedRgb[i + 2]) / 255.0;
            }

            return pixels;
        }

        /// <summary>
        /// Resizes to exactly width x height (ignoring aspect ratio) with a Catmull-Rom (bicubic) filter.
        /// </summary>
        private static void Resize(IMagickImage<byte> image, int width, int height)
        {
            image.FilterType = FilterType.Catrom;
            image.Resize(new MagickGeometry((uint)width, (uint)height) { IgnoreAspectRatio = true });
        }

        /// <summary>
        /// Reads the image as packed 8-bit R, G, B triplets, row-major.
        /// </summary>
        private static byte[] ReadRgb(IMagickImage<byte> image)
        {
            using var pixels = image.GetPixelsUnsafe();
            return pixels.ToByteArray(PixelMapping.RGB)
                ?? throw new InvalidOperationException("Failed to read image pixels.");
        }

        public static void ResizeAllImagesRecursive(string sourceFolder)
        {
            int targetHeight = 50;

            Console.WriteLine($"Resizing all images under '{sourceFolder}' to height {targetHeight}px...");

            foreach (string file in Directory.EnumerateFiles(sourceFolder, "*.*", SearchOption.AllDirectories))
            {
                string ext = Path.GetExtension(file).ToLowerInvariant();
                if (ext is not (".jpg" or ".jpeg" or ".png" or ".bmp" or ".gif" or ".webp"))
                    continue;

                try
                {
                    using var image = new MagickImage(file);
                    double scale = (double)targetHeight / image.Height;
                    int newWidth = (int)Math.Round(image.Width * scale);

                    Resize(image, newWidth, targetHeight);

                    image.Write(file);
                    Console.WriteLine($" {file} -> {newWidth}x{targetHeight}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($" Error processing {file}: {ex.Message}");
                }
            }

            Console.WriteLine("Done!");
        }

        public static void MoveTenPercent(string sourceDirectory, string destinationDirectory)
        {
            double validationFraction = 0.10; // 10%
            string[] imageExtensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"];

            var rng = new Random(12345); // use fixed seed for reproducibility (optional)

            Console.WriteLine($"Scanning directories under:\n  {sourceDirectory}\n");

            foreach (var dir in Directory.GetDirectories(sourceDirectory, "*", SearchOption.AllDirectories))
            {
                var files = Directory.GetFiles(dir)
                    .Where(f => imageExtensions.Contains(Path.GetExtension(f).ToLower()))
                    .ToList();

                if (files.Count == 0)
                    continue;

                int countToMove = (int)Math.Round(files.Count * validationFraction);
                if (countToMove == 0)
                    continue;

                // randomly select subset
                var toMove = files.OrderBy(_ => rng.Next()).Take(countToMove).ToList();

                // Determine relative path and matching validation subfolder
                string relativePath = Path.GetRelativePath(sourceDirectory, dir);
                string validationDir = Path.Combine(destinationDirectory, relativePath);
                Directory.CreateDirectory(validationDir);

                Console.WriteLine($"{relativePath,-40}  moving {countToMove,4} / {files.Count,4} images");

                foreach (var src in toMove)
                {
                    string dest = Path.Combine(validationDir, Path.GetFileName(src));

                    // Ensure unique filename if collisions happen
                    int counter = 1;
                    while (File.Exists(dest))
                    {
                        string name = Path.GetFileNameWithoutExtension(src);
                        string ext = Path.GetExtension(src);
                        dest = Path.Combine(validationDir, $"{name}_{counter++}{ext}");
                    }

                    File.Move(src, dest);
                }
            }
        }
    }
}
