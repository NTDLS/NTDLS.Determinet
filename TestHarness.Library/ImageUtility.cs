using NTDLS.Determinet;
using NTDLS.Determinet.Types;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace TestHarness.Library
{
    public class ImageUtility
    {
        public delegate void PreviewImageHandler(Image<Rgba32> img, int randomAngle, Point randomShift, int randomBlur, Point randomScale);

        public static double[]? GetImageGrayscaleBytes(string imagePath, int resizeWidth, int resizeHeight, DniRange<int>? angleVariance,
            DniRange<int>? shiftVariance, DniRange<int>? blurVariance, DniRange<double>? scaleVariance, PreviewImageHandler? previewImageHandler = null)
        {
            var imageBytes = File.ReadAllBytes(imagePath);

            /*
            blurVariance = new DniRange<int>(4, 4);
            previewImageHandler = ((img, randomAngle, randomShift, randomBlur, randomScale) =>
            {
                var name = Path.GetFileNameWithoutExtension(imagePath);
                img.Save($"C:\\NTDLS\\NTDLS.Determinet\\DebugImages\\{name}_A{randomAngle}_SH{randomShift.X},{randomShift.Y}_B{randomBlur}_SC{randomScale.X},{randomScale.Y}.png");
            });
            */

            return GetImageGrayscaleBytes(imageBytes, resizeWidth, resizeHeight, angleVariance, shiftVariance, blurVariance, scaleVariance, previewImageHandler);
        }

        public static double[]? GetImageGrayscaleBytes(byte[] imageBytes, int resizeWidth, int resizeHeight, DniRange<int>? angleVariance,
            DniRange<int>? shiftVariance, DniRange<int>? blurVariance, DniRange<double>? scaleVariance, PreviewImageHandler? previewImageHandler = null)
        {
            // Load the image in RGB format and convert to RGBA.
            using var img = Image.Load<Rgba32>(new MemoryStream(imageBytes));
            return GetImageGrayscaleBytes(img, resizeWidth, resizeHeight, angleVariance, shiftVariance, blurVariance, scaleVariance, previewImageHandler);
        }

        public static double[]? GetImageGrayscaleBytes(Image<Rgba32> img, int resizeWidth, int resizeHeight, DniRange<int>? angleVariance,
            DniRange<int>? shiftVariance, DniRange<int>? blurVariance, DniRange<double>? scaleVariance, PreviewImageHandler? previewImageHandler = null)
        {
            angleVariance ??= new DniRange<int>(0, 0);
            shiftVariance ??= new DniRange<int>(0, 0);
            blurVariance ??= new DniRange<int>(0, 0);
            scaleVariance ??= new DniRange<double>(0, 0);

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

            // Create a square white canvas(to center drawing)
            int squareSize = Math.Max(cropWidth + (margin * 2), cropHeight + (margin * 2));
            using var squareCanvas = new Image<Rgba32>(squareSize, squareSize, Color.White);

            // Apply random scaling
            double scale = DniUtility.NextDouble(scaleVariance.Value.Min, scaleVariance.Value.Max);
            int scaledWidth = (int)(cropWidth * scale);
            int scaledHeight = (int)(cropHeight * scale);

            using var scaled = cropped.Clone(ctx => ctx.Resize(scaledWidth, scaledHeight));

            // Center scaled drawing
            int offsetX = (squareSize - scaledWidth) / 2;
            int offsetY = (squareSize - scaledHeight) / 2;
            squareCanvas.Mutate(ctx => ctx.DrawImage(scaled, new Point(offsetX, offsetY), 1f));

            // Apply rotation
            int randomAngle = DniUtility.Random.Next(angleVariance.Value.Min, angleVariance.Value.Max);
            using var rotated = squareCanvas.Clone(ctx => ctx.Rotate(randomAngle));



            // flatten the transparency onto a white background
            using var flattened = new Image<Rgba32>(rotated.Width, rotated.Height, Color.White);
            int shiftX = DniUtility.Random.Next(shiftVariance.Value.Min, shiftVariance.Value.Max);
            int shiftY = DniUtility.Random.Next(shiftVariance.Value.Min, shiftVariance.Value.Max);
            //Draw rotated image onto white background with a small random shift in position:
            flattened.Mutate(ctx => ctx.DrawImage(rotated, new Point(shiftX, shiftY), 1f));

            int randomBlur = DniUtility.Random.Next(blurVariance.Value.Min, blurVariance.Value.Max);

            if (randomBlur > 0)
            {
                flattened.Mutate(ctx => ctx.GaussianBlur(randomBlur));
            }

            using var resized = flattened.Clone(ctx => ctx.Resize(resizeWidth, resizeHeight));

            previewImageHandler?.Invoke(resized, randomAngle, new Point(shiftX, shiftY), randomBlur, new Point(scaledWidth, scaledHeight));

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
                    using Image image = Image.Load(file);
                    double scale = (double)targetHeight / image.Height;
                    int newWidth = (int)Math.Round(image.Width * scale);

                    image.Mutate(x => x.Resize(newWidth, targetHeight));


                    image.Save(file);
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
