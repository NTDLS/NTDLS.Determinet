using NTDLS.Determinet;
using SixLabors.ImageSharp.Processing;
using System.Drawing.Imaging;
using System.Windows.Forms.DataVisualization.Charting;

namespace TestHarnessDraw
{
    public partial class FormMain : Form
    {
        private char[] distinctCharacters = new char[] {
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'a', 'b', 'c', 'd', 'e', 'f','g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                'A', 'B', 'C', 'D', 'E', 'F','G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
            };

        private DniNeuralNetwork? _dni;
        const int _imageWidth = 64;              // Downscale for faster processing with minimal quality loss.
        const int _imageHeight = 64;             // Downscale for faster processing with minimal quality loss.

        public FormMain()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            InitializeChart();

            var timer = new System.Windows.Forms.Timer()
            {
                Interval = 250,
                Enabled = true
            };

            timer.Tick += Timer_Tick;

            var debugModelFile = @"C:\NTDLS\NTDLS.Determinet\TestHarness\bin\Release\net9.0\trained.dni";
            if (File.Exists(debugModelFile))
            {
                LoadModelFromFile(debugModelFile);
                simpleDrawControl.LoadImageFromFile("C:\\NTDLS\\NTDLS.Determinet\\Training Characters\\K 004 Heebo Thin.png");
            }
        }

        private void Timer_Tick(object? sender, EventArgs e)
        {
            var inputBits = GetImageGrayscaleBytes(_imageWidth, _imageHeight);

            if (_dni != null && inputBits != null)
            {
                try
                {
                    var outputs = _dni.Forward(inputBits);
                    var prediction = DniUtility.IndexOfMaxValue(outputs, out var confidence);

                    textBoxDetected.Text = $"{distinctCharacters[prediction]}";
                    textBoxConfidence.Text = $"{confidence:n4}";

                    UpdateChart(outputs, distinctCharacters);
                }
                catch
                {
                }
            }
        }

        private void InitializeChart()
        {
            chartPredictions.Series.Clear();
            chartPredictions.ChartAreas.Clear();

            var chartArea = new ChartArea("OutputArea");
            chartPredictions.ChartAreas.Add(chartArea);

            var series = new Series("Confidence")
            {
                ChartType = SeriesChartType.Column,
                IsValueShownAsLabel = true,
                Color = Color.OrangeRed,
                XValueType = ChartValueType.String,
                IsXValueIndexed = false,
                IsVisibleInLegend = false
            };
            chartPredictions.Series.Add(series);

            chartArea.AxisX.Interval = 1;
            chartArea.AxisX.MajorGrid.Enabled = false;
            chartArea.AxisY.MajorGrid.LineColor = Color.LightGray;
            chartArea.AxisY.Minimum = 0;
            chartArea.AxisY.Maximum = 1;
        }

        private void UpdateChart(double[] outputs, char[] distinctCharacters)
        {
            var series = chartPredictions.Series["Confidence"];
            series.Points.Clear();

            var top = outputs
                .Select((v, i) => new { Index = i, Value = v })
                .OrderByDescending(x => x.Value)
                .Take(5)
                .ToArray();

            // Add each prediction as its own bar
            for (int i = 0; i < top.Length; i++)
            {
                string label = distinctCharacters[top[i].Index].ToString();
                double value = top[i].Value;

                var point = new DataPoint(i + 1, value)
                {
                    AxisLabel = label,
                    Label = value.ToString("0.##"),
                    Color = Color.OrangeRed
                };
                series.Points.Add(point);
            }

            series.Points[0].Color = Color.OrangeRed;
            series.Points[1].Color = Color.Orange;
            series.Points[2].Color = Color.Gold;
            for (int i = 3; i < series.Points.Count; i++)
                series.Points[i].Color = Color.LightGray;

            // Adjust axes for better spacing
            var area = chartPredictions.ChartAreas[0];
            area.AxisX.Interval = 1;
            area.AxisY.Minimum = 0;
            area.AxisY.Maximum = 1;
            area.AxisX.MajorGrid.Enabled = false;
            area.AxisY.MajorGrid.LineColor = Color.LightGray;
            chartPredictions.Invalidate();
        }

        public static Bitmap ToBitmap(SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> image)
        {
            using var ms = new MemoryStream();
            image.Save(ms, new SixLabors.ImageSharp.Formats.Png.PngEncoder());   // lossless
            ms.Position = 0;
            return new Bitmap(ms);
        }

        public static SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> ConvertBitmapToImageSharp(Bitmap bitmap)
        {
            using var memoryStream = new MemoryStream();

            // Save the System.Drawing.Bitmap to a memory stream in a lossless format
            bitmap.Save(memoryStream, ImageFormat.Png);
            memoryStream.Position = 0;

            // Load that stream as an ImageSharp image
            return SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(memoryStream);
        }

        private double[]? GetImageGrayscaleBytes(int resizeWidth, int resizeHeight)
        {
            var bitmap = simpleDrawControl.GetDrawingBitmap();

            // Load the image in RGB format and convert to RGBA.
            using var img = ConvertBitmapToImageSharp(bitmap);

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
                        SixLabors.ImageSharp.PixelFormats.Rgba32 p = row[x];
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
            var bounds = new SixLabors.ImageSharp.Rectangle(left, top, cropWidth, cropHeight);
            using var cropped = img.Clone(ctx => ctx.Crop(bounds));

            // Create a square white canvas (to center drawing)
            int squareSize = Math.Max(cropWidth + (margin * 2), cropHeight + (margin * 2));
            using var squareCanvas = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>(squareSize, squareSize, SixLabors.ImageSharp.Color.White);

            int offsetX = (squareSize - cropWidth) / 2;
            int offsetY = (squareSize - cropHeight) / 2;
            squareCanvas.Mutate(ctx => ctx.DrawImage(cropped, new SixLabors.ImageSharp.Point(offsetX, offsetY), 1f));

            squareCanvas.Mutate(ctx => ctx.GaussianBlur(DniUtility.Random.Next(4, 8)));

            using var resized = squareCanvas.Clone(ctx => ctx.Resize(resizeWidth, resizeHeight));

            pictureBoxAiView.Image = ToBitmap(resized);

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
                        SixLabors.ImageSharp.PixelFormats.Rgba32 p = row[x];
                        double gray = (0.299 * p.R + 0.587 * p.G + 0.114 * p.B) / 255.0;
                        pixels[index++] = gray;
                    }
                }
            });

            return pixels;
        }

        private void ButtonClear_Click(object sender, EventArgs e)
        {
            simpleDrawControl.Clear();
        }

        private void ButtonLoadModel_Click(object sender, EventArgs e)
        {
            using var openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "DNI Files (*.dni)|*.dni|All Files (*.*)|*.*";
            openFileDialog.FilterIndex = 1;
            openFileDialog.Multiselect = false;
            openFileDialog.RestoreDirectory = true;

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                LoadModelFromFile(openFileDialog.FileName);
            }
        }

        private void LoadModelFromFile(string fileName)
        {
            _dni = DniNeuralNetwork.LoadFromFile(fileName)
                ?? throw new Exception("Failed to load the network from file.");

            // Remove "UseBatchNorm" parameter from all layers if it exists
            foreach (var layer in _dni.State.Layers)
            {
                layer.Parameters.Remove("UseBatchNorm");
            }
        }

        private void ButtonLoadImage_Click(object sender, EventArgs e)
        {
            using var openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "PNG Files (*.png)|*.png|All Files (*.*)|*.*";
            openFileDialog.FilterIndex = 1;
            openFileDialog.Multiselect = false;
            openFileDialog.RestoreDirectory = true;

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                simpleDrawControl.LoadImageFromFile(openFileDialog.FileName);
            }
        }
    }
}
