using NTDLS.Determinet;
using NTDLS.Determinet.Types;
using System.Drawing.Imaging;
using System.Windows.Forms.DataVisualization.Charting;
using TestHarness.Library;
using static NTDLS.Determinet.DniParameters;

namespace TestHarness.Draw
{
    public partial class FormMain : Form
    {
        private DniNeuralNetwork? _dni;

        public int BrushSize => trackBarBrushSize.Value;

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

            simpleDrawControl.SetParent(this);

            var debugModelFile = @"..\..\..\..\Trained Models\CharacterRecognition_Best.dni";
            if (File.Exists(debugModelFile))
            {
                LoadModelFromFile(debugModelFile);
                //simpleDrawControl.LoadImageFromFile("C:\\NTDLS\\NTDLS.Determinet\\Sample Images\Training\\K 004 Heebo Thin.png");
            }
        }

        private void Timer_Tick(object? sender, EventArgs e)
        {
            var bitmap = simpleDrawControl.GetDrawingBitmap();

            var image = ConvertBitmapToImageSharp(bitmap);

            var inputBits = ImageUtility.GetImageGrayscaleBytes(image, Constants.ImageWidth, Constants.ImageHeight, DniRange<int>.Zero, DniRange<int>.Zero, new DniRange<float>(0.1f, 1), DniRange<double>.One,
                (img, randomAngle, randomShift, randomBlur, randomScale) =>
                {
                    var previewBmp = ToBitmap(img);
                    pictureBoxAiView.Image = previewBmp;
                });

            if (_dni != null && inputBits != null)
            {
                try
                {
                    textBoxLearningRate.Text = $"{_dni.Parameters.Get<double>(Network.LearningRate):n10}";
                    textBoxLoss.Text = $"{_dni.Parameters.Get<double>("BatchLoss"):n10}";
                    textBoxEpoch.Text = $"{_dni.Parameters.Get<double>("Epochs"):n0}";

                    var outputs = _dni.Forward(inputBits, out var labelValues);

                    var prediction = labelValues.Max();

                    textBoxDetected.Text = $"{prediction.Key}";
                    textBoxConfidence.Text = $"{prediction.Value:n4}";

                    UpdateChart(labelValues);
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

        private void UpdateChart(DniNamedLabelValues labelValues)
        {
            var series = chartPredictions.Series["Confidence"];
            series.Points.Clear();

            var top = labelValues.Values
                .OrderByDescending(x => x.Value)
                .Take(5)
                .ToArray();

            // Add each prediction as its own bar
            for (int i = 0; i < top.Length; i++)
            {
                string label = top[i].Key;
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
