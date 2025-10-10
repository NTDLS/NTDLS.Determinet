using NTDLS.Determinet;

namespace TestHarnessDraw
{
    public partial class FormMain : Form
    {
        private DniNeuralNetwork? _dni;
        const int _imageWidth = 64;              // Downscale for faster processing with minimal quality loss.
        const int _imageHeight = 64;             // Downscale for faster processing with minimal quality loss.

        public FormMain()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            var timer = new System.Windows.Forms.Timer()
            {
                Interval = 250,
                Enabled = true
            };

            timer.Tick += Timer_Tick;

            var debugModelFile = @"C:\NTDLS\NTDLS.Determinet\TestHarness\bin\Release\net9.0\trained.dni";
            if (File.Exists(debugModelFile))
            {
                _dni = DniNeuralNetwork.LoadFromFile(debugModelFile);
            }
        }

        private void Timer_Tick(object? sender, EventArgs e)
        {
            var inputBits = GetImageGrayscaleBytes(_imageWidth, _imageHeight);

            if (_dni != null && inputBits != null)
            {
                try
                {
                    var detectedValue = DniUtility.GetIndexOfMaxValue(_dni.Forward(inputBits), out var confidence);
                    textBoxDetected.Text = detectedValue.ToString();
                    textBoxConfidence.Text = $"{confidence:n4}";
                }
                catch
                {
                }
            }
        }

        private double[]? GetImageGrayscaleBytes(int resizeWidth, int resizeHeight)
        {
            var bitmap = simpleDrawControl.GetDrawingBitmap();

            int left = bitmap.Width, right = 0, top = bitmap.Height, bottom = 0;
            int threshold = 250;

            // Detect non-white pixels
            for (int y = 0; y < bitmap.Height; y++)
            {
                for (int x = 0; x < bitmap.Width; x++)
                {
                    Color pixel = bitmap.GetPixel(x, y);
                    if (pixel.R < threshold || pixel.G < threshold || pixel.B < threshold)
                    {
                        if (x < left) left = x;
                        if (x > right) right = x;
                        if (y < top) top = y;
                        if (y > bottom) bottom = y;
                    }
                }
            }

            // Blank drawing detection
            if (right <= left || bottom <= top)
            {
                pictureBoxAiView.Image = null;
                return null;
            }

            // Add margin and clamp
            int margin = 5;
            int cropLeft = Math.Max(0, left - margin);
            int cropTop = Math.Max(0, top - margin);
            int cropRight = Math.Min(bitmap.Width - 1, right + margin);
            int cropBottom = Math.Min(bitmap.Height - 1, bottom + margin);
            int cropWidth = cropRight - cropLeft + 1;
            int cropHeight = cropBottom - cropTop + 1;

            // Crop region of interest
            Rectangle bounds = new Rectangle(cropLeft, cropTop, cropWidth, cropHeight);
            using var cropped = bitmap.Clone(bounds, bitmap.PixelFormat);

            // Center in square canvas
            int squareSize = Math.Max(cropWidth, cropHeight);
            using var squareCanvas = new Bitmap(squareSize, squareSize);

            using (Graphics g = Graphics.FromImage(squareCanvas))
            {
                g.Clear(Color.White);
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

                int offsetX = (squareSize - cropWidth) / 2;
                int offsetY = (squareSize - cropHeight) / 2;

                g.DrawImage(cropped, offsetX, offsetY, cropWidth, cropHeight);
            }

            // Resize to target network input
            using var resized = new Bitmap(squareCanvas, new Size(resizeWidth, resizeHeight));

            // Visualize what the network sees
            pictureBoxAiView.Image = (Bitmap)resized.Clone();

            // Convert to grayscale normalized [0..1], 0 = black, 1 = white
            double[] pixelData = new double[resizeWidth * resizeHeight];
            int index = 0;

            for (int y = 0; y < resizeHeight; y++)
            {
                for (int x = 0; x < resizeWidth; x++)
                {
                    Color p = resized.GetPixel(x, y);
                    double gray = (0.299 * p.R + 0.587 * p.G + 0.114 * p.B) / 255.0;
                    pixelData[index++] = gray;
                }
            }

            return pixelData;
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
                _dni = DniNeuralNetwork.LoadFromFile(openFileDialog.FileName)
                    ?? throw new Exception("Failed to load the network from file.");
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
                simpleDrawControl.LoadFromFile(openFileDialog.FileName);
            }
        }
    }
}
