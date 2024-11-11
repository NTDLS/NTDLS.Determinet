using NTDLS.Determinet;
using System.Windows.Forms;

namespace TestHarnessDraw
{
    public partial class FormMain : Form
    {
        private DniNeuralNetwork? _dni;

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

            var debugModelFile = @"C:\NTDLS\NTDLS.Determinet\TestHarness\bin\Release\net8.0\trained.json";
            if (File.Exists(debugModelFile))
            {
                _dni = DniNeuralNetwork.LoadFromFile(debugModelFile);
            }
        }

        private void Timer_Tick(object? sender, EventArgs e)
        {
            if (_dni != null)
            {
                try
                {
                    var inputBits = GetImageGrayscaleBytes();
                    var detectedValue = DniUtility.GetIndexOfMaxValue(_dni.Forward(inputBits), out var confidence);
                    textBoxDetected.Text = detectedValue.ToString();
                    textBoxConfidence.Text = $"{confidence:n4}";
                }
                catch
                {
                }
            }
        }

        private double[] GetImageGrayscaleBytes(int resizeWidth = 28, int resizeHeight = 28)
        {
            var bitmap = simpleDrawControl.GetDrawingBitmap();

            // Convert the bitmap to grayscale
            int left = bitmap.Width, right = 0, top = bitmap.Height, bottom = 0;

            // Loop through pixels to find non-white areas for cropping
            for (int y = 0; y < bitmap.Height; y++)
            {
                for (int x = 0; x < bitmap.Width; x++)
                {
                    Color pixel = bitmap.GetPixel(x, y);
                    if (pixel.R < 255 || pixel.G < 255 || pixel.B < 255) // Adjust for your background color
                    {
                        if (x < left) left = x;
                        if (x > right) right = x;
                        if (y < top) top = y;
                        if (y > bottom) bottom = y;
                    }
                }
            }

            // Define bounding box for cropping
            int width = right - left + 1;
            int height = bottom - top + 1;
            var bounds = new Rectangle(left, top, width, height);

            // Crop to the bounding box
            Bitmap croppedBitmap = bitmap.Clone(bounds, bitmap.PixelFormat);

            // Resize the cropped image
            Bitmap resizedBitmap = new Bitmap(croppedBitmap, new Size(resizeWidth, resizeHeight));

            pictureBoxAiView.Image = resizedBitmap.Clone() as Bitmap;

            double[] pixelData = new double[resizeWidth * resizeHeight];
            int index = 0;

            // Process each pixel to get grayscale values normalized to 0-1
            for (int y = 0; y < resizedBitmap.Height; y++)
            {
                for (int x = 0; x < resizedBitmap.Width; x++)
                {
                    Color pixel = resizedBitmap.GetPixel(x, y);
                    byte grayValue = (byte)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
                    pixelData[index++] = grayValue / 255.0; // Scale to 0-1
                }
            }

            // Clean up resources
            croppedBitmap.Dispose();
            resizedBitmap.Dispose();

            return pixelData;
        }

        private void ButtonClear_Click(object sender, EventArgs e)
        {
            simpleDrawControl.Clear();
        }

        private void ButtonLoadModel_Click(object sender, EventArgs e)
        {
            using var openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "JSON Files (*.json)|*.json|All Files (*.*)|*.*";
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
