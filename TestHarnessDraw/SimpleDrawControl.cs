using System.Drawing.Drawing2D;

namespace TestHarnessDraw
{
    public partial class SimpleDrawControl : UserControl
    {
        private readonly Bitmap drawingBitmap;
        private bool isDrawing;
        private MouseButtons mouseButton;
        private Point lastPoint;

        public SimpleDrawControl()
        {
            InitializeComponent();
            DoubleBuffered = true;

            Width = 200;
            Height = 200;
            BackColor = Color.White;

            drawingBitmap = new Bitmap(Width, Height);
            MouseUp += (s, e) => { isDrawing = false; };

            MouseDown += (s, e) =>
            {
                mouseButton = e.Button;
                isDrawing = true;
                lastPoint = e.Location;
            };

            MouseMove += (s, e) =>
            {
                if (isDrawing)
                {
                    using (Graphics g = Graphics.FromImage(drawingBitmap))
                    {
                        Color color = Color.White;

                        if (mouseButton == MouseButtons.Left)
                        {
                            color = Color.Black;
                        }

                        using Pen thickPen = new Pen(color, 12)
                        {
                            StartCap = LineCap.Round,
                            EndCap = LineCap.Round,
                            LineJoin = LineJoin.Round
                        };
                        g.DrawLine(thickPen, lastPoint, e.Location);

                        // Draw a circle at the current point to fill gaps
                        using var brush = new SolidBrush(color);
                        g.FillEllipse(brush, e.X - 2, e.Y - 2, 4, 4);
                    }

                    // Update lastPoint to the current location for the next line segment
                    lastPoint = e.Location;
                    Invalidate(); // Causes the control to redraw
                }
            };
        }

        public void Clear()
        {
            using (Graphics g = Graphics.FromImage(drawingBitmap))
            {
                g.Clear(Color.White);
            }

            Invalidate();
        }

        public void LoadFromFile(string filePath, int x = 0, int y = 0)
        {
            using (Graphics g = Graphics.FromImage(drawingBitmap))
            {
                g.Clear(Color.White);
            }

            using (var loadedImage = new Bitmap(filePath))
            {
                using Graphics g = Graphics.FromImage(drawingBitmap);
                g.DrawImage(loadedImage, x, y, loadedImage.Width, loadedImage.Height);
            }

            Invalidate();
        }

        public Bitmap GetDrawingBitmap()
        {
            return drawingBitmap;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            e.Graphics.DrawImage(drawingBitmap, Point.Empty);
        }

        private void SimpleDrawControl_Load(object sender, EventArgs e)
        {

        }
    }
}
