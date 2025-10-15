namespace TestHarness.Draw
{
    partial class FormMain
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea1 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Series series1 = new System.Windows.Forms.DataVisualization.Charting.Series();
            simpleDrawControl = new SimpleDrawControl();
            buttonClear = new Button();
            buttonLoadModel = new Button();
            textBoxDetected = new TextBox();
            labelDetected = new Label();
            pictureBoxAiView = new PictureBox();
            buttonLoadImage = new Button();
            labelConfidence = new Label();
            textBoxConfidence = new TextBox();
            labelSanitized = new Label();
            chartPredictions = new System.Windows.Forms.DataVisualization.Charting.Chart();
            trackBarBrushSize = new TrackBar();
            labelBrushSize = new Label();
            labelLearningRate = new Label();
            textBoxLearningRate = new TextBox();
            labelLoss = new Label();
            textBoxLoss = new TextBox();
            labelEpoch = new Label();
            textBoxEpoch = new TextBox();
            ((System.ComponentModel.ISupportInitialize)pictureBoxAiView).BeginInit();
            ((System.ComponentModel.ISupportInitialize)chartPredictions).BeginInit();
            ((System.ComponentModel.ISupportInitialize)trackBarBrushSize).BeginInit();
            SuspendLayout();
            // 
            // simpleDrawControl
            // 
            simpleDrawControl.BackColor = Color.White;
            simpleDrawControl.Location = new Point(12, 41);
            simpleDrawControl.Name = "simpleDrawControl";
            simpleDrawControl.Size = new Size(400, 382);
            simpleDrawControl.TabIndex = 0;
            // 
            // buttonClear
            // 
            buttonClear.Location = new Point(226, 12);
            buttonClear.Name = "buttonClear";
            buttonClear.Size = new Size(75, 23);
            buttonClear.TabIndex = 1;
            buttonClear.Text = "Clear";
            buttonClear.UseVisualStyleBackColor = true;
            buttonClear.Click += ButtonClear_Click;
            // 
            // buttonLoadModel
            // 
            buttonLoadModel.Location = new Point(12, 12);
            buttonLoadModel.Name = "buttonLoadModel";
            buttonLoadModel.Size = new Size(110, 23);
            buttonLoadModel.TabIndex = 2;
            buttonLoadModel.Text = "Load Model";
            buttonLoadModel.UseVisualStyleBackColor = true;
            buttonLoadModel.Click += ButtonLoadModel_Click;
            // 
            // textBoxDetected
            // 
            textBoxDetected.Location = new Point(465, 191);
            textBoxDetected.Name = "textBoxDetected";
            textBoxDetected.ReadOnly = true;
            textBoxDetected.Size = new Size(102, 23);
            textBoxDetected.TabIndex = 3;
            // 
            // labelDetected
            // 
            labelDetected.AutoSize = true;
            labelDetected.Location = new Point(465, 173);
            labelDetected.Name = "labelDetected";
            labelDetected.Size = new Size(54, 15);
            labelDetected.TabIndex = 4;
            labelDetected.Text = "Detected";
            // 
            // pictureBoxAiView
            // 
            pictureBoxAiView.BackgroundImageLayout = ImageLayout.None;
            pictureBoxAiView.Location = new Point(465, 296);
            pictureBoxAiView.Name = "pictureBoxAiView";
            pictureBoxAiView.Size = new Size(128, 128);
            pictureBoxAiView.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBoxAiView.TabIndex = 5;
            pictureBoxAiView.TabStop = false;
            // 
            // buttonLoadImage
            // 
            buttonLoadImage.Location = new Point(128, 12);
            buttonLoadImage.Name = "buttonLoadImage";
            buttonLoadImage.Size = new Size(92, 23);
            buttonLoadImage.TabIndex = 6;
            buttonLoadImage.Text = "Load Image";
            buttonLoadImage.UseVisualStyleBackColor = true;
            buttonLoadImage.Click += ButtonLoadImage_Click;
            // 
            // labelConfidence
            // 
            labelConfidence.AutoSize = true;
            labelConfidence.Location = new Point(465, 217);
            labelConfidence.Name = "labelConfidence";
            labelConfidence.Size = new Size(68, 15);
            labelConfidence.TabIndex = 9;
            labelConfidence.Text = "Confidence";
            // 
            // textBoxConfidence
            // 
            textBoxConfidence.Location = new Point(465, 235);
            textBoxConfidence.Name = "textBoxConfidence";
            textBoxConfidence.ReadOnly = true;
            textBoxConfidence.Size = new Size(102, 23);
            textBoxConfidence.TabIndex = 8;
            // 
            // labelSanitized
            // 
            labelSanitized.AutoSize = true;
            labelSanitized.Location = new Point(465, 278);
            labelSanitized.Name = "labelSanitized";
            labelSanitized.Size = new Size(54, 15);
            labelSanitized.TabIndex = 10;
            labelSanitized.Text = "Sanitized";
            // 
            // chartPredictions
            // 
            chartArea1.Name = "ChartArea1";
            chartPredictions.ChartAreas.Add(chartArea1);
            chartPredictions.Location = new Point(611, 12);
            chartPredictions.Name = "chartPredictions";
            series1.ChartArea = "ChartArea1";
            series1.Name = "Series1";
            chartPredictions.Series.Add(series1);
            chartPredictions.Size = new Size(197, 411);
            chartPredictions.TabIndex = 11;
            chartPredictions.Text = "chartPredictions";
            // 
            // trackBarBrushSize
            // 
            trackBarBrushSize.Location = new Point(418, 41);
            trackBarBrushSize.Maximum = 50;
            trackBarBrushSize.Minimum = 1;
            trackBarBrushSize.Name = "trackBarBrushSize";
            trackBarBrushSize.Orientation = Orientation.Vertical;
            trackBarBrushSize.Size = new Size(45, 382);
            trackBarBrushSize.TabIndex = 12;
            trackBarBrushSize.Value = 15;
            // 
            // labelBrushSize
            // 
            labelBrushSize.AutoSize = true;
            labelBrushSize.Location = new Point(414, 23);
            labelBrushSize.Name = "labelBrushSize";
            labelBrushSize.Size = new Size(37, 15);
            labelBrushSize.TabIndex = 13;
            labelBrushSize.Text = "Brush";
            // 
            // labelLearningRate
            // 
            labelLearningRate.AutoSize = true;
            labelLearningRate.Location = new Point(465, 41);
            labelLearningRate.Name = "labelLearningRate";
            labelLearningRate.Size = new Size(79, 15);
            labelLearningRate.TabIndex = 15;
            labelLearningRate.Text = "Learning Rate";
            // 
            // textBoxLearningRate
            // 
            textBoxLearningRate.Location = new Point(465, 59);
            textBoxLearningRate.Name = "textBoxLearningRate";
            textBoxLearningRate.ReadOnly = true;
            textBoxLearningRate.Size = new Size(102, 23);
            textBoxLearningRate.TabIndex = 14;
            // 
            // labelLoss
            // 
            labelLoss.AutoSize = true;
            labelLoss.Location = new Point(465, 129);
            labelLoss.Name = "labelLoss";
            labelLoss.Size = new Size(30, 15);
            labelLoss.TabIndex = 17;
            labelLoss.Text = "Loss";
            // 
            // textBoxLoss
            // 
            textBoxLoss.Location = new Point(465, 147);
            textBoxLoss.Name = "textBoxLoss";
            textBoxLoss.ReadOnly = true;
            textBoxLoss.Size = new Size(102, 23);
            textBoxLoss.TabIndex = 16;
            // 
            // labelEpoch
            // 
            labelEpoch.AutoSize = true;
            labelEpoch.Location = new Point(465, 85);
            labelEpoch.Name = "labelEpoch";
            labelEpoch.Size = new Size(40, 15);
            labelEpoch.TabIndex = 19;
            labelEpoch.Text = "Epoch";
            // 
            // textBoxEpoch
            // 
            textBoxEpoch.Location = new Point(465, 103);
            textBoxEpoch.Name = "textBoxEpoch";
            textBoxEpoch.ReadOnly = true;
            textBoxEpoch.Size = new Size(102, 23);
            textBoxEpoch.TabIndex = 18;
            // 
            // FormMain
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(820, 436);
            Controls.Add(labelEpoch);
            Controls.Add(textBoxEpoch);
            Controls.Add(labelLoss);
            Controls.Add(textBoxLoss);
            Controls.Add(labelLearningRate);
            Controls.Add(textBoxLearningRate);
            Controls.Add(simpleDrawControl);
            Controls.Add(labelBrushSize);
            Controls.Add(trackBarBrushSize);
            Controls.Add(chartPredictions);
            Controls.Add(labelSanitized);
            Controls.Add(labelConfidence);
            Controls.Add(textBoxConfidence);
            Controls.Add(buttonLoadImage);
            Controls.Add(pictureBoxAiView);
            Controls.Add(labelDetected);
            Controls.Add(textBoxDetected);
            Controls.Add(buttonLoadModel);
            Controls.Add(buttonClear);
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;
            Name = "FormMain";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "TestHarnessDraw";
            Load += Form1_Load;
            ((System.ComponentModel.ISupportInitialize)pictureBoxAiView).EndInit();
            ((System.ComponentModel.ISupportInitialize)chartPredictions).EndInit();
            ((System.ComponentModel.ISupportInitialize)trackBarBrushSize).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private SimpleDrawControl simpleDrawControl;
        private Button buttonClear;
        private Button buttonLoadModel;
        private TextBox textBoxDetected;
        private Label labelDetected;
        private PictureBox pictureBoxAiView;
        private Button buttonLoadImage;
        private Label labelConfidence;
        private TextBox textBoxConfidence;
        private Label labelSanitized;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartPredictions;
        private TrackBar trackBarBrushSize;
        private Label labelBrushSize;
        private Label labelLearningRate;
        private TextBox textBoxLearningRate;
        private Label labelLoss;
        private TextBox textBoxLoss;
        private Label labelEpoch;
        private TextBox textBoxEpoch;
    }
}
