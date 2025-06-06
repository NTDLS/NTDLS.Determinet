﻿namespace TestHarnessDraw
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
            simpleDrawControl = new SimpleDrawControl();
            buttonClear = new Button();
            buttonLoadModel = new Button();
            textBoxDetected = new TextBox();
            labelDetected = new Label();
            pictureBoxAiView = new PictureBox();
            buttonLoadImage = new Button();
            label1 = new Label();
            labelConfidence = new Label();
            textBoxConfidence = new TextBox();
            labelSanitized = new Label();
            ((System.ComponentModel.ISupportInitialize)pictureBoxAiView).BeginInit();
            SuspendLayout();
            // 
            // simpleDrawControl
            // 
            simpleDrawControl.BackColor = Color.White;
            simpleDrawControl.Location = new Point(24, 94);
            simpleDrawControl.Name = "simpleDrawControl";
            simpleDrawControl.Size = new Size(200, 200);
            simpleDrawControl.TabIndex = 0;
            // 
            // buttonClear
            // 
            buttonClear.Location = new Point(122, 300);
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
            textBoxDetected.Location = new Point(230, 112);
            textBoxDetected.Name = "textBoxDetected";
            textBoxDetected.ReadOnly = true;
            textBoxDetected.Size = new Size(92, 23);
            textBoxDetected.TabIndex = 3;
            // 
            // labelDetected
            // 
            labelDetected.AutoSize = true;
            labelDetected.Location = new Point(230, 94);
            labelDetected.Name = "labelDetected";
            labelDetected.Size = new Size(82, 15);
            labelDetected.TabIndex = 4;
            labelDetected.Text = "Detected Digit";
            // 
            // pictureBoxAiView
            // 
            pictureBoxAiView.BackgroundImageLayout = ImageLayout.None;
            pictureBoxAiView.Location = new Point(230, 216);
            pictureBoxAiView.Name = "pictureBoxAiView";
            pictureBoxAiView.Size = new Size(28, 28);
            pictureBoxAiView.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBoxAiView.TabIndex = 5;
            pictureBoxAiView.TabStop = false;
            // 
            // buttonLoadImage
            // 
            buttonLoadImage.Location = new Point(24, 300);
            buttonLoadImage.Name = "buttonLoadImage";
            buttonLoadImage.Size = new Size(92, 23);
            buttonLoadImage.TabIndex = 6;
            buttonLoadImage.Text = "Load Image";
            buttonLoadImage.UseVisualStyleBackColor = true;
            buttonLoadImage.Click += ButtonLoadImage_Click;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(128, 16);
            label1.Name = "label1";
            label1.Size = new Size(273, 15);
            label1.TabIndex = 7;
            label1.Text = "<- Load a trained model and then draw in the box.";
            // 
            // labelConfidence
            // 
            labelConfidence.AutoSize = true;
            labelConfidence.Location = new Point(230, 146);
            labelConfidence.Name = "labelConfidence";
            labelConfidence.Size = new Size(68, 15);
            labelConfidence.TabIndex = 9;
            labelConfidence.Text = "Confidence";
            // 
            // textBoxConfidence
            // 
            textBoxConfidence.Location = new Point(230, 164);
            textBoxConfidence.Name = "textBoxConfidence";
            textBoxConfidence.ReadOnly = true;
            textBoxConfidence.Size = new Size(102, 23);
            textBoxConfidence.TabIndex = 8;
            // 
            // labelSanitized
            // 
            labelSanitized.AutoSize = true;
            labelSanitized.Location = new Point(230, 198);
            labelSanitized.Name = "labelSanitized";
            labelSanitized.Size = new Size(54, 15);
            labelSanitized.TabIndex = 10;
            labelSanitized.Text = "Sanitized";
            // 
            // FormMain
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(451, 356);
            Controls.Add(labelSanitized);
            Controls.Add(labelConfidence);
            Controls.Add(textBoxConfidence);
            Controls.Add(label1);
            Controls.Add(buttonLoadImage);
            Controls.Add(pictureBoxAiView);
            Controls.Add(labelDetected);
            Controls.Add(textBoxDetected);
            Controls.Add(buttonLoadModel);
            Controls.Add(buttonClear);
            Controls.Add(simpleDrawControl);
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;
            Name = "FormMain";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "TestHarnessDraw";
            Load += Form1_Load;
            ((System.ComponentModel.ISupportInitialize)pictureBoxAiView).EndInit();
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
        private Label label1;
        private Label labelConfidence;
        private TextBox textBoxConfidence;
        private Label labelSanitized;
    }
}
