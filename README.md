# Determinet

📦 Be sure to check out the NuGet package: https://www.nuget.org/packages/NTDLS.Determinet

Determinet is a versatile multilayer perceptron neural network for .NET, designed for ease of use and extendibility.
In addition to the library, you'll find a test harness which includes a character recognition trainer, validator and visual testing tool.
These provide working examples of training a model as well as generating predictions.

<img width="819" height="465" alt="image" src="https://github.com/user-attachments/assets/0cc01d7d-84d1-44ff-8a54-c1c74797aec7" />

## Features

- Fully-connected feed-forward networks with any number of hidden layers.
- 18 built-in activation functions (see below), each with a gradient-checked derivative.
- Cross-entropy loss for SoftMax outputs (classification), squared-error loss for everything else (regression).
- SGD or Adam (AdamW) optimization, single-sample or mini-batch training.
- Optional layer normalization, weight decay and global gradient-norm clipping.
- Activation-aware weight initialization (He, LeCun or Glorot).
- Named input/output labels, so you can work with `"temperature"` and `"rain"` instead of array indexes.
- Compact save/load of the whole model, including optimizer state, so training can be resumed exactly.
- Thread-safe inference: `Forward()` never modifies the network.
- Targets .NET 8, 9 and 10.

## Installation

```bash
dotnet add package NTDLS.Determinet
```

## Quick start

A network that learns XOR, a classic problem a single layer can't solve:

```csharp
using NTDLS.Determinet;
using NTDLS.Determinet.Types;
using static NTDLS.Determinet.DniParameters;

// 2 inputs -> 8 hidden (Tanh) -> 2 outputs (SoftMax), with a label for each output.
var configuration = new DniConfiguration { LearningRate = 0.01 };
configuration.AddInputLayer(2);
configuration.AddIntermediateLayer(8, DniActivationType.Tanh);
configuration.AddOutputLayer(2, DniActivationType.SoftMax, ["false", "true"]);

var dni = new DniNeuralNetwork(configuration);
dni.Parameters.Set(Network.UseAdamOptimization, true);

// Each sample is (inputs, expected outputs). For classification the expected output is one-hot.
(double[] inputs, double[] expected)[] samples =
[
    ([0, 0], [1, 0]),
    ([0, 1], [0, 1]),
    ([1, 0], [0, 1]),
    ([1, 1], [1, 0]),
];

for (int epoch = 0; epoch < 500; epoch++)
{
    foreach (var (inputs, expected) in samples)
    {
        double loss = dni.Train(inputs, expected);
    }
}
```

## Making predictions

```csharp
// Raw outputs: for SoftMax these are probabilities that sum to 1.
double[] outputs = dni.Forward([1, 0]);
int index = outputs.IndexOfMaxValue(out double confidence);   // 1, ~0.99

// Or use the output labels.
dni.Forward([1, 0], out var labels);
var best = labels.Max();
Console.WriteLine($"{best.Key} ({best.Value:n2})");         // true (0.99)
```

## Saving and loading

```csharp
dni.SaveToFile("xor.dni");

var loaded = DniNeuralNetwork.LoadFromFile("xor.dni");
```

The file includes the weights, the layer configuration, all parameters and the optimizer state, so a loaded model
can be used for predictions or trained further right where it left off.

## Mini-batch training

`Train()` updates the weights after every sample. `TrainBatch()` averages the gradient over several samples and
applies a single update, which gives smoother, less noisy training:

```csharp
foreach (var batch in samples.Chunk(32))
{
    double averageLoss = dni.TrainBatch(batch);
}
```

## Regression

For predicting real numbers, use an `Identity` output layer. The network is then trained with squared-error loss:

```csharp
var configuration = new DniConfiguration { LearningRate = 0.01 };
configuration.AddInputLayer(1);
configuration.AddIntermediateLayer(16, DniActivationType.Tanh);
configuration.AddOutputLayer(1, DniActivationType.Identity);

var dni = new DniNeuralNetwork(configuration);
dni.Parameters.Set(Network.UseAdamOptimization, true);

for (int epoch = 0; epoch < 2000; epoch++)
{
    for (double x = -3; x <= 3; x += 0.25)
    {
        dni.Train([x], [Math.Sin(x)]);
    }
}

double y = dni.Forward([1.0])[0];   // approximately sin(1)
```

## Named inputs

Give the input layer labels and you can supply values by name. Missing names are treated as 0.

```csharp
var configuration = new DniConfiguration();
configuration.AddInputLayer(2, ["temperature", "humidity"]);
configuration.AddIntermediateLayer(8, DniActivationType.ReLU);
configuration.AddOutputLayer(2, DniActivationType.SoftMax, ["dry", "rain"]);

var dni = new DniNeuralNetwork(configuration);

var input = new DniNamedLabelValues();
input.Set("temperature", 0.7);
input.Set("humidity", 0.9);

dni.Forward(input, out var prediction);
Console.WriteLine(prediction.Max().Key);
```

## Parameters

Network and layer behavior is controlled through named parameters.

```csharp
// Network-wide settings.
dni.Parameters.Set(Network.LearningRate, 0.001);
dni.Parameters.Set(Network.UseAdamOptimization, true);
dni.Parameters.Set(Network.WeightDecay, 0.01);
dni.Parameters.Set(Network.GradientClip, 5.0);

// Per-layer settings are passed when the layer is added.
var hiddenParams = new DniNamedParameterCollection();
hiddenParams.Set(LeakyReLU.Alpha, 0.05);
hiddenParams.Set(Layer.UseLayerNorm, true);
configuration.AddIntermediateLayer(32, DniActivationType.LeakyReLU, hiddenParams);

// You can also store your own values in the model; they are saved with it.
dni.Parameters.Set("Epochs", 12);
int epochs = dni.Parameters.Get<int>("Epochs");
```

| Parameter                     | Default | Description                                                                                                  |
| ----------------------------- | ------- | ------------------------------------------------------------------------------------------------------------ |
| `Network.LearningRate`        | 0.005   | Step size of each update. Adam typically wants a smaller value (e.g. 0.0001–0.001) than SGD.                  |
| `Network.UseAdamOptimization` | false   | Use Adam (with decoupled weight decay) instead of plain SGD.                                                  |
| `Network.WeightDecay`         | 0.0001  | Shrinks weights toward zero to reduce overfitting. With Adam, values around 0.01 are typical.                 |
| `Network.GradientClip`        | 5.0     | Maximum overall size of a single update's gradient. 0 disables clipping.                                      |
| `Network.ComputedLoss`        | —       | Set by `Train()`/`TrainBatch()` to the most recent loss.                                                      |
| `Layer.UseLayerNorm`          | false   | Normalizes a hidden layer's values before its activation function. Can help deeper networks train.            |
| `SoftMax.Temperature`         | 1.0     | Values above 1 soften the output probabilities, below 1 sharpen them.                                         |
| `LeakyReLU.Alpha`             | 0.01    | Slope for negative inputs.                                                                                    |
| `ELU.Alpha`                   | 1.0     | Scale of the negative region.                                                                                 |
| `SELU.Alpha`, `SELU.Lambda`   | paper   | Self-normalizing constants from the SELU paper.                                                               |
| `Linear.Alpha`, `Linear.Range`   | 1, [-1, 1] | Slope, and the range the output is clamped to.                                                        |
| `Piecewise.Alpha`, `Piecewise.Range` | 1, [-1, 1] | Slope used outside the range; inside the range the slope is 1.                                 |

## Built-in Activation Functions

| Activation Type                   | What It Does                                                                                      | When To Use It                                                                                                                                                          | Where To Use It                                                             |
| --------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Identity**                      | Passes values through unchanged. The neuron is basically “transparent.”                           | Use for **pure regression** or as a **linear output layer** when predicting real numbers — e.g., predicting house prices, speed, or temperature.                        | Output                                                                      |
| **PiecewiseLinear**               | Slope 1 inside a range and a different slope (Alpha) outside it, with no jumps at the edges.      | Use when you want **bounded but mostly linear** responses — e.g., controlling actuator strength, image intensity normalization, or testing stable transitions.          | Hidden (sometimes Output for bounded regression)                            |
| **Linear**                        | Scales input by a slope and optionally clamps to a range.                                         | Great for **controlled continuous outputs** where scaling matters — e.g., steering angle, torque output, or color channel prediction.                                   | Output (regression), Hidden (feature scaling)                               |
| **ReLU (Rectified Linear Unit)**  | Keeps positive values, zeros out negatives. Very fast and stable.                                 | The **default** for most deep learning tasks. Ideal for **vision models, embeddings, and dense MLPs** — e.g., image recognition, feature extraction, signal processing. | Hidden                                                                      |
| **Sigmoid**                       | Squashes output between 0 and 1 (probability curve).                                              | Use for **binary classification** (“yes/no,” “on/off,” “spam/not spam”) or when modeling **probability per neuron** in multi-label tasks.                               | Output (binary classification), Hidden (gating in LSTMs or attention)       |
| **Tanh**                          | Squashes output between -1 and 1, centered around 0.                                              | Use when values can be **positive or negative** and symmetry matters — e.g., **RNNs**, control systems, or **directional outputs** (-1 left, +1 right).                 | Hidden                                                                      |
| **LeakyReLU**                     | Like ReLU, but negative values leak through slightly instead of being zeroed.                     | Use in **deep MLPs** to avoid “dead neurons” and keep gradients flowing — especially in **dense** or **deep** architectures.                                            | Hidden                                                                      |
| **SoftMax**                       | Converts outputs into a probability distribution that sums to 1.                                  | Use for **multi-class classification** — e.g., recognizing characters, digits, objects, or anything where exactly one class is correct.                                 | Output only                                                                 |
| **SimpleSoftMax**                 | A simplified SoftMax (no temperature scaling).                                                    | Use when you need **fast, stable classification** and temperature tuning isn’t required. Works the same for most classification tasks.                                  | Output only                                                                 |
| **ELU (Exponential Linear Unit)** | Works like ReLU for positive inputs but smoothly bends for negatives instead of dropping to zero. | Use when you want **ReLU-like behavior** but prefer smoother gradients and **mean activations closer to zero** — e.g., deep networks prone to “dead neurons.”           | Hidden                                                                      |
| **Gaussian (RBF)**                | Outputs a bell curve centered around zero (f(x) = e<sup>-x²</sup>).                               | Use for **radial basis networks**, **feature clustering**, or **anomaly detection** where proximity to a center value matters.                                          | Hidden                                                                      |
| **HardSigmoid**                   | A linear, fast approximation of the sigmoid function.                                             | Use in **lightweight models** or embedded systems where performance matters, or when full sigmoid precision isn’t required.                                             | Hidden, Output (binary)                                                     |
| **HardTanh**                      | Like Tanh but clipped to the range [-1, 1] using straight lines.                                  | Use when you need **bounded outputs** but want faster computation — e.g., control tasks, quantized models, or when smoothness isn’t critical.                           | Hidden                                                                      |
| **Mish**                          | Smoothly blends ReLU and Swish behavior (f(x) = x·tanh(ln(1+eˣ))).                                | Use when you want **smoother gradients** and slightly better **generalization** in deep MLPs or CNNs — can outperform ReLU/Swish on some tasks.                         | Hidden                                                                      |
| **SELU (Scaled ELU)**             | A self-normalizing version of ELU that keeps activations around mean=0, var=1.                    | Use in **deep fully-connected networks** without normalization layers; helps stabilize training automatically.                                                          | Hidden                                                                      |
| **SoftSign**                      | Smoothly scales inputs as x / (1 + \|x\|).                                                        | Use as a **lightweight, stable alternative** to tanh when you want smooth gradients without exponential cost — good for RNNs or bounded outputs.                        | Hidden                                                                      |
| **Swish**                         | Smooth, self-gated function (f(x) = x · sigmoid(x)).                                              | Use in **modern deep networks** (CNNs, Transformers, MLPs) where you want ReLU-like performance but with **better gradient flow** and smoother transitions.             | Hidden                                                                      |
| **SoftPlus**                      | Smooth version of ReLU (f(x) = ln(1 + eˣ)); never zeroes out completely.                          | Use when you want **ReLU-like behavior** but need continuous derivatives — e.g., **probabilistic networks**, VAEs, or any model requiring gradient stability.           | Hidden, sometimes Output (positive-only regression)                         |

**Loss is chosen from the output layer:** SoftMax and SimpleSoftMax outputs are trained with cross-entropy loss; every
other output activation is trained with squared-error loss. SoftMax is only allowed on the output layer.

## Test harness

The repository includes a complete handwritten character recognition example (digits plus upper and lower case letters):

- **TestHarness.Train** trains a model from a folder of labeled images, with augmentation (rotation, shift and blur),
  a learning-rate scheduler, early stopping and checkpointing.
- **TestHarness.Validate** measures a trained model's accuracy against a separate validation set.
- **TestHarness.Draw** lets you draw characters with the mouse and see the model's predictions live (pictured above).
