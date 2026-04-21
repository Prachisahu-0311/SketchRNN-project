# SketchRNN — Generating Sketches Autoregressively with Deep Learning

This project builds and trains an autoregressive neural network that can generate 2D sketches line-by-line, replicating how humans sketch objects. We utilize the Google QuickDraw Dataset as the training corpus, focusing on a set of predefined classes: **apple, circle, star, triangle**, and more. The model can generalize these sketches and create novel, smooth, and recognizable drawings.

---

## 🚀 Project Overview

- **Model Type**: This is a sequence-level generative model based on the SketchRNN architecture.
- **Backbone Components**:
  - LSTM Encoder-Decoder with VAE (Variational Autoencoder).
  - Fully generative model with MDN (Mixture Density Network) outputs for coordinate prediction.
  - Transformer layers for richer autoregressive context in the decoder.
- **Data Source**: Based on Google's QuickDraw Dataset (available from https://quickdraw.withgoogle.com/data).

This improved implementation adds **distributed data parallelism** (DDP) for multi-GPU setups, **EMA (Exponential Moving Average)** model weight averaging for stability, and **optimized generation logic** for sampling diverse outputs.

---

## 🧱 Features and Enhancements

### **1. Data Preprocessing:**

- Tokenized Google’s QuickDraw `.ndjson` data and normalized relative coordinates into absolute `(x, y)` values.
- Balanced dataset across chosen classes, ensuring uniformity for the training process.
- Sequences were augmented to a fixed maximum length using `stroke-5` vectorization, encoding `(delta_x, delta_y, pen_down, pen_up, end_of_stroke)`.

---

### **2. Model Architecture:**

Our architecture is a hybrid of **RNN**, **VAE**, and **Transformer layers**:

1. **Encoder (Bidirectional LSTM):**
   - Encodes input strokes into a latent space vector (`mu` and `logvar` for the latent z-space).
   - Class embeddings condition the generation process by appending object class context to the latent space.

2. **Variational Autoencoder (VAE):**
   - Imposes a regularization constraint on the latent space to encourage smooth interpolation between different latent variables, thus avoiding collapsing modes.
   - Uses KL divergence (`β-KL`) loss to ensure smooth learning.

3. **Decoder:**
   - Combines Transformer-based self-attention layers for expressive autoregressive decoding.
   - Employs **Mixture Density Networks (MDN)** to predict distributions over the next stroke step.
   - Outputs pen-down, pen-up, and end-of-sequence probabilities for realistic multi-stroke generation.

---

### **3. Training Methodology:**

- **Distributed Training**: Utilized **PyTorch DistributedDataParallel (DDP)** for efficient GPU utilization.
- **Losses Used**:
  - **Reconstruction Loss**: Penalizes mismatches in predicted and actual coordinates for each stroke.
  - **KL Loss**: Optimization of the VAE latent space.
  - **Classification Loss**: For modeling pen transitions `(pen-down, pen-up, end-of-drawing)`.
- **Optimization Techniques**:
  - `AdamW` optimizer with `ReduceLROnPlateau` learning rate scheduler.
  - Gradient clipping and automatic mixed precision (AMP) for faster and stable training steps.

---

### **4. Key Training Improvements**

- **Gradient Fighting**: Multi-head self-attention avoids overfitting by expanding expressiveness without data leakage.
- **EMA Model Weight Averaging**: Exponentially averages weights over iterations for more robust and smoother inference.
- **Early Stopping**: Stops training upon stagnation in validation loss for optimal training efficiency.

---

### **5. Sketch Generation:**

1. **Class-conditional Generation:** Generates sketches for predefined classes with tunable parameters like:
   - `temperature` for diversity.
   - `top_k_mixtures` to consider the most likely stroke steps.
   - `eos_bias` to control end-point smoothness.
2. **Multiple Samples:** Generates multiple candidates for a class, scoring and selecting the most plausible sketch to display.
3. **GIF Animations:** Saves stroke-by-stroke drawing progress as a GIF file.

---

## 📊 Performance Metrics

### **Training Results**
- Recovered **fine strokes** with no mode collapse.
- Maintained accuracy of predicting pen states (`pen-down/up`) across sketch sequences.

### **Final Outputs**

| Object Class  | Example Output                       |
|---------------|-------------------------------------|
| Apple         | ![apple-sketch](outputs/apple.gif) |
| Circle        | ![circle-sketch](outputs/circle.png) |
| Star          | ![star-sketch](outputs/star.gif)   |
| Triangle      | ![triangle-sketch](outputs/triangle.gif) |

---

## 🛠 Configuration

All hyperparameters are configurable via the `CONFIG` dictionary in the primary script. See below for the key setups:

```
CONFIG = {
    "epochs": 40,
    "classes": ["apple", "circle", "star", "triangle"],
    "learning_rate": 0.0005,
    "batch_size_per_gpu": 4,
    ...
}
```

---

## 📂 Project Structure

```plaintext
SketchRNN Project/
├── data/                   - Contains preprocessed QuickDraw data
├── outputs/                - Directory for generated sketches (gifs, saved models, grid visualizations)
├── scripts/                - Python training and preprocessing scripts
├── utils/                  - Auxiliary files (e.g., collate functions, data loaders)
└── README.md               - This documentation file
```

---

## 🧑‍💻 How to Run

### Step 1: Clone the repository

```bash
git clone https://github.com/Prachisahu-0311/sketchRNN.git
cd sketchRNN
```

### Step 2: Install prerequisites

Install all required Python dependencies with pip:

```bash
pip install -r requirements.txt
```

### Step 3: Download the QuickDraw dataset

```bash
wget https://storage.googleapis.com/quickdraw_dataset/full/raw/*.ndjson -P data/
```

### Step 4: Train the model

Run the training script locally or using distributed GPUs:

```bash
python sketch_rnn.py
```

For multi-GPU setups:

```bash
WORLD_SIZE=4 python -m torch.distributed.launch --nproc_per_node 4 sketch_rnn.py
```

### Step 5: Generate and visualize outputs

After training, visualizations, grid plots, and GIF animations are saved in the `outputs/` directory.

---

## 📜 License

This project is licensed under the **MIT License**. You are free to modify, distribute, and use it for commercial and personal projects.

---

## 🤖 Future Work

- Add support for additional object categories.
- Experiment with fully Transformer-based architectures.
- Fine-tune MDN hyperparameters for sharper but diverse sketch patterns.
