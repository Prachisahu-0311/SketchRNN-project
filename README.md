# AI Sketch Generator

A class-conditional generative model that learns to draw sketches stroke-by-stroke. Built with a BiLSTM-VAE-Transformer hybrid trained on Google QuickDraw.

**[Live demo →](https://sketchrnn-project-jjbuqfrjr2dd5j8hmqpngt.streamlit.app/)**

---

## What it does

Picks a class (apple, circle, star, triangle) and generates a novel sketch by predicting one stroke at a time. The model samples from a learned latent space, so every generation is unique.

The demo lets you:
- Watch the model draw stroke-by-stroke (animated)
- Compare how sampling temperature affects creativity vs accuracy
- Generate galleries showing variety in outputs

---

## Architecture

```
Input sketch (stroke-5 sequence)
        │
        ▼
  ┌─────────────┐
  │ BiLSTM Enc  │ ──▶ μ, log σ²
  └─────────────┘
        │
        ▼ (reparameterize)
       z ~ N(μ, σ²)
        │
        ▼
  ┌─────────────┐    [class_id]
  │ Transformer │◀───────┘
  │  Decoder    │
  └─────────────┘
        │
   ┌────┴────┐
   ▼         ▼
 [MDN]    [Pen state]
  │          │
  ▼          ▼
(dx,dy)   (down/up/end)
```

### Components

- **Encoder**: Bidirectional LSTM (192 hidden, 1 layer) compresses input sketch into a latent vector
- **VAE**: Reparameterization trick + free-bits KL prevents posterior collapse
- **Decoder**: 4-layer causal Transformer (192 dim, 6 heads) generates strokes autoregressively
- **MDN head**: Predicts pen movement (dx, dy) as a mixture of 16 bivariate Gaussians — handles multi-modal next-stroke distributions
- **Pen head**: Classifies pen state (down / up / end-of-sketch)

Total parameters: ~1.7M

---

## Training

| Setting | Value |
|---|---|
| Dataset | Google QuickDraw, `recognized=True` only |
| Classes | apple, circle, star, triangle |
| Samples per class | 4,000 |
| Sequence length cap | 80 (post-RDP simplification) |
| Epochs | 30 |
| Batch size | 24 |
| Optimizer | AdamW, lr=5e-4 |
| Hardware | NVIDIA GTX 1080 Ti |
| Training time | ~24 hours |

### Key engineering decisions

1. **RDP stroke simplification** removes redundant collinear points before training. Reduces sequence length ~3x and lets the model focus on shape, not noise.

2. **Free-bits KL** instead of standard KL divergence. Each latent dimension gets a guaranteed information budget, preventing the encoder from collapsing to the prior.

3. **Weighted pen loss** with class weights `[1.0, 2.0, 5.0]` for `[down, up, end]`. Without this weighting, the model achieves 97% overall pen accuracy but 0% on the rare end-of-sketch class. After weighting, end-of-sketch accuracy reaches 95%.

4. **Per-class temperature tuning at inference**. Different classes have different optimal sampling temperatures — stars want T=0.20 (conservative), apples want T=0.50 (creative). Discovered via systematic sweep, not training-time hyperparameters.

5. **Best-of-N sampling at inference**. Generative models produce variation, including occasional poor samples. Generating multiple candidates and ranking by class-specific geometric scores (closure, roundness, corner count) yields consistent quality without retraining.

---

## Diagnostic Insight: Hidden Class Imbalance

Initially the model showed 97.3% pen-state accuracy, which seemed strong. Adding per-class breakdown revealed:

| Pen state | Accuracy | Frequency |
|---|---|---|
| pen down | ~99% | majority class |
| pen up | 0% | rare |
| end-of-sketch | 0% | once per sketch |

The 97% global figure was almost entirely majority-class accuracy. The model had not learned to terminate sketches at all.

After adjusting loss weighting + free bits + label smoothing:

| Pen state | Accuracy (after fix) |
|---|---|
| pen down | 94.1% |
| pen up | 30.7% |
| end-of-sketch | 94.7% |

Overall accuracy dropped from 97% to 92% — **but the 92% is now real performance across all classes** instead of an inflated number dominated by the easy class.

---

## Files in this repo

```
.
├── app.py                          # Streamlit UI
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── deploy/
    └── model_for_deploy.pt         # Trained model (FP16, ~8 MB)
```

The full training code, dataset preparation, and tuning scripts are in [the training repo](#) (separate repo).

---

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Limitations and future work

- Trained on only 4 classes. Adding more classes requires retraining.
- Generation is stochastic; some samples are noticeably worse than others. Best-of-N filtering mitigates this but doesn't eliminate it.
- Star generation is the weakest class due to its 5-corner geometric complexity.
- Inference is autoregressive and CPU-only on the deployed app; each generation takes 5-15 seconds.

Future directions: more classes, replacing the BiLSTM encoder with a Transformer encoder for consistency, exploring diffusion-based approaches as an alternative to autoregressive generation.

---

## Author

**Prachi Sahu** • M.Tech, Data and Computational Science, IIT Jodhpur

[GitHub](#) • [LinkedIn](#)
=======
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
>>>>>>> 12549f57cbbf847154089047d587131e60ded84f
