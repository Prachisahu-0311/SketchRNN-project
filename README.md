# AI Sketch Generator

A class-conditional generative model that learns to draw sketches stroke-by-stroke. Built with a BiLSTM-VAE-Transformer hybrid trained on Google QuickDraw.

**[Live demo →](https://YOUR_STREAMLIT_URL_HERE.streamlit.app/)**

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
