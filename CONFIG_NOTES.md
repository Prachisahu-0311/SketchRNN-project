# CONFIG_NOTES.md — What We Know From Previous Training

This document captures the hyperparameter choices that produced good results, and what went wrong with earlier iterations. Use this if results don't look right.

---

## Final config (in train_sketch_v2.py)

These values were validated through 6+ training iterations to converge well:

```python
CONFIG = {
    # DATA
    "max_drawings_per_class": 4000,    # 4 classes × 4K = 16K total samples
    "max_seq_len": 80,                 # RDP simplification keeps sequences short
    "min_seq_len": 8,
    "rdp_epsilon": 2.5,                # Ramer-Douglas-Peucker threshold

    # TRAINING
    "batch_size_per_gpu": 24,          # Reduce to 16 or 8 if OOM
    "epochs": 30,                      # Usually stops around 25 with early stopping
    "learning_rate": 5e-4,
    "weight_decay": 5e-5,
    "patience": 6,
    "min_delta": 0.005,

    # MODEL
    "encoder_hidden_dim": 192,
    "latent_dim": 128,
    "d_model": 192,
    "num_heads": 6,
    "decoder_layers": 4,
    "ff_dim": 384,
    "dropout": 0.12,
    "num_mixtures": 16,

    # VAE — Critical fix
    "beta": 0.15,
    "free_bits": 0.1,                  # Prevents posterior collapse
    "kl_warmup_epochs": 8,

    # LOSS BALANCING — Critical fix
    "coord_loss_weight": 1.0,
    "pen_loss_weight": 5.0,            # Boost pen loss; without this, model ignores rare classes

    # TEACHER FORCING
    "teacher_forcing_start": 1.0,
    "teacher_forcing_end": 0.7,
}
```

Inside `sketch_loss()`, the pen class weights are:
```python
pen_class_weights = torch.tensor([1.0, 2.0, 5.0])  # [down, up, end]
```
Plus `label_smoothing=0.1` in the cross-entropy call.

---

## What good results look like

After ~25-30 epochs, expect roughly these final metrics:

| Metric | Expected value |
|---|---|
| train_loss | 3.6 – 4.2 |
| val_loss | 3.7 – 3.9 |
| pen_acc (overall) | 0.92 – 0.94 |
| pen_acc_down | 0.94 – 0.95 |
| pen_acc_up | 0.25 – 0.35 |
| pen_acc_end | 0.94 – 0.96 |
| AR pen_acc (autoregressive eval) | 0.87 – 0.89 |
| coord_loss | -1.7 to -1.9 |

If your `pen_acc_end` is below 0.5 at epoch 30, something is wrong.

---

## Known pitfalls and fixes

### Pitfall 1: "pen_acc_end stuck at 0% throughout training"
**Cause:** Pen loss weighting too low — coord loss dominates and rare classes get no gradient signal.
**Fix:** Verify `"pen_loss_weight": 5.0` AND `pen_class_weights = [1.0, 2.0, 5.0]` inside `sketch_loss()`.

### Pitfall 2: "Loss is INCREASING epoch over epoch"
**Cause:** Free bits floor + beta too high → KL term dominates total loss.
**Diagnostic:** Print `kl_loss` value during training. If it's stuck at exactly `free_bits × latent_dim` (12.8 here), the floor is dominating.
**Fix:** Lower `free_bits` to 0.05 or `beta` to 0.05.

### Pitfall 3: "Posterior collapse — generation produces noise"
**Cause:** KL term too aggressive early in training.
**Diagnostic:** `kl_loss` drops to near zero rapidly, latent has no information.
**Fix:** Increase `kl_warmup_epochs` to 12, lower `beta` to 0.08.

### Pitfall 4: "Validation loss << Training loss"
This is normal here, NOT a bug. Reasons:
- Training uses teacher forcing decay; validation uses 100% teacher forcing
- Validation uses EMA weights (smoother)
- Don't panic if val_loss is 1.5x lower than train_loss

### Pitfall 5: "Model trains but generates random scribbles"
**Cause 1:** Look at `pen_acc_end` — if 0, the model never learned to terminate. See Pitfall 1.
**Cause 2:** Sampling temperature too high at inference. Generation tuning matters separately from training.

---

## What earlier iterations failed at

For context — these mistakes were made before arriving at the final config:

| Version | What was wrong | What was fixed |
|---|---|---|
| v4 | `beta=0.001` (too small, latent unregularized), no per-class diagnostics | added free bits, per-class pen accuracy |
| v6 | overall pen_acc=97% but pen_acc_end=0% (hidden failure) | added pen_loss_weight=5.0 + class weights |
| v7-early | sample images were random noise on small data | needed full dataset + 30 epochs |
| v8-attempt | `free_bits=0.5` × beta=1.0 was too aggressive | lowered to 0.1 + 0.15 |

The current config in `train_sketch_v2.py` reflects all these lessons.

---

## Post-training generation tuning

After training completes, the per-class optimal sampling temperatures (from sweep on the original v7 model) were:

| Class | Optimal temperature |
|---|---|
| apple | 0.50 |
| circle | 0.30 |
| star | 0.20 |
| triangle | 0.50 |

These might shift slightly for the retrained model. Run `python tune_generation.py` after training to verify or re-tune.

---

## Sequence length post-RDP

After RDP simplification with `epsilon=2.5`, expect:
- Mean: 22-25 tokens
- Median: 20-22 tokens
- P95: 35-45 tokens
- Max: 60-80 tokens

If sequences are much longer (mean > 40), RDP isn't simplifying enough — increase `rdp_epsilon` to 3.0.
If much shorter (mean < 15), too aggressive — lower to 2.0.

---

## Sanity check before full training

Before committing to a 24-hour run, do a sanity test:
1. Open `train_sketch_v2.py`, temporarily change `max_drawings_per_class: 200` and `epochs: 3`
2. Run for 3 epochs (~10 min)
3. Confirm:
   - Loss DECREASES across epochs
   - `pen_acc_end` reaches at least 5% by epoch 3
   - No NaN/Inf in any metric
4. Revert config and run real training

This was the single most important practice that we figured out — don't burn 24 hours on broken config.
