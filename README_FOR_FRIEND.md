# Recovery Instructions — Sketch Generation Model Training

Hi! Thanks for helping run this training. Original work was lost when the IIT Jodhpur server was disabled — we need to retrain the model to recover the project.

This file tells you exactly what to do. Should take ~24 hours of GPU time and ~30 minutes of your active attention spread over those 24 hours.

---

## What you'll need

- Linux machine with NVIDIA GPU (8GB+ VRAM ideal, 4GB minimum)
- Python 3.10 or 3.11
- ~5 GB free disk space
- Internet access (for downloading dataset)
- ~24 hours of uptime (training is long but unattended)

This was originally trained on a **GTX 1080 Ti**. Newer GPUs (RTX 30/40 series) will finish in 6-12 hours instead of 24.

---

## Step-by-step

### Step 1: Set up environment (5 min)

```bash
# Create a fresh project folder
mkdir sketch-project
cd sketch-project

# Copy all the files from the recovery folder into this directory:
# - train_sketch_v2.py
# - download_quickdraw.py
# - extract_model.py
# - generate_only.py
# - tune_generation.py
# - app.py
# - requirements.txt
# - README_FOR_FRIEND.md (this file)

# Create a Python virtual environment
python3 -m venv venv
source venv/bin/activate    # On Linux/Mac
# OR for Windows: venv\Scripts\activate

# Install dependencies for training
pip install -r requirements_training.txt
```

### Step 2: VERIFY GPU is visible (1 min)

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

You should see something like:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

If `CUDA available: False`, **stop here**. Fix CUDA before continuing — training on CPU will take weeks.

### Step 3: Download the dataset (5 min)

```bash
python download_quickdraw.py
```

This downloads 4 NDJSON files (~170 MB total) from Google's public bucket to `./data/`.

Expected output at end:
```
Download summary:
  apple      OK  (57.0 MB)
  circle     OK  (39.0 MB)
  star       OK  (43.0 MB)
  triangle   OK  (31.0 MB)
```

### Step 4: Initialize git and push to GitHub FIRST (10 min)

**Critical step — don't skip.** The original work was lost because we never pushed to GitHub. Don't repeat that.

```bash
git init
git add train_sketch_v2.py download_quickdraw.py extract_model.py generate_only.py tune_generation.py app.py requirements.txt README_FOR_FRIEND.md
git commit -m "Initial recovery setup"

# Create a NEW private repo at github.com (suggest: sketch-generator-training)
# Then:
git remote add origin https://github.com/PRACHI_USERNAME/sketch-generator-training.git
git branch -M main
git push -u origin main
```

(Replace PRACHI_USERNAME with Prachi's GitHub username. **Use a PRIVATE repo for training code/data, separate from the public deployment repo we'll make later.**)

Add a `.gitignore` to avoid committing junk:
```bash
cat > .gitignore << 'EOF'
data/
outputs_*/
__pycache__/
*.pyc
venv/
.env
EOF

git add .gitignore
git commit -m "Add gitignore"
git push
```

### Step 5: Run the training (24 hours on 1080 Ti, less on newer GPUs)

```bash
python train_sketch_v2.py
```

The script will:
1. Load the dataset, apply RDP simplification
2. Print model architecture (~4M params)
3. Train for up to 30 epochs (early stopping likely around epoch 20-25)
4. Save checkpoints to `outputs_v2_real/` every epoch
5. Generate sample images every 3 epochs in `outputs_v2_real/training_samples/`
6. Save final results to `outputs_v2_real/final.pt` + `summary.json` + `results_grid.png`

**You don't need to babysit it.** Just check on it every few hours to make sure it hasn't crashed.

#### Checkpoints in progress
The script saves `latest.pt` every epoch. If training crashes or you need to stop it (Ctrl+C), running `python train_sketch_v2.py` again will resume from the last checkpoint automatically.

#### Expected progress signals
- Epoch 1: train_loss around 5-9 (high initially, this is normal)
- Epoch 5: train_loss around 3-4, val_loss decreasing
- Epoch 10: pen_acc_end starting to rise above 50%
- Epoch 20-30: convergence, pen_acc_end reaching 90%+

If after epoch 5 the loss is still going UP (not down), something is wrong — message Prachi.

### Step 6: When training finishes (5 min)

You should see:
```
Final test metrics: {...}
AR Test: coord_nll=... | pen_acc=...
Saved grid: outputs_v2_real/results_grid.png
```

Verify these files exist:
```bash
ls -la outputs_v2_real/
```

You should see:
- `final.pt` (30-80 MB — the trained model)
- `best.pt` (same)
- `latest.pt` (same)
- `summary.json`
- `history.json`
- `results_grid.png`
- `training_samples/` folder with progress images
- `loss_curves.png`

### Step 7: Send the trained model back to Prachi

The full `final.pt` is too large to email. Run the extraction script to slim it:

```bash
python extract_model.py
```

This creates `deploy/model_for_deploy.pt` — about 5-15 MB. **This is the file to send.**

**Also send:**
- `outputs_v2_real/results_grid.png` (final sample grid — shows what the model generates)
- `outputs_v2_real/summary.json` (final metrics)
- `outputs_v2_real/history.json` (training curves data)
- `outputs_v2_real/loss_curves.png` (visual training curves)
- Push the latest state of the repo to GitHub (don't include the `outputs_v2_real/` folder, it's in .gitignore)

You can ZIP these few small files and send via WhatsApp, Telegram, email, Google Drive, etc.

Optionally also send `outputs_v2_real/training_samples/` folder — shows generation quality progression over epochs. Nice to have but not critical.

---

## Troubleshooting

### "CUDA out of memory"
Open `train_sketch_v2.py`, find the CONFIG dict, lower `batch_size_per_gpu` from 24 to 16, then 8 if needed. Save and re-run.

### "ModuleNotFoundError: No module named X"
```bash
pip install <X>
```
Common missing modules: `tqdm`, `matplotlib`, `pillow`. All should be in requirements.txt.

### Training seems to be hung (no progress for 30+ min)
- Check `nvidia-smi` in another terminal — is the GPU actually being used (high utilization, memory used)?
- If GPU utilization is 0%, the script is stuck somewhere. Kill it (Ctrl+C) and rerun — it will resume.

### Loss going to NaN
Should not happen with the current config, but if it does:
- Stop training (Ctrl+C)
- Open `train_sketch_v2.py`, find `"learning_rate": 5e-4`, change to `3e-4`
- Delete `outputs_v2_real/latest.pt` (to restart from scratch)
- Rerun

### "Data file not found" error
The download script failed. Run `python download_quickdraw.py` again. If it keeps failing, manually download from:
https://storage.googleapis.com/quickdraw_dataset/full/simplified/apple.ndjson
(replace apple with circle, star, triangle)
Put in `./data/`

---

## What this is being used for

This is a class-conditional generative model for sketch generation. Architecture: BiLSTM encoder + VAE + Transformer decoder with Mixture Density Network outputs. Trained on Google QuickDraw subset (4 classes: apple, circle, star, triangle).

Prachi will use the trained model for a Streamlit demo deployed publicly, as a portfolio project.

---

## Thank you

Truly appreciate the GPU access and time. If something doesn't make sense in these instructions, message Prachi immediately — better to pause than train with broken config.
