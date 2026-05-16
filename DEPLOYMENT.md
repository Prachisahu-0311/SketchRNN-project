# Deployment Guide — AI Sketch Generator on Streamlit Cloud

## Step 1: Slim down the model file

Streamlit Cloud deploys from GitHub. GitHub blocks files >100 MB and warns at 50 MB. Your `final.pt` contains training-only data (optimizer state, scheduler, history) — strip it.

```bash
# In your project root (where outputs_v2_real/ lives)
python extract_model.py
```

This creates `deploy/model_for_deploy.pt` — should be 5–15 MB.

If output is >50 MB, see "Troubleshooting" below.

---

## Step 2: Create the deployment folder structure

You want a clean repo for deployment. Don't push your training code — that's separate.

```
sketch-generator-app/          ← new folder for the public repo
├── app.py
├── requirements.txt
├── README.md
└── deploy/
    └── model_for_deploy.pt
```

**Why a separate folder:** the public repo should only have files needed to run the demo. Your training scripts, data folder, intermediate outputs, and v6/v7/etc backups stay in your private project folder.

```bash
mkdir sketch-generator-app
cd sketch-generator-app
mkdir deploy
cp ../my_project_m24mac005/deploy/model_for_deploy.pt deploy/
cp ../path/to/app.py .
cp ../path/to/requirements.txt .
```

---

## Step 3: Test locally before pushing

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser. Test all 4 tabs:
1. Generate (try each class)
2. Compare temperatures
3. Gallery
4. About

**If anything breaks locally, fix it before deploying.** Don't debug on the cloud.

---

## Step 4: Push to GitHub

```bash
git init
git add .
git commit -m "Initial deploy of sketch generator demo"
git branch -M main

# Create a new public repo on GitHub.com first (e.g. sketch-generator-demo)
git remote add origin https://github.com/YOUR_USERNAME/sketch-generator-demo.git
git push -u origin main
```

If you see warnings about file sizes, check `deploy/model_for_deploy.pt` is under 100 MB.

---

## Step 5: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Pick your `sketch-generator-demo` repo, branch `main`, file `app.py`
5. Click "Deploy"

First deploy takes 3-5 minutes (installs PyTorch, etc).

Your app will be live at:
`https://YOUR_USERNAME-sketch-generator-demo-app-XXXX.streamlit.app/`

---

## Step 6: After it's live

- Add the live link to your CV / GitHub README / LinkedIn
- Take a screenshot of the deployed app for your portfolio
- Test from a different device/browser to make sure it works for visitors

---

## Troubleshooting

### "Model file too large for GitHub"

If `model_for_deploy.pt` is between 50–100 MB:
- It will work but you'll get a warning during git push
- Use Git LFS for cleaner handling: `git lfs track "*.pt"`

If >100 MB:
- The FP16 conversion in `extract_model.py` should prevent this
- If it still happens, the model architecture is bigger than expected — verify dimensions match

### "App crashes with OOM"

Streamlit Cloud free tier has 1 GB RAM. If your app crashes:
- The `@st.cache_resource` decorator on `load_model()` already helps
- Reduce `n_samples` slider max in Gallery tab from 9 to 6
- Reduce `figsize` in `render_static()` from (3.5, 3.5) to (2.5, 2.5)

### "Generation is slow"

CPU-only generation is 5-15x slower than GPU. Each sketch takes 3-8 seconds. This is normal.
- Make sure `torch.set_num_threads(2)` if needed (uncomment in app.py)
- Consider showing a progress bar or animation while generating

### "GIF rendering fails"

Streamlit Cloud might not have all matplotlib animation backends. If GIFs break:
- Set `show_animation=False` as default
- Falls back to static rendering automatically

---

## Optional Polish

After basic deployment works, consider adding:

1. **Real samples for comparison** — show "Real" QuickDraw samples next to generated ones. Pre-render a few static images from your test set.

2. **Loss curves visualization** — in the About tab, show your training loss curves as a static PNG. Adds technical depth.

3. **Sample of bad outputs** — show what the model generates with WRONG settings (e.g., star at T=0.8). Demonstrates honesty about model limitations.

4. **Custom CSS** — Streamlit allows custom theming via `.streamlit/config.toml`. Pick a clean color palette.

5. **Analytics** — add Google Analytics or Plausible to see if visitors interact with the demo.
