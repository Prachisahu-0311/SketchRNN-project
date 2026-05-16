# Sketch Generator Project — Complete Recovery Package

This package contains the **full project** plus the **multi-object composition extension**.

---

## What's in this package

```
sketch-generator-final/
├── README_MASTER.md              ← you are here
├── README_FOR_FRIEND.md          ← instructions for the friend running training
├── CONFIG_NOTES.md               ← hyperparameter playbook / troubleshooting
├── DEPLOYMENT.md                 ← Streamlit Cloud deployment instructions
├── README.md                     ← the GitHub README (for the deployed repo)
│
├── train_sketch_v2.py            ← TRAINING SCRIPT (friend runs this)
├── download_quickdraw.py         ← downloads the dataset
├── requirements_training.txt     ← Python deps for training
│
├── tune_generation.py            ← post-training temperature tuning
├── generate_only.py              ← generate samples without retraining
├── extract_model.py              ← slims final.pt for deployment
│
├── app.py                        ← STREAMLIT APP (now with 5 tabs)
├── multi_object_composition.py   ← NEW: extension #2 — multi-object scenes
├── requirements.txt              ← deps for the deployed app
```

---

## The Big Picture: 3 Phases

You can't do everything at once. Follow this order.

### Phase 1: Friend retrains the model
You send the package to your friend. They follow `README_FOR_FRIEND.md`.
Output: a `final.pt` file (and `model_for_deploy.pt` after running `extract_model.py`).

### Phase 2: You receive the trained model and develop locally
On your laptop, place the trained `model_for_deploy.pt` in `deploy/` folder.
You can now run the Streamlit app, test the multi-object composition tab, etc.

### Phase 3: Deploy publicly
Push to GitHub. Connect Streamlit Cloud. Project goes live.

---

## How to Open This Project in VS Code

### Step 1: Unzip the package somewhere sensible

Put it in a folder like `~/projects/sketch-generator/` (Linux/Mac) or `C:\Users\YOU\projects\sketch-generator\` (Windows).

### Step 2: Open the folder in VS Code

Three ways to do this:
1. **GUI**: Open VS Code → `File → Open Folder...` → navigate to the project folder → click Open
2. **Command line**: In a terminal, `cd` into the folder, then run `code .` (the dot means "current folder")
3. **Drag-and-drop**: Drag the project folder onto the VS Code icon/window

### Step 3: Install VS Code extensions you need

When VS Code opens the project, it may suggest extensions. Accept these or install manually:
- **Python** (by Microsoft) — required for syntax/linting
- **Pylance** (by Microsoft) — usually installs with Python
- **Jupyter** (optional, only if you want to add notebooks)
- **GitLens** (optional, helpful for tracking what changed)

### Step 4: Open a terminal in VS Code

`Terminal → New Terminal` (or `Ctrl+`` ` backtick).

The terminal will already be in your project folder. From here, you'll run all commands.

### Step 5: Set up Python environment

In the VS Code terminal:

```bash
# Create virtual environment (one-time)
python3 -m venv venv

# Activate it
# Linux/Mac:
source venv/bin/activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (cmd):
.\venv\Scripts\activate.bat

# You should see "(venv)" at the start of your prompt
# Install dependencies (use training requirements for full setup)
pip install -r requirements_training.txt
pip install streamlit
```

### Step 6: Tell VS Code which Python to use

`Ctrl+Shift+P` → type "Python: Select Interpreter" → pick the one inside your `venv/` folder.

This makes VS Code use your project's environment, not the system Python.

---

## What to Run, In Order

### If you're starting fresh (no trained model yet):

```bash
# 1. Download the dataset (your friend or you, ~5 min)
python download_quickdraw.py

# 2. Train the model (your friend's GPU, ~12-24 hours)
python train_sketch_v2.py
# → produces outputs_v2_real/final.pt

# 3. (After training) slim the model for deployment
python extract_model.py
# → produces deploy/model_for_deploy.pt

# 4. (Optional) tune generation parameters
python tune_generation.py

# 5. Run the Streamlit app locally to test
streamlit run app.py
```

### If you've already received the trained model from your friend:

```bash
# 1. Place the friend's model file
mkdir -p deploy
# Copy their model_for_deploy.pt into deploy/

# 2. Install Streamlit
pip install streamlit

# 3. Run the app
streamlit run app.py
```

The app opens in your browser at http://localhost:8501.

---

## The Streamlit App's 5 Tabs

| Tab | What it does |
|---|---|
| 🎨 Generate | Single-object generation with best-of-6 selection. Watch it draw stroke-by-stroke. |
| 🌡️ Compare Temperatures | Shows how sampling temperature affects output for one class. |
| 🖼️ Gallery | Generates multiple samples of a class to show variety. |
| 🎬 Compose Scene | **NEW (extension)**: pick multiple objects + layout, get a multi-object scene. |
| ℹ️ About | Architecture, training details, engineering notes. |

The Compose Scene tab is the **multi-object composition extension** mentioned in the assignment.

---

## How the Extension Works (Multi-object Composition)

Code lives in `multi_object_composition.py`. The pipeline:

```
User input: ["apple", "apple", "star", "triangle"]   (4 objects)
        │
        ▼
For each object: run model with best-of-4 → stroke-5 sequence
        │
        ▼
For each sequence: compute bounding box (xmin, ymin, xmax, ymax)
        │
        ▼
Layout strategy picks (cx, cy) for each object:
  - Horizontal row: evenly spaced left-to-right
  - Grid: roughly square arrangement
  - Random (non-overlapping): with minimum distance check
  - Circular: arranged on a circle around the center
        │
        ▼
Render: each sketch is normalized to its bbox center,
        scaled to a uniform size, and translated to its target position.
        All objects render onto a single matplotlib figure.
```

What I deliberately did NOT implement:
- **Spatial relationship semantics** (e.g., "apple ON TOP of triangle"). This would require training the model on paired multi-object data, which we don't have.
- **Object interactions / occlusion**. Same reason.
- **Background or scene context**. Out of scope.

The implementation is honest about its scope — see the "How does this work?" expander in the Compose Scene tab.

---

## Important: Save Your Work to GitHub IMMEDIATELY

You lost everything last time because you never pushed to GitHub. **Don't repeat that mistake.**

Right now, before doing anything else:

```bash
git init
git add .
git status   # check what's being added — should NOT include data/ or outputs_*/

# Create .gitignore first if not already there
cat > .gitignore << 'EOF'
data/
outputs_*/
deploy/*.pt
__pycache__/
*.pyc
venv/
.venv/
.env
*.gif
EOF

git add .gitignore
git commit -m "Initial project recovery setup"

# Create a new private repo at https://github.com/new
# Then connect:
git remote add origin https://github.com/YOUR_USERNAME/sketch-generator.git
git branch -M main
git push -u origin main
```

After this, every time you make a meaningful change:
```bash
git add .
git commit -m "describe what you changed"
git push
```

---

## Two Repos Strategy (Recommended)

For your CV, use TWO repos:

1. **Private repo: `sketch-generator-training`** — has training scripts, full code, possibly data
2. **Public repo: `sketch-generator-demo`** — has just `app.py`, `multi_object_composition.py`, `requirements.txt`, `deploy/model_for_deploy.pt`, README

The public one is what you put on Streamlit Cloud and link from CV.

See `DEPLOYMENT.md` for the public-repo deployment steps.

---

## Quick Sanity Test

After everything is set up but BEFORE you run the long training, do a sanity check.

Open `train_sketch_v2.py`, find this line near the top:
```python
"max_drawings_per_class": 4000,
"epochs": 30,
```

Change temporarily to:
```python
"max_drawings_per_class": 200,
"epochs": 3,
```

Run:
```bash
python train_sketch_v2.py
```

Should finish in ~10 minutes. Verify:
- Loss decreases epoch over epoch
- No NaN/Inf values
- `outputs_v2_real/` folder is created

If anything fails — STOP. Don't start the real 24-hour training. Fix the issue first (see `CONFIG_NOTES.md` for diagnostics).

Then change values back to 4000 and 30, and start the real training.

---

## Help: Something is broken

- **Read `CONFIG_NOTES.md`** — known issues and how to fix them
- **Read `README_FOR_FRIEND.md`** — has a troubleshooting section
- **Check syntax**: `python -c "import ast; ast.parse(open('FILE.py').read())"`
- **VS Code issue**: try restarting VS Code; pick the right Python interpreter again

---

## Final Word

This is the third or fourth time we've packaged this project. **Don't break the chain — push to GitHub now.** Push early, push often. The cost of pushing is zero. The cost of losing it again is everything.
