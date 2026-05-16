"""
app.py — AI Sketch Generator
Streamlit demo for the BiLSTM-VAE-Transformer sketch generation model.

Deployment: Streamlit Community Cloud
Model file: deploy/model_for_deploy.pt
"""

import io
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import streamlit as st

# Extension: multi-object composition
import multi_object_composition as moc


# =============================================================================
# CONSTANTS
# =============================================================================
MODEL_PATH = "deploy/model_for_deploy.pt"
START_TOKEN = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)

# Tuned per-class temperatures from your sweep results
TUNED_DEFAULTS = {
    "apple":    {"temperature": 0.50, "min_steps": 18, "top_k_mixtures": 5, "eos_bias": 0.005},
    "circle":   {"temperature": 0.30, "min_steps": 22, "top_k_mixtures": 3, "eos_bias": 0.002},
    "star":     {"temperature": 0.20, "min_steps": 22, "top_k_mixtures": 5, "eos_bias": 0.000},
    "triangle": {"temperature": 0.50, "min_steps": 18, "top_k_mixtures": 3, "eos_bias": 0.002},
}

CLASS_EMOJIS = {"apple": "🍎", "circle": "⭕", "star": "⭐", "triangle": "🔺"}


# =============================================================================
# MODEL ARCHITECTURE (must match training)
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, latent_dim, num_classes, class_embed_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=0.0 if num_layers == 1 else dropout, bidirectional=True)
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        self.mu_head = nn.Linear(hidden_dim * 2 + class_embed_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim * 2 + class_embed_dim, latent_dim)

    def forward(self, strokes, lengths, class_ids):
        packed = nn.utils.rnn.pack_padded_sequence(strokes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        max_len = out.size(1)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths.clamp_min(1).unsqueeze(-1)
        pooled = torch.cat([pooled, self.class_embedding(class_ids)], dim=-1)
        return self.mu_head(pooled), self.logvar_head(pooled)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Decoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, ff_dim, dropout, latent_dim,
                 num_classes, class_embed_dim, max_seq_len, num_mixtures):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.input_proj = nn.Linear(input_dim, d_model)
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        self.class_proj = nn.Linear(class_embed_dim, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.fuse = nn.Linear(d_model * 2, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_seq_len + 4)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
                                           dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.mdn_head = nn.Linear(d_model, num_mixtures * 6)
        self.pen_head = nn.Linear(d_model, 3)

    def causal_mask(self, seq_len, device):
        m = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, decoder_inputs, class_ids, z):
        tok = self.input_proj(decoder_inputs)
        cls = self.class_proj(self.class_embedding(class_ids))
        ztok = self.latent_proj(z)
        cls_ctx = cls.unsqueeze(1).expand(-1, tok.size(1), -1)
        tok = self.fuse(torch.cat([tok, cls_ctx], dim=-1))
        h = torch.cat([cls.unsqueeze(1), ztok.unsqueeze(1), tok], dim=1)
        h = self.pos(h)
        h = self.transformer(h, mask=self.causal_mask(h.size(1), h.device))
        h = self.norm(h)
        h = h[:, 2:, :]
        return self.mdn_head(h), self.pen_head(h)


class SketchModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg["latent_dim"]
        self.num_mixtures = cfg["num_mixtures"]
        self.encoder = Encoder(5, cfg["encoder_hidden_dim"], cfg["encoder_layers"],
                               cfg["latent_dim"], len(cfg["classes"]), cfg["class_embed_dim"], cfg["dropout"])
        self.decoder = Decoder(5, cfg["d_model"], cfg["num_heads"], cfg["decoder_layers"],
                               cfg["ff_dim"], cfg["dropout"], cfg["latent_dim"], len(cfg["classes"]),
                               cfg["class_embed_dim"], cfg["max_seq_len"], cfg["num_mixtures"])
        self.register_buffer("start_token", START_TOKEN.clone(), persistent=False)


# =============================================================================
# SAMPLING
# =============================================================================
def unpack_mdn_params(mdn_params, num_mixtures):
    pi_logits, mu_x, mu_y, log_sx, log_sy, rho_raw = torch.split(mdn_params, num_mixtures, dim=-1)
    sx = torch.exp(log_sx).clamp(min=1e-4, max=4.0)
    sy = torch.exp(log_sy).clamp(min=1e-4, max=4.0)
    rho = torch.tanh(rho_raw).clamp(min=-0.95, max=0.95)
    return pi_logits, mu_x, mu_y, sx, sy, rho


def sample_step(mdn_params, pen_logits, num_mixtures, temperature, eos_bias, allow_eos, top_k_mixtures):
    pi_logits, mu_x, mu_y, sx, sy, rho = unpack_mdn_params(mdn_params, num_mixtures)
    if top_k_mixtures is not None and top_k_mixtures < num_mixtures:
        top_vals, top_idx = torch.topk(pi_logits, k=top_k_mixtures, dim=-1)
        masked = torch.full_like(pi_logits, float("-inf"))
        masked.scatter_(dim=-1, index=top_idx, src=top_vals)
        pi_logits = masked
    pi = F.softmax(pi_logits / max(temperature, 1e-4), dim=-1)
    mix_idx = torch.distributions.Categorical(pi).sample()
    bi = torch.arange(mix_idx.size(0), device=mix_idx.device)
    mx = mu_x[bi, mix_idx]; my = mu_y[bi, mix_idx]
    sx_sel = sx[bi, mix_idx]; sy_sel = sy[bi, mix_idx]; r = rho[bi, mix_idx]
    eps_x = torch.randn_like(mx); eps_y = torch.randn_like(my)
    dx = mx + sx_sel * eps_x
    dy = my + sy_sel * (r * eps_x + torch.sqrt(1 - r ** 2 + 1e-6) * eps_y)
    pen_probs = F.softmax(pen_logits / max(temperature, 1e-4), dim=-1).clone()
    pen_probs[:, 2] = pen_probs[:, 2] + eos_bias
    if not allow_eos:
        pen_probs[:, 2] = 0.0
    pen_probs = pen_probs / pen_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    pen_idx = torch.distributions.Categorical(pen_probs).sample()
    pen_onehot = F.one_hot(pen_idx, num_classes=3).float()
    return torch.stack([dx, dy], dim=-1), pen_onehot


@torch.no_grad()
def generate_sketch(model, class_id, settings, device, max_seq_len):
    model.eval()
    class_ids = torch.tensor([class_id], dtype=torch.long, device=device)
    z = torch.randn(1, model.latent_dim, device=device)
    history = model.start_token.to(device).view(1, 1, -1)
    out_tokens = []

    for step in range(max_seq_len):
        mdn_params, pen_logits = model.decoder(history, class_ids, z)
        next_mdn = mdn_params[:, -1, :]
        next_pen_logits = pen_logits[:, -1, :]

        dxdy, pen = sample_step(
            next_mdn, next_pen_logits, num_mixtures=model.num_mixtures,
            temperature=settings["temperature"],
            eos_bias=settings["eos_bias"],
            allow_eos=(step + 1 >= settings["min_steps"]),
            top_k_mixtures=settings.get("top_k_mixtures"),
        )

        tok = torch.cat([dxdy, pen], dim=-1)
        out_tokens.append(tok.squeeze(0))
        history = torch.cat([history, tok.unsqueeze(1)], dim=1)

        if step + 1 >= settings["min_steps"] and pen[0, 2].item() == 1.0:
            break

    if len(out_tokens) == 0:
        return torch.zeros((0, 5), device=device)
    if out_tokens[-1][4].item() != 1.0:
        out_tokens.append(torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], device=device))
    return torch.stack(out_tokens, dim=0)


# =============================================================================
# SCORING — same logic as tune_generation.py, picks well-formed samples
# =============================================================================
def _geometry_features(sequence):
    """Extract closure, roundness, corner_count from a sketch."""
    if isinstance(sequence, torch.Tensor):
        seq_np = sequence.detach().cpu().float().numpy()
    else:
        seq_np = sequence
    pts = []
    x, y = 0.0, 0.0
    for token in seq_np:
        dx, dy = float(token[0]), float(token[1])
        pen_idx = int(np.argmax(token[2:5]))
        x += dx; y += dy
        pts.append([x, y, pen_idx])
        if pen_idx == 2:
            break
    pts = np.asarray(pts, dtype=np.float32)
    pts = pts[pts[:, 2] != 2] if len(pts) > 0 else pts
    if len(pts) < 3:
        return {"length": len(pts), "closure": 1.0, "roundness": -1.0, "corners": 0}
    xy = pts[:, :2]
    bbox_min = xy.min(axis=0); bbox_max = xy.max(axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min) + 1e-6
    closure = float(np.linalg.norm(xy[-1] - xy[0]) / diag)
    center = xy.mean(axis=0)
    radii = np.linalg.norm(xy - center, axis=1)
    roundness = float(1.0 - (np.std(radii) / (np.mean(radii) + 1e-6)))
    segs = np.diff(xy, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    valid = lens > 1e-5
    corners = 0
    if valid.sum() >= 2:
        segs_v = segs[valid]
        ang = np.unwrap(np.arctan2(segs_v[:, 1], segs_v[:, 0]))
        dang = np.abs(np.diff(ang))
        corners = int((dang > 0.45).sum())
    return {"length": len(xy), "closure": closure, "roundness": roundness, "corners": corners}


def score_sample(sequence, class_name):
    """Class-specific geometric score. Higher = better fit for the class."""
    n = len(sequence)
    if n < 8:
        return -1e9
    feat = _geometry_features(sequence)
    closure = feat["closure"]
    roundness = feat["roundness"]
    corners = feat["corners"]
    length = feat["length"]

    if class_name == "circle":
        return 4.0 * roundness - 3.0 * closure - 0.04 * abs(length - 25)
    if class_name == "triangle":
        return -3.0 * closure - 0.5 * abs(corners - 3) - 0.03 * abs(length - 18)
    if class_name == "star":
        return -2.5 * closure - 0.4 * abs(corners - 5) - 0.03 * abs(length - 24)
    if class_name == "apple":
        return -2.5 * closure + 0.4 * max(roundness, 0.0) - 0.03 * abs(length - 22)
    return -closure


def generate_best_of_n(model, class_name, class_id, settings, device, max_seq_len, n_candidates):
    """Generate N candidates, return the highest-scoring one."""
    best_score = -1e18
    best_seq = None
    for _ in range(n_candidates):
        seq = generate_sketch(model, class_id, settings, device, max_seq_len)
        if len(seq) < 3:
            continue
        score = score_sample(seq, class_name)
        if score > best_score:
            best_score = score
            best_seq = seq
    return best_seq, best_score


def generate_top_k_of_n(model, class_name, class_id, settings, device, max_seq_len, n_candidates, k):
    """Generate N candidates, return the top-K by score (sorted high to low)."""
    scored = []
    for _ in range(n_candidates):
        seq = generate_sketch(model, class_id, settings, device, max_seq_len)
        if len(seq) < 3:
            continue
        score = score_sample(seq, class_name)
        scored.append((score, seq))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


# =============================================================================
# RENDERING
# =============================================================================
def stroke5_to_points(sequence):
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.detach().cpu().float().numpy()
    pts = []
    x, y = 0.0, 0.0
    for token in sequence:
        dx, dy = float(token[0]), float(token[1])
        pen_idx = int(np.argmax(token[2:5]))
        x += dx; y += dy
        pts.append([x, y, pen_idx])
        if pen_idx == 2:
            break
    return np.asarray(pts, dtype=np.float32)


def render_static(sequence, figsize=(3.5, 3.5)):
    coords = stroke5_to_points(sequence)
    fig, ax = plt.subplots(figsize=figsize)
    xs, ys = [], []
    for x, y, pen_idx in coords:
        if pen_idx == 2:
            if len(xs) > 1:
                ax.plot(xs, ys, color="black", linewidth=2.5)
            break
        xs.append(x); ys.append(-y)
        if pen_idx == 1:
            if len(xs) > 1:
                ax.plot(xs, ys, color="black", linewidth=2.5)
            xs, ys = [], []
    if len(xs) > 1:
        ax.plot(xs, ys, color="black", linewidth=2.5)
    ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout()
    return fig


def render_animation_gif(sequence, interval=80):
    """Returns a GIF as bytes for st.image().

    matplotlib's ani.save() requires a real file path (not a BytesIO buffer)
    when using the pillow writer. We write to a NamedTemporaryFile, read it
    back as bytes, and clean up.
    """
    import tempfile
    import os as _os

    coords = stroke5_to_points(sequence)
    if len(coords) < 2:
        return None

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    line, = ax.plot([], [], color="black", linewidth=2.5)
    ax.set_aspect("equal"); ax.axis("off")

    # Pre-set axis limits so view doesn't jump around as drawing progresses
    xs_all, ys_all = [], []
    for x, y, p in coords:
        if p != 2:
            xs_all.append(x); ys_all.append(-y)
    if xs_all:
        margin = 0.15
        x_range = max(xs_all) - min(xs_all)
        y_range = max(ys_all) - min(ys_all)
        ax.set_xlim(min(xs_all) - margin * max(x_range, 0.1), max(xs_all) + margin * max(x_range, 0.1))
        ax.set_ylim(min(ys_all) - margin * max(y_range, 0.1), max(ys_all) + margin * max(y_range, 0.1))

    xs, ys = [], []

    def init():
        line.set_data([], [])
        return (line,)

    def update(i):
        x, y, pen_idx = coords[i]
        if pen_idx != 2:
            xs.append(x); ys.append(-y)
            if pen_idx == 1:
                xs.append(np.nan); ys.append(np.nan)
            line.set_data(xs, ys)
        return (line,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(coords), init_func=init,
        interval=interval, blit=True, repeat=False,
    )

    # Write to a temp file, read bytes, delete file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            tmp_path = tmp.name
        ani.save(tmp_path, writer="pillow", fps=12)
        plt.close(fig)
        with open(tmp_path, "rb") as f:
            data = f.read()
        return data
    finally:
        plt.close(fig)
        if tmp_path is not None and _os.path.exists(tmp_path):
            try:
                _os.unlink(tmp_path)
            except OSError:
                pass


# =============================================================================
# MODEL LOADING (cached)
# =============================================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not Path(MODEL_PATH).exists():
        st.error(f"Model file not found at {MODEL_PATH}. Run extract_model.py first.")
        st.stop()

    payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    class_to_idx = payload["class_to_idx"]

    model = SketchModel(cfg)
    # Convert FP16 weights back to FP32 for inference (more stable on CPU)
    weights_fp32 = {k: v.float() if v.dtype == torch.float16 else v for k, v in payload["weights"].items()}
    model.load_state_dict(weights_fp32)
    model.eval()

    return model, cfg, class_to_idx


# =============================================================================
# UI
# =============================================================================
def main():
    st.set_page_config(
        page_title="AI Sketch Generator",
        page_icon="✏️",
        layout="centered",
    )

    # Header
    st.title("✏️ AI Sketch Generator")
    st.caption("BiLSTM-VAE-Transformer trained on Google QuickDraw • by Prachi Sahu, M.Tech IIT Jodhpur")

    # Load model (cached)
    model, cfg, class_to_idx = load_model()
    device = torch.device("cpu")  # Streamlit Cloud is CPU-only
    classes = cfg["classes"]
    max_seq_len = cfg["max_seq_len"]

    # ------------------------------------------------------------------
    # MAIN VIEW: simple class picker + generate
    # ------------------------------------------------------------------
    tab_generate, tab_compare, tab_gallery, tab_compose, tab_about = st.tabs([
        "🎨 Generate", "🌡️ Compare Temperatures", "🖼️ Gallery", "🎬 Compose Scene", "ℹ️ About"
    ])

    # ============== TAB 1: GENERATE ==============
    with tab_generate:
        st.subheader("Pick a class. Watch the AI draw it.")

        col1, col2 = st.columns([2, 1])
        with col1:
            chosen_class = st.selectbox(
                "Choose what to draw:",
                classes,
                format_func=lambda c: f"{CLASS_EMOJIS.get(c, '')} {c.title()}",
                key="gen_class",
            )

        with col2:
            show_animation = st.checkbox("Show animation", value=True, help="Watch the model draw stroke-by-stroke")

        if st.button("✨ Generate", type="primary", use_container_width=True):
            settings = TUNED_DEFAULTS[chosen_class]
            class_id = class_to_idx[chosen_class]

            with st.spinner(f"Generating {chosen_class} (picking best of 6 candidates)..."):
                t0 = time.time()
                # Best-of-N: generate 6 candidates, keep the highest-scoring one.
                # This filters out the occasional "bad sample" the model produces.
                seq, best_score = generate_best_of_n(
                    model, chosen_class, class_id, settings, device, max_seq_len, n_candidates=6
                )
                gen_time = time.time() - t0

            if seq is None or len(seq) < 3:
                st.warning("Generation produced too few strokes. Try again.")
            else:
                col_a, col_b = st.columns([3, 2])
                with col_a:
                    if show_animation:
                        st.write("**Watch it draw:**")
                        try:
                            gif_bytes = render_animation_gif(seq)
                            if gif_bytes:
                                st.image(gif_bytes, use_container_width=True)
                            else:
                                st.write("(animation unavailable, showing static)")
                                fig = render_static(seq)
                                st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Animation failed ({type(e).__name__}); showing static image.")
                            fig = render_static(seq)
                            st.pyplot(fig)
                    else:
                        st.write("**Generated sketch:**")
                        fig = render_static(seq)
                        st.pyplot(fig)

                with col_b:
                    st.write("**Details:**")
                    st.write(f"- Class: **{chosen_class}**")
                    st.write(f"- Temperature: **{settings['temperature']}**")
                    st.write(f"- Strokes: **{len(seq)}**")
                    st.write(f"- Time: **{gen_time:.2f}s**")

                    pen_idx = np.argmax(seq.cpu().numpy()[:, 2:5], axis=-1)
                    pen_down = int((pen_idx == 0).sum())
                    pen_up = int((pen_idx == 1).sum())
                    st.write(f"- Pen lifts: **{pen_up}**")

        st.divider()
        st.caption("💡 Click Generate again for a different sample. Each generation is unique because the model samples from a learned distribution.")

    # ============== TAB 2: COMPARE TEMPERATURES ==============
    with tab_compare:
        st.subheader("How temperature affects generation")
        st.write(
            "Temperature controls how 'creative' the model is. "
            "Lower = conservative (sticks to learned patterns). "
            "Higher = creative (more variation, risk of artifacts)."
        )

        compare_class = st.selectbox(
            "Class to compare:",
            classes,
            format_func=lambda c: f"{CLASS_EMOJIS.get(c, '')} {c.title()}",
            key="compare_class",
        )

        if st.button("🌡️ Compare 4 temperatures", type="primary", use_container_width=True):
            temperatures = [0.20, 0.40, 0.60, 0.80]
            class_id = class_to_idx[compare_class]
            base = TUNED_DEFAULTS[compare_class]

            cols = st.columns(4)
            progress = st.progress(0.0, text="Generating...")

            # Best-of-3 per temperature: gives a fairer comparison than single-shot,
            # since otherwise a bad random sample at one temperature could mislead.
            for i, temp in enumerate(temperatures):
                settings = {**base, "temperature": temp}
                seq, _ = generate_best_of_n(
                    model, compare_class, class_id, settings, device, max_seq_len, n_candidates=3
                )
                with cols[i]:
                    st.caption(f"**T = {temp}**")
                    if seq is not None and len(seq) >= 3:
                        fig = render_static(seq, figsize=(2.5, 2.5))
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.write("(failed)")
                progress.progress((i + 1) / len(temperatures), text=f"Generated {i+1}/{len(temperatures)}")

            progress.empty()

            tuned_t = base["temperature"]
            st.info(
                f"💡 The optimal temperature for **{compare_class}** (found via sweep) is **T = {tuned_t}**. "
                f"This was discovered by generating 12 candidates per temperature and scoring them with class-specific geometric metrics. "
                f"Each cell above shows the best of 3 candidates per temperature."
            )

    # ============== TAB 3: GALLERY ==============
    with tab_gallery:
        st.subheader("Generation variety")
        st.write("Generates multiple samples of the chosen class to show variation.")

        col1, col2 = st.columns([2, 1])
        with col1:
            gallery_class = st.selectbox(
                "Class:",
                classes,
                format_func=lambda c: f"{CLASS_EMOJIS.get(c, '')} {c.title()}",
                key="gallery_class",
            )
        with col2:
            n_samples = st.slider("Samples", min_value=4, max_value=9, value=6, step=1)

        if st.button("🖼️ Generate gallery", type="primary", use_container_width=True):
            settings = TUNED_DEFAULTS[gallery_class]
            class_id = class_to_idx[gallery_class]

            n_cols = 3
            n_rows = (n_samples + n_cols - 1) // n_cols

            # Best-of-N filtering: generate 3x more candidates than displayed,
            # keep the top-N by geometric score. This removes the visibly broken
            # samples while preserving variety among well-formed ones.
            n_candidates = n_samples * 3
            progress = st.progress(0.0, text=f"Generating {n_candidates} candidates...")

            scored = []
            for i in range(n_candidates):
                seq = generate_sketch(model, class_id, settings, device, max_seq_len)
                if len(seq) >= 8:
                    score = score_sample(seq, gallery_class)
                    scored.append((score, seq))
                progress.progress((i + 1) / n_candidates, text=f"Candidate {i+1}/{n_candidates}")

            progress.empty()

            # Pick top-N by score
            scored.sort(key=lambda x: x[0], reverse=True)
            top_sequences = [s for _, s in scored[:n_samples]]

            if len(top_sequences) < n_samples:
                st.warning(f"Only {len(top_sequences)} valid samples produced; showing what we have.")

            st.caption(
                f"💡 Showing top {len(top_sequences)} of {n_candidates} candidates "
                f"(filtered by class-specific geometric score)."
            )

            for r in range(n_rows):
                row_cols = st.columns(n_cols)
                for c in range(n_cols):
                    idx = r * n_cols + c
                    if idx < len(top_sequences):
                        with row_cols[c]:
                            seq = top_sequences[idx]
                            # Fixed figure size for consistent layout
                            fig = render_static(seq, figsize=(2.5, 2.5))
                            st.pyplot(fig, use_container_width=True)

    # ============== TAB 4: COMPOSE SCENE (Extension #2) ==============
    with tab_compose:
        st.subheader("Compose a multi-object scene")
        st.write(
            "Pick objects to include in the scene. The model generates each object "
            "individually and places them on a single canvas. This is the *multi-object "
            "composition* extension of the project."
        )

        # User picks how many of each object to include
        st.write("**Step 1: Choose objects**")
        cols_count = st.columns(len(classes))
        object_counts = {}
        for i, c in enumerate(classes):
            with cols_count[i]:
                object_counts[c] = st.number_input(
                    f"{CLASS_EMOJIS.get(c, '')} {c}",
                    min_value=0, max_value=4, value=1 if c in ("apple", "circle") else 0,
                    key=f"count_{c}",
                )

        # Build the actual list of class names (with duplicates)
        class_names = []
        for c, count in object_counts.items():
            class_names.extend([c] * int(count))

        st.write(f"**Selected:** {len(class_names)} object(s) — {class_names if class_names else '(none yet)'}")

        st.write("**Step 2: Choose layout**")
        col_l1, col_l2 = st.columns([2, 1])
        with col_l1:
            layout_name = st.selectbox(
                "Layout strategy:",
                list(moc.LAYOUT_STRATEGIES.keys()),
                index=1,  # default to Grid
                key="compose_layout",
            )
        with col_l2:
            object_scale = st.slider("Object size", 0.8, 2.5, 1.4, 0.1, key="compose_scale")

        show_labels = st.checkbox("Show object labels under each sketch", value=True, key="compose_labels")

        if st.button("🎬 Compose scene", type="primary", use_container_width=True, disabled=(len(class_names) == 0)):
            if len(class_names) > 8:
                st.warning("Limiting scene to 8 objects (more would be slow on CPU).")
                class_names = class_names[:8]

            # Wrap our best-of-N generator for the composition module
            def gen_for_class(class_name):
                settings = TUNED_DEFAULTS[class_name]
                class_id = class_to_idx[class_name]
                seq, _ = generate_best_of_n(
                    model, class_name, class_id, settings, device, max_seq_len, n_candidates=4
                )
                if seq is None:
                    # fallback: single generation if best-of-N completely failed
                    seq = generate_sketch(model, class_id, settings, device, max_seq_len)
                return seq.cpu().numpy()

            progress = st.progress(0.0, text=f"Generating {len(class_names)} objects...")
            t0 = time.time()

            # Generate sequences one at a time so we can show progress
            sequences = []
            for i, cn in enumerate(class_names):
                seq = gen_for_class(cn)
                sequences.append(seq)
                progress.progress((i + 1) / len(class_names), text=f"Generated {i+1}/{len(class_names)}: {cn}")

            # Compose using pre-generated sequences (skip re-generation in compose_scene)
            from multi_object_composition import PlacedObject, compute_bbox, LAYOUT_STRATEGIES, render_scene, describe_scene
            canvas_size = 10.0
            layout_fn = LAYOUT_STRATEGIES[layout_name]
            centers = layout_fn(len(sequences), canvas_size=canvas_size)
            placed = []
            for i, seq in enumerate(sequences):
                placed.append(PlacedObject(
                    class_name=class_names[i],
                    sequence=seq,
                    bbox=compute_bbox(seq),
                    center=centers[i] if i < len(centers) else (0.0, 0.0),
                    scale=object_scale,
                ))

            elapsed = time.time() - t0
            progress.empty()

            fig = render_scene(
                placed,
                canvas_size=canvas_size,
                show_labels=show_labels,
                figsize=(7, 7),
            )
            st.pyplot(fig, use_container_width=True)

            st.success(f"✅ Generated in {elapsed:.1f}s — {moc.describe_scene(placed)}")

            with st.expander("How does this work?"):
                st.markdown("""
                1. **Generate**: For each selected object, the model generates a sketch independently using best-of-4 selection.
                2. **Bounding box**: We compute the bounding box of each sketch's strokes.
                3. **Layout**: A chosen strategy (grid / row / random / circular) decides where to place each object's center on the scene canvas.
                4. **Render**: Each sketch is normalized to its bounding box center, scaled, and translated to its target position. All objects render onto one figure.

                **Limitation**: Objects are placed independently — they don't *interact* with each other (no occlusion, no spatial reasoning).
                Adding spatial reasoning (e.g., "apple ON TOP of triangle") would require training the model on multi-object data, which is out of scope here.
                """)

    # ============== TAB 5: ABOUT ==============
    with tab_about:
        st.subheader("About this project")

        st.markdown("""
**A class-conditional generative model that learns to draw simple sketches stroke-by-stroke.**

### Architecture
- **Encoder:** Bidirectional LSTM compresses input sketch into a latent vector
- **VAE:** Regularizes latent space using free-bits to enable sampling
- **Decoder:** Causal Transformer generates strokes autoregressively
- **MDN head:** Predicts pen movement (dx, dy) as a mixture of 16 bivariate Gaussians
- **Pen head:** Classifies pen state (down / up / end-of-sketch)

### Training
- Dataset: Google QuickDraw, 4 classes (apple, circle, star, triangle)
- 4,000 drawings per class, 30 epochs on a GTX 1080 Ti
- Stroke simplification via Ramer-Douglas-Peucker
- Free bits + KL annealing prevents posterior collapse
- Weighted pen loss + class weights to learn rare end-of-sketch tokens

### Key Engineering Insights
1. **Per-class accuracy hides failures.** A 97% global pen accuracy was masking 0% accuracy on rare classes — diagnostic per-class tracking caught this.
2. **Sampling matters as much as training.** Different classes need different temperatures: stars want T=0.20, apples want T=0.50.
3. **Cheap interventions first.** Temperature tuning recovered exceptional samples without retraining.

### Architecture diagram

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
""")

        st.divider()
        st.caption("Built by Prachi Sahu • IIT Jodhpur • [GitHub Repo](#)")


if __name__ == "__main__":
    main()
