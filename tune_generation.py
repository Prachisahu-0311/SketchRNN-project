"""
tune_generation.py

Systematic generation tuning for the trained v7 model.
NO RETRAINING. Loads final.pt and explores generation settings.

Outputs:
  - outputs_v7_tuned/temperature_sweep_<class>.png  (grid of samples at different temps)
  - outputs_v7_tuned/best_grid.png                  (final best samples per class)
  - outputs_v7_tuned/<class>_best_<n>.gif           (animated best samples)
  - outputs_v7_tuned/tuning_report.json             (which settings won per class)

Usage:
    python tune_generation.py
"""

import os
import sys
import json
import math
import copy
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# =============================================================================
# CONFIG — match training-time architecture, sweep generation settings
# =============================================================================
SOURCE_OUTPUT_DIR = "./outputs_v2"             # where final.pt lives
TUNED_OUTPUT_DIR = "./outputs_v7_tuned"        # where new outputs go (separate from v7 baseline)

CONFIG = {
    "seed": 42,
    "classes": ["apple", "circle", "star", "triangle"],

    # Architecture (must match training)
    "max_seq_len": 80,
    "encoder_hidden_dim": 192,
    "encoder_layers": 1,
    "latent_dim": 128,
    "d_model": 192,
    "num_heads": 6,
    "decoder_layers": 4,
    "ff_dim": 384,
    "class_embed_dim": 128,
    "dropout": 0.12,
    "num_mixtures": 16,
}

# Per-class sweep: each class tested across multiple temperatures
TEMPERATURE_SWEEP = [0.20, 0.30, 0.40, 0.50, 0.65, 0.80]

# Per-class baseline settings (we override temperature in the sweep)
PER_CLASS_DEFAULTS = {
    "apple":    {"min_steps": 18, "top_k_mixtures": 5, "eos_bias": 0.005, "pen_mode": "sample"},
    "circle":   {"min_steps": 22, "top_k_mixtures": 3, "eos_bias": 0.002, "pen_mode": "sample"},
    "star":     {"min_steps": 22, "top_k_mixtures": 5, "eos_bias": 0.000, "pen_mode": "sample"},
    "triangle": {"min_steps": 18, "top_k_mixtures": 3, "eos_bias": 0.002, "pen_mode": "sample"},
}

# How many candidates to generate per (class, temperature) cell
CANDIDATES_PER_CELL = 12   # was 8 in v7
# How many top samples per cell to display in the sweep grid
TOP_K_DISPLAY = 3
# Final best-of-N for the showcase grid
FINAL_CANDIDATES_PER_CLASS = 30  # was 8 in v7
FINAL_SHOWCASE_PER_CLASS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_TOKEN = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


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


class VAE(nn.Module):
    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std


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


def unpack_mdn_params(mdn_params, num_mixtures):
    pi_logits, mu_x, mu_y, log_sx, log_sy, rho_raw = torch.split(mdn_params, num_mixtures, dim=-1)
    sx = torch.exp(log_sx).clamp(min=1e-4, max=4.0)
    sy = torch.exp(log_sy).clamp(min=1e-4, max=4.0)
    rho = torch.tanh(rho_raw).clamp(min=-0.95, max=0.95)
    return pi_logits, mu_x, mu_y, sx, sy, rho


def sample_mdn_step(mdn_params, pen_logits, num_mixtures, temperature, eos_bias, allow_eos, pen_mode, top_k_mixtures):
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
    if pen_mode == "greedy":
        pen_idx = torch.argmax(pen_probs, dim=-1)
    else:
        pen_idx = torch.distributions.Categorical(pen_probs).sample()
    pen_onehot = F.one_hot(pen_idx, num_classes=3).float()
    return torch.stack([dx, dy], dim=-1), pen_onehot


class SketchModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg["latent_dim"]
        self.num_mixtures = cfg["num_mixtures"]
        self.encoder = Encoder(5, cfg["encoder_hidden_dim"], cfg["encoder_layers"],
                               cfg["latent_dim"], len(cfg["classes"]), cfg["class_embed_dim"], cfg["dropout"])
        self.vae = VAE()
        self.decoder = Decoder(5, cfg["d_model"], cfg["num_heads"], cfg["decoder_layers"],
                               cfg["ff_dim"], cfg["dropout"], cfg["latent_dim"], len(cfg["classes"]),
                               cfg["class_embed_dim"], cfg["max_seq_len"], cfg["num_mixtures"])
        self.register_buffer("start_token", START_TOKEN.clone(), persistent=False)

    @torch.no_grad()
    def generate_with_settings(self, class_id, settings, device):
        """Generate one sample with specific settings (no caching defaults)."""
        self.eval()
        class_ids = torch.tensor([class_id], dtype=torch.long, device=device)
        z = torch.randn(1, self.latent_dim, device=device)
        history = self.start_token.to(device).view(1, 1, -1)
        out_tokens = []

        for step in range(self.cfg["max_seq_len"]):
            mdn_params, pen_logits = self.decoder(history, class_ids, z)
            next_mdn = mdn_params[:, -1, :]
            next_pen_logits = pen_logits[:, -1, :]

            dxdy, pen = sample_mdn_step(
                next_mdn, next_pen_logits, num_mixtures=self.num_mixtures,
                temperature=settings["temperature"],
                eos_bias=settings["eos_bias"],
                allow_eos=(step + 1 >= settings["min_steps"]),
                pen_mode=settings["pen_mode"],
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
# RENDERING & SCORING
# =============================================================================
def stroke5_to_absolute_points(sequence):
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


def render_sketch(sequence, ax=None, title=None, title_fontsize=8):
    coords = stroke5_to_absolute_points(sequence)
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    xs, ys = [], []
    for x, y, pen_idx in coords:
        if pen_idx == 2:
            if len(xs) > 1:
                ax.plot(xs, ys, color="black", linewidth=1.5)
            break
        xs.append(x); ys.append(-y)
        if pen_idx == 1:
            if len(xs) > 1:
                ax.plot(xs, ys, color="black", linewidth=1.5)
            xs, ys = [], []
    if len(xs) > 1:
        ax.plot(xs, ys, color="black", linewidth=1.5)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    return ax


def animate_sketch(sequence, save_path, title=None, interval=60):
    coords = stroke5_to_absolute_points(sequence)
    fig, ax = plt.subplots(figsize=(4, 4))
    line, = ax.plot([], [], color="black", linewidth=2)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title)
    xs, ys = [], []

    def init():
        line.set_data([], []); return (line,)

    def update(i):
        x, y, pen_idx = coords[i]
        if pen_idx != 2:
            xs.append(x); ys.append(-y)
            if pen_idx == 1:
                xs.append(np.nan); ys.append(np.nan)
            line.set_data(xs, ys)
            ax.relim(); ax.autoscale_view()
        return (line,)

    ani = animation.FuncAnimation(fig, update, frames=len(coords), init_func=init,
                                   interval=interval, blit=True, repeat=False)
    ani.save(save_path, writer="pillow")
    plt.close(fig)


def geometry_features(sequence):
    pts = stroke5_to_absolute_points(sequence)
    pts = pts[pts[:, 2] != 2] if len(pts) > 0 else pts
    if len(pts) < 3:
        return {"length": len(pts), "closure": 1.0, "roundness": -1.0, "corner_count": 0, "self_intersect": 0}
    xy = pts[:, :2]
    bbox_min = xy.min(axis=0); bbox_max = xy.max(axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min) + 1e-6
    closure = np.linalg.norm(xy[-1] - xy[0]) / diag
    center = xy.mean(axis=0)
    radii = np.linalg.norm(xy - center, axis=1)
    roundness = 1.0 - (np.std(radii) / (np.mean(radii) + 1e-6))
    segs = np.diff(xy, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    valid = lens > 1e-5
    corner_count = 0
    if valid.sum() >= 2:
        segs_v = segs[valid]
        ang = np.unwrap(np.arctan2(segs_v[:, 1], segs_v[:, 0]))
        dang = np.abs(np.diff(ang))
        corner_count = int((dang > 0.45).sum())
    return {
        "length": len(xy), "closure": float(closure),
        "roundness": float(roundness), "corner_count": int(corner_count),
    }


def score_sample(sequence, class_name, model, class_id, device):
    """Combined score: class-specific geometry + encoder-based plausibility."""
    feat = geometry_features(sequence)
    n = feat["length"]
    closure = feat["closure"]
    roundness = feat["roundness"]
    corners = feat["corner_count"]

    if n < 8:
        return -1e9, feat  # too short

    # Class-specific geometric reward
    if class_name == "circle":
        geom_score = 4.0 * roundness - 3.0 * closure - 0.04 * abs(n - 25)
    elif class_name == "triangle":
        geom_score = -3.0 * closure - 0.50 * abs(corners - 3) - 0.03 * abs(n - 18)
    elif class_name == "star":
        # Stars want 5 corners, NOT closed (single continuous zigzag)
        geom_score = -2.5 * closure - 0.40 * abs(corners - 5) - 0.03 * abs(n - 24)
    elif class_name == "apple":
        # Apple: closed body + small extension at top (stem)
        geom_score = -2.5 * closure + 0.4 * max(roundness, 0.0) - 0.03 * abs(n - 22)
    else:
        geom_score = -closure

    # Encoder-based plausibility: re-encode generated and check it's near prior
    enc_score = 0.0
    try:
        if len(sequence) >= 3:
            seq_in = sequence.unsqueeze(0).to(device).float()
            length = torch.tensor([len(sequence)], dtype=torch.long, device=device)
            cls = torch.tensor([class_id], dtype=torch.long, device=device)
            mu, _ = model.encoder(seq_in, length, cls)
            # Negative L2 from origin = closer to prior = more typical
            enc_score = -mu.pow(2).sum(dim=-1).item() / 100.0  # scale down
    except Exception:
        enc_score = -10.0

    combined = geom_score + 0.3 * enc_score
    return combined, feat


# =============================================================================
# MAIN
# =============================================================================
def main():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])

    print(f"Device: {DEVICE}")

    # Load checkpoint
    src_dir = Path(SOURCE_OUTPUT_DIR)
    ckpt_path = src_dir / "final.pt"
    if not ckpt_path.exists():
        ckpt_path = src_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = src_dir / "latest.pt"
    if not ckpt_path.exists():
        print(f"ERROR: No checkpoint in {src_dir}")
        sys.exit(1)

    print(f"Loading: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = payload.get("config", CONFIG)

    # Sync architecture from saved config
    arch_keys = ["encoder_hidden_dim", "latent_dim", "d_model", "num_heads",
                 "decoder_layers", "ff_dim", "class_embed_dim", "num_mixtures",
                 "max_seq_len", "classes", "dropout"]
    for k in arch_keys:
        if k in saved_cfg and saved_cfg[k] != CONFIG[k]:
            CONFIG[k] = saved_cfg[k]

    class_to_idx = payload.get("class_to_idx", {c: i for i, c in enumerate(CONFIG["classes"])})
    print(f"Classes: {class_to_idx}")

    # Build model & load EMA weights (preferred)
    model = SketchModel(CONFIG).to(DEVICE)
    if payload.get("ema_state") is not None:
        try:
            model.load_state_dict(payload["ema_state"])
            print("Loaded EMA weights")
        except Exception as e:
            print(f"  EMA load failed ({e}), using raw weights")
            model.load_state_dict(payload["model_state"])
    else:
        model.load_state_dict(payload["model_state"])
    model.eval()

    # Output dir
    out_dir = Path(TUNED_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir = out_dir / "sweeps"
    sweep_dir.mkdir(exist_ok=True)

    # =====================================================
    # PHASE 1: TEMPERATURE SWEEP per class
    # =====================================================
    print("\n=== PHASE 1: Temperature sweep ===")
    tuning_report = {}

    for class_name in CONFIG["classes"]:
        print(f"\n--- {class_name} ---")
        class_id = class_to_idx[class_name]
        defaults = PER_CLASS_DEFAULTS[class_name]

        # Grid: rows = temperatures, columns = top-K display samples
        n_temps = len(TEMPERATURE_SWEEP)
        fig, axes = plt.subplots(n_temps, TOP_K_DISPLAY, figsize=(2.5 * TOP_K_DISPLAY, 2.5 * n_temps))
        if n_temps == 1:
            axes = np.expand_dims(axes, axis=0)
        if TOP_K_DISPLAY == 1:
            axes = np.expand_dims(axes, axis=1)

        per_temp_results = {}

        for ti, temp in enumerate(TEMPERATURE_SWEEP):
            settings = {**defaults, "temperature": temp}

            # Generate candidates at this temperature
            candidates = []
            for _ in range(CANDIDATES_PER_CELL):
                seq = model.generate_with_settings(class_id, settings, DEVICE)
                if len(seq) >= 8:
                    score, feat = score_sample(seq, class_name, model, class_id, DEVICE)
                    candidates.append((score, seq, feat))

            if not candidates:
                print(f"  T={temp:.2f}: ALL FAILED (sequences too short)")
                continue

            # Sort by score (high to low)
            candidates.sort(key=lambda x: x[0], reverse=True)

            avg_score = np.mean([c[0] for c in candidates])
            best_score = candidates[0][0]
            avg_length = np.mean([c[2]["length"] for c in candidates])

            per_temp_results[temp] = {
                "n_valid": len(candidates),
                "avg_score": float(avg_score),
                "best_score": float(best_score),
                "avg_length": float(avg_length),
            }
            print(f"  T={temp:.2f}: n={len(candidates)}, avg_score={avg_score:.3f}, best={best_score:.3f}, avg_len={avg_length:.1f}")

            # Render top-K for the sweep grid
            for j in range(min(TOP_K_DISPLAY, len(candidates))):
                seq = candidates[j][1].cpu()
                title = f"T={temp:.2f}" if j == 0 else ""
                render_sketch(seq, axes[ti, j], title=title)

            # Pad remaining columns
            for j in range(len(candidates), TOP_K_DISPLAY):
                axes[ti, j].axis("off")

        plt.suptitle(f"{class_name}: temperature sweep (rows=T, cols=top-{TOP_K_DISPLAY})", fontsize=12)
        plt.tight_layout()
        sweep_path = sweep_dir / f"sweep_{class_name}.png"
        plt.savefig(sweep_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {sweep_path}")

        # Pick best temperature for this class (by avg score, with sanity)
        if per_temp_results:
            best_temp = max(per_temp_results.items(), key=lambda x: x[1]["avg_score"])[0]
            print(f"  >>> Best T for {class_name}: {best_temp:.2f}")
            tuning_report[class_name] = {
                "best_temperature": float(best_temp),
                "per_temp_results": per_temp_results,
            }
        else:
            print(f"  WARNING: No valid samples for {class_name}")
            tuning_report[class_name] = {"best_temperature": None, "per_temp_results": {}}

    # =====================================================
    # PHASE 2: Generate FINAL showcase using winning temperatures
    # =====================================================
    print("\n=== PHASE 2: Generating final showcase ===")

    fig, axes = plt.subplots(FINAL_SHOWCASE_PER_CLASS, len(CONFIG["classes"]),
                              figsize=(4 * len(CONFIG["classes"]), 4 * FINAL_SHOWCASE_PER_CLASS))
    if FINAL_SHOWCASE_PER_CLASS == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(CONFIG["classes"]) == 1:
        axes = np.expand_dims(axes, axis=1)

    final_diagnostics = {}

    for col, class_name in enumerate(CONFIG["classes"]):
        info = tuning_report.get(class_name, {})
        best_temp = info.get("best_temperature", 0.4)
        if best_temp is None:
            best_temp = 0.4
        defaults = PER_CLASS_DEFAULTS[class_name]
        settings = {**defaults, "temperature": best_temp}
        class_id = class_to_idx[class_name]

        print(f"\n{class_name}: T={best_temp:.2f}, generating {FINAL_CANDIDATES_PER_CLASS} candidates...")

        candidates = []
        for _ in range(FINAL_CANDIDATES_PER_CLASS):
            seq = model.generate_with_settings(class_id, settings, DEVICE)
            if len(seq) >= 8:
                score, feat = score_sample(seq, class_name, model, class_id, DEVICE)
                candidates.append((score, seq, feat))

        candidates.sort(key=lambda x: x[0], reverse=True)
        n_valid = len(candidates)
        print(f"  {n_valid}/{FINAL_CANDIDATES_PER_CLASS} valid candidates")

        final_diagnostics[class_name] = {
            "best_temperature": float(best_temp),
            "n_candidates_generated": FINAL_CANDIDATES_PER_CLASS,
            "n_valid": n_valid,
            "showcase": [],
        }

        # Render top-K and save GIFs
        for row in range(FINAL_SHOWCASE_PER_CLASS):
            if row < len(candidates):
                score, seq, feat = candidates[row]
                seq_cpu = seq.cpu()
                render_sketch(seq_cpu, axes[row, col], title=f"{class_name} #{row+1} (T={best_temp:.2f})", title_fontsize=10)

                # GIF
                gif_path = out_dir / f"{class_name}_best_{row+1}.gif"
                try:
                    animate_sketch(seq_cpu, gif_path, title=f"{class_name} #{row+1}")
                except Exception as e:
                    print(f"  GIF failed for {class_name} #{row+1}: {e}")

                final_diagnostics[class_name]["showcase"].append({
                    "rank": row + 1,
                    "score": float(score),
                    "length": int(feat["length"]),
                    "closure": float(feat["closure"]),
                    "roundness": float(feat["roundness"]),
                    "corners": int(feat["corner_count"]),
                })
                print(f"  #{row+1}: score={score:.3f}, len={feat['length']}, closure={feat['closure']:.2f}, corners={feat['corner_count']}")
            else:
                axes[row, col].axis("off")

    plt.suptitle("v7 Tuned: Best-of-N generation per class", fontsize=14)
    plt.tight_layout()
    showcase_path = out_dir / "best_grid.png"
    plt.savefig(showcase_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved final showcase: {showcase_path}")

    # Save report
    report_path = out_dir / "tuning_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "tuning_report": tuning_report,
            "final_diagnostics": final_diagnostics,
            "settings": {
                "temperature_sweep": TEMPERATURE_SWEEP,
                "candidates_per_cell": CANDIDATES_PER_CELL,
                "final_candidates_per_class": FINAL_CANDIDATES_PER_CLASS,
                "per_class_defaults": PER_CLASS_DEFAULTS,
            },
        }, f, indent=2)
    print(f"Saved report: {report_path}")

    print("\n=== DONE ===")
    print(f"Open {showcase_path} to see best samples")
    print(f"Open {sweep_dir}/ to see temperature sweeps per class")


if __name__ == "__main__":
    main()
