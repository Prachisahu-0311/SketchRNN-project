"""
generate_only.py

Loads the already-trained model from final.pt and generates samples.
Does NOT retrain. Use this to recover from generation-time bugs without
losing your 24-hour training run.

Usage:
    python generate_only.py

Output:
    outputs_v2_real/results_grid.png
    outputs_v2_real/<class>_sample_<n>.gif
    outputs_v2_real/diagnostics.json
"""

import os
import sys
import json
import math
import copy
import random
from pathlib import Path
from dataclasses import dataclass

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
# CONFIG — must match what the model was trained with
# (load_ckpt will override most of this from the saved config, but generation
# settings are read from THIS dict so we can iterate on sampling without retrain)
# =============================================================================
CONFIG = {
    "seed": 42,
    "data_dir": "./data",
    "output_dir": "./outputs_v2_real",   # SAME as training run

    "classes": ["apple", "circle", "star", "triangle"],

    # Must match training (used for model architecture)
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

    # Generation settings — TUNE THESE without retraining
    "samples_per_class": 3,
    "generation_candidates": 8,
    "save_gifs": True,

    "generation_defaults": {
        "temperature": 0.4,
        "min_steps": 15,
        "eos_bias": 0.005,
        "top_k_mixtures": 5,
        "pen_mode": "sample",
    },
    "generation_overrides": {
        "apple":    {"temperature": 0.38, "min_steps": 18},
        "circle":   {"temperature": 0.32, "min_steps": 22, "top_k_mixtures": 3},
        "star":     {"temperature": 0.35, "min_steps": 22, "top_k_mixtures": 3},
        "triangle": {"temperature": 0.34, "min_steps": 18, "top_k_mixtures": 3},
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_TOKEN = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


# =============================================================================
# COPY THE NECESSARY CLASSES FROM train4.py
# (we need these to instantiate the model and load weights)
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
    def generate_once(self, class_name, class_id, cfg, device):
        # FIX: Use this script's CONFIG explicitly, not the loaded checkpoint's config
        settings = dict(cfg["generation_defaults"])
        settings.update(cfg["generation_overrides"].get(class_name, {}))

        self.eval()
        class_ids = torch.tensor([class_id], dtype=torch.long, device=device)
        z = torch.randn(1, self.latent_dim, device=device)
        history = self.start_token.to(device).view(1, 1, -1)
        out_tokens = []

        for step in range(cfg["max_seq_len"]):
            mdn_params, pen_logits = self.decoder(history, class_ids, z)
            next_mdn = mdn_params[:, -1, :]
            next_pen_logits = pen_logits[:, -1, :]

            dxdy, pen = sample_mdn_step(
                next_mdn, next_pen_logits, num_mixtures=self.num_mixtures,
                temperature=settings["temperature"], eos_bias=settings["eos_bias"],
                allow_eos=(step + 1 >= settings["min_steps"]),
                pen_mode=settings["pen_mode"], top_k_mixtures=settings.get("top_k_mixtures"),
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
# RENDERING
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


def render_sketch(sequence, ax=None, title=None):
    coords = stroke5_to_absolute_points(sequence)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    xs, ys = [], []
    for x, y, pen_idx in coords:
        if pen_idx == 2:
            if len(xs) > 1:
                ax.plot(xs, ys, color="black", linewidth=2)
            break
        xs.append(x); ys.append(-y)
        if pen_idx == 1:
            if len(xs) > 1:
                ax.plot(xs, ys, color="black", linewidth=2)
            xs, ys = [], []
    if len(xs) > 1:
        ax.plot(xs, ys, color="black", linewidth=2)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title)
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
    ckpt_path = Path(CONFIG["output_dir"]) / "final.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(CONFIG["output_dir"]) / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(CONFIG["output_dir"]) / "latest.pt"
    if not ckpt_path.exists():
        print(f"ERROR: No checkpoint found in {CONFIG['output_dir']}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # CRITICAL: Use saved config for ARCHITECTURE, but our CONFIG for GENERATION SETTINGS
    saved_cfg = payload.get("config", CONFIG)

    # Verify architecture compat
    arch_keys = ["encoder_hidden_dim", "latent_dim", "d_model", "num_heads",
                 "decoder_layers", "ff_dim", "class_embed_dim", "num_mixtures",
                 "max_seq_len", "classes"]
    for k in arch_keys:
        if k in saved_cfg and saved_cfg[k] != CONFIG[k]:
            print(f"WARNING: arch mismatch on {k}: saved={saved_cfg[k]}, current={CONFIG[k]}")
            print(f"  Using saved value to load weights correctly.")
            CONFIG[k] = saved_cfg[k]

    class_to_idx = payload.get("class_to_idx", {c: i for i, c in enumerate(CONFIG["classes"])})
    print(f"Classes: {class_to_idx}")

    # Build model
    model = SketchModel(CONFIG).to(DEVICE)

    # Load EMA weights (these are what was used for evaluation in training)
    if payload.get("ema_state") is not None:
        print("Loading EMA weights (preferred for inference)")
        try:
            model.load_state_dict(payload["ema_state"])
        except Exception as e:
            print(f"  EMA load failed ({e}), falling back to model_state")
            model.load_state_dict(payload["model_state"])
    else:
        print("Loading model_state weights")
        model.load_state_dict(payload["model_state"])

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params/1e6:.2f}M params")

    # Generate samples
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(CONFIG["classes"])
    rows = CONFIG["samples_per_class"]
    fig, axes = plt.subplots(rows, num_classes, figsize=(4 * num_classes, 3.7 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_classes == 1:
        axes = np.expand_dims(axes, axis=1)

    diagnostics = {}

    for col, class_name in enumerate(CONFIG["classes"]):
        print(f"\nGenerating samples for: {class_name}")
        diagnostics[class_name] = []
        class_id = class_to_idx[class_name]

        for row in range(CONFIG["samples_per_class"]):
            # Generate `generation_candidates` and pick best by length-not-too-extreme heuristic
            best_seq = None
            best_score = -1e9
            for cand in range(CONFIG["generation_candidates"]):
                seq = model.generate_once(class_name, class_id, CONFIG, DEVICE)
                # Simple score: prefer sequences that are not too short (>10) and not at max
                length = len(seq)
                if length < 8:
                    score = -100
                elif length >= CONFIG["max_seq_len"] - 2:
                    score = -50  # didn't terminate properly
                else:
                    score = -abs(length - 25)  # prefer ~25 length
                if score > best_score:
                    best_score = score
                    best_seq = seq

            seq_cpu = best_seq.cpu()
            render_sketch(seq_cpu, axes[row, col], title=f"{class_name} #{row + 1}")

            # GIF
            if CONFIG["save_gifs"]:
                gif_path = out_dir / f"{class_name}_sample_{row + 1}.gif"
                try:
                    animate_sketch(seq_cpu, gif_path, title=f"{class_name} #{row + 1}")
                    print(f"  saved {gif_path.name}")
                except Exception as e:
                    print(f"  GIF failed for {class_name} #{row + 1}: {e}")

            # Diagnostics
            seq_np = seq_cpu.numpy()
            if len(seq_np) > 0:
                pen_idx = np.argmax(seq_np[:, 2:5], axis=-1)
                pen_counts = {int(v): int((pen_idx == v).sum()) for v in sorted(set(pen_idx.tolist()))}
            else:
                pen_counts = {}
            diagnostics[class_name].append({
                "sample": row + 1,
                "length": len(seq_np),
                "pen_counts": pen_counts,
            })
            print(f"  sample {row + 1}: length={len(seq_np)}, pen_counts={pen_counts}")

    plt.tight_layout()
    grid_path = out_dir / "results_grid_v2.png"
    plt.savefig(grid_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved grid: {grid_path}")

    # Save diagnostics
    diag_path = out_dir / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Saved diagnostics: {diag_path}")

    print("\nDone. Open results_grid_v2.png to see generated samples.")


if __name__ == "__main__":
    main()
