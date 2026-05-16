"""
Sketch Generation Model — Improved Version (v2)

Key changes from v1:
  1. RDP stroke simplification (Ramer-Douglas-Peucker) — same as original Sketch-RNN paper.
     Reduces sequence length and removes noise. Model learns shape, not redundant points.
  2. Pen loss weighting — fixes the scale imbalance between coord (NLL ~3) and pen (CE ~0.2).
  3. Per-class pen accuracy — global accuracy hides minority-class failure (end-of-sketch).
  4. Autoregressive validation — true measure of generation quality (not teacher-forced).
  5. Free bits KL — prevents posterior collapse without manual beta tuning.
  6. Per-epoch sample generation — visual progress tracking.
  7. Cleaner MDN sampling — removed double-temperature bug, removed step clamping band-aid.
  8. Min-delta early stopping — stop when improvements are noise.
  9. Encoder-based generation scoring — uses model's own learned representation.
 10. Bumped data and capacity — biggest quality lever.

Author: Prachi Sahu | M.Tech, IIT Jodhpur
"""

import os
import gc
import json
import math
import copy
import random
from pathlib import Path
from dataclasses import dataclass
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "seed": 42,
    "data_dir": "./data",
    "output_dir": "./outputs_v2",

    "classes": ["apple", "circle", "star", "triangle"],

    # DATA — more data is the #1 quality lever
    "max_drawings_per_class": 8000,    # was 3000
    "max_seq_len": 100,                # was 160 (RDP makes sequences shorter)
    "min_seq_len": 10,
    "rdp_epsilon": 2.0,                # NEW — stroke simplification threshold

    "train_ratio": 0.70,
    "val_ratio": 0.15,

    # TRAINING
    "batch_size_per_gpu": 16,          # was 4 — more stable gradients
    "num_workers": 4,
    "epochs": 60,                      # was 40
    "learning_rate": 5e-4,
    "weight_decay": 5e-5,

    # ENCODER
    "encoder_hidden_dim": 256,         # was 128
    "encoder_layers": 1,
    "latent_dim": 128,

    # DECODER (Transformer)
    "d_model": 256,                    # was 192
    "num_heads": 8,                    # was 4
    "decoder_layers": 6,               # was 4
    "ff_dim": 512,                     # was 384
    "class_embed_dim": 128,
    "dropout": 0.1,

    # MDN
    "num_mixtures": 20,                # was 12 — original Sketch-RNN used 20

    # VAE — free bits replaces aggressive KL annealing
    "beta": 1.0,                       # was 0.001 (now safe with free bits)
    "free_bits": 0.5,                  # NEW — nats per latent dim
    "kl_warmup_epochs": 8,             # shorter warmup since free bits handles collapse

    # LOSS BALANCING — NEW
    "coord_loss_weight": 1.0,
    "pen_loss_weight": 2.0,            # pen errors are visually catastrophic

    # TEACHER FORCING
    "teacher_forcing_start": 1.0,
    "teacher_forcing_end": 0.7,        # was 0.75

    # OPTIMIZATION
    "grad_clip": 1.0,
    "patience": 8,
    "min_delta": 0.005,                # NEW — improvements smaller than this don't count

    "amp": True,
    "ema_decay": 0.999,

    # GENERATION
    "save_gifs": True,
    "samples_per_class": 3,
    "generation_candidates": 8,        # was 4
    "val_sample_every_n_epochs": 5,    # NEW — visual tracking during training
    "ar_eval_every_n_epochs": 5,       # NEW — autoregressive validation cadence

    "generation_defaults": {
        "temperature": 0.4,
        "min_steps": 15,
        "eos_bias": 0.005,
        "top_k_mixtures": 5,
        "pen_mode": "sample",
    },
    "generation_overrides": {
        "apple": {"temperature": 0.38, "min_steps": 18},
        "circle": {"temperature": 0.32, "min_steps": 22, "top_k_mixtures": 3},
        "star": {"temperature": 0.35, "min_steps": 22, "top_k_mixtures": 3},
        "triangle": {"temperature": 0.34, "min_steps": 18, "top_k_mixtures": 3},
    },
}

START_TOKEN = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


# =============================================================================
# DDP SETUP (unchanged from v1)
# =============================================================================
def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return distributed, rank, world_size, local_rank, device


DISTRIBUTED, RANK, WORLD_SIZE, LOCAL_RANK, DEVICE = setup_distributed()


def is_main():
    return RANK == 0


def barrier():
    if DISTRIBUTED:
        dist.barrier()


def cleanup():
    if DISTRIBUTED and dist.is_initialized():
        dist.destroy_process_group()


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_rank_seed(seed):
    s = seed + RANK
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_global_seed(CONFIG["seed"])

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

Path(CONFIG["data_dir"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

if is_main():
    print(f"Device: {DEVICE}")
    print(f"Distributed: {DISTRIBUTED} | world_size={WORLD_SIZE}")

gc.collect()


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class SequenceExample:
    sequence: torch.Tensor
    length: int
    class_name: str
    class_id: int
    scale: float
    center_x: float
    center_y: float


# =============================================================================
# IMPROVEMENT #1: RDP Stroke Simplification
# =============================================================================
# Why: Raw QuickDraw strokes have many redundant collinear points. A circle might
# have 80 raw points when 25 would describe the same shape. Forcing the model to
# predict these tiny redundant deltas (a) wastes capacity, (b) adds noise, and
# (c) makes sequences longer than needed.
#
# RDP recursively keeps points that are MORE than `epsilon` away from the line
# between their neighbors. Drops the rest. Same algorithm Sketch-RNN paper used.

def _rdp_simplify(points_2d, epsilon):
    """Recursive RDP on a (N, 2) numpy array."""
    if len(points_2d) < 3:
        return points_2d
    start, end = points_2d[0], points_2d[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-9:
        # degenerate: start == end
        dists = np.linalg.norm(points_2d - start, axis=1)
    else:
        line_unit = line_vec / line_len
        # perpendicular distance from each point to line
        vecs = points_2d - start
        cross = np.abs(vecs[:, 0] * line_unit[1] - vecs[:, 1] * line_unit[0])
        dists = cross

    max_idx = int(np.argmax(dists))
    max_dist = float(dists[max_idx])

    if max_dist > epsilon and 0 < max_idx < len(points_2d) - 1:
        left = _rdp_simplify(points_2d[:max_idx + 1], epsilon)
        right = _rdp_simplify(points_2d[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([start, end])


def simplify_drawing(drawing, epsilon=2.0):
    """Apply RDP per stroke. Preserves stroke boundaries."""
    if not isinstance(drawing, list):
        return drawing
    simplified = []
    for stroke in drawing:
        if not isinstance(stroke, list) or len(stroke) < 2:
            continue
        xs, ys = stroke[0], stroke[1]
        if not isinstance(xs, list) or len(xs) < 2 or len(xs) != len(ys):
            continue
        try:
            pts = np.array(list(zip(xs, ys)), dtype=np.float32)
        except Exception:
            continue
        simp = _rdp_simplify(pts, epsilon)
        if len(simp) >= 2:
            simplified.append([simp[:, 0].tolist(), simp[:, 1].tolist()])
    return simplified


# =============================================================================
# DATA LOADING
# =============================================================================
def get_local_class_file(class_name: str, data_dir: str):
    candidates = [
        Path(data_dir) / f"{class_name}.ndjson",
        Path(data_dir) / f"{class_name}.json",
        Path(data_dir) / class_name / f"{class_name}.ndjson",
        Path(data_dir) / class_name / "data.ndjson",
        Path(data_dir) / class_name / "data.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing file for class {class_name}")


def load_ndjson(file_path, max_drawings=None):
    drawings = []
    bad_lines = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if max_drawings is not None and len(drawings) >= max_drawings:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("recognized", False) and isinstance(obj.get("drawing"), list):
                    drawings.append(obj)
            except json.JSONDecodeError:
                bad_lines += 1
    if is_main():
        print(f"Loaded {len(drawings)} from {Path(file_path).name} | bad_lines={bad_lines}")
    return drawings


def drawing_to_absolute_points(drawing):
    if not drawing or not isinstance(drawing, list):
        return None
    points = []
    valid = False
    for stroke in drawing:
        if not isinstance(stroke, list) or len(stroke) < 2:
            continue
        xs, ys = stroke[0], stroke[1]
        if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) == 0 or len(xs) != len(ys):
            continue
        added = 0
        for x, y in zip(xs, ys):
            try:
                x = float(x); y = float(y)
            except Exception:
                continue
            points.append([x, y, 0])
            valid = True
            added += 1
        if added > 0:
            points[-1][2] = 1
    if not valid:
        return None
    points[-1][2] = 2
    return np.array(points, dtype=np.float32)


def normalize_absolute_points(points):
    xy = points[:, :2].copy()
    center = xy.mean(axis=0)
    xy -= center
    scale = max(np.abs(xy).max(), 1.0)
    xy /= scale
    out = points.copy()
    out[:, :2] = xy
    return out, float(scale), float(center[0]), float(center[1])


def absolute_points_to_stroke5(points, max_seq_len=None):
    if points is None or len(points) == 0:
        return None
    seq = []
    px, py = 0.0, 0.0
    for x, y, pen_code in points:
        dx, dy = float(x) - px, float(y) - py
        px, py = float(x), float(y)
        if int(pen_code) == 0:
            token = [dx, dy, 1.0, 0.0, 0.0]
        elif int(pen_code) == 1:
            token = [dx, dy, 0.0, 1.0, 0.0]
        else:
            token = [dx, dy, 0.0, 0.0, 1.0]
        seq.append(token)
        if max_seq_len is not None and len(seq) >= max_seq_len:
            seq[-1] = [seq[-1][0], seq[-1][1], 0.0, 0.0, 1.0]
            break
    if len(seq) == 0:
        return None
    if seq[-1][4] != 1.0:
        if max_seq_len is None or len(seq) < max_seq_len:
            seq.append([0.0, 0.0, 0.0, 0.0, 1.0])
        else:
            seq[-1] = [seq[-1][0], seq[-1][1], 0.0, 0.0, 1.0]
    return torch.tensor(seq, dtype=torch.float32)


def build_example_from_drawing(drawing, class_name, class_id, max_seq_len, rdp_epsilon):
    # Apply RDP simplification BEFORE conversion to absolute points
    drawing = simplify_drawing(drawing, epsilon=rdp_epsilon)
    points = drawing_to_absolute_points(drawing)
    if points is None or len(points) == 0:
        return None
    points, scale, cx, cy = normalize_absolute_points(points)
    seq = absolute_points_to_stroke5(points, max_seq_len=max_seq_len)
    if seq is None:
        return None
    return SequenceExample(seq, len(seq), class_name, class_id, scale, cx, cy)


def split_list(items, train_ratio=0.70, val_ratio=0.15):
    n = len(items)
    a = int(n * train_ratio)
    b = a + int(n * val_ratio)
    return items[:a], items[a:b], items[b:]


# =============================================================================
# DATASET BUILD
# =============================================================================
if is_main():
    print("Preparing dataset (with RDP simplification)...")

class_to_idx = {c: i for i, c in enumerate(CONFIG["classes"])}
idx_to_class = {i: c for c, i in class_to_idx.items()}

train_examples, val_examples, test_examples = [], [], []
sequence_lengths = []  # for diagnostics

for class_name in CONFIG["classes"]:
    file_path = get_local_class_file(class_name, CONFIG["data_dir"])
    drawings = load_ndjson(file_path, CONFIG["max_drawings_per_class"])
    examples = []
    skipped = 0
    for d in drawings:
        ex = build_example_from_drawing(
            d.get("drawing", []),
            class_name=class_name,
            class_id=class_to_idx[class_name],
            max_seq_len=CONFIG["max_seq_len"],
            rdp_epsilon=CONFIG["rdp_epsilon"],
        )
        if ex is None or ex.length < CONFIG["min_seq_len"]:
            skipped += 1
            continue
        examples.append(ex)
        sequence_lengths.append(ex.length)
    random.shuffle(examples)
    tr, va, te = split_list(examples, CONFIG["train_ratio"], CONFIG["val_ratio"])
    train_examples.extend(tr)
    val_examples.extend(va)
    test_examples.extend(te)
    if is_main():
        print(f"{class_name}: valid={len(examples)} skipped={skipped} train={len(tr)} val={len(va)} test={len(te)}")

if is_main():
    print(f"Total: train={len(train_examples)} val={len(val_examples)} test={len(test_examples)}")
    if sequence_lengths:
        sl = np.array(sequence_lengths)
        print(f"Sequence length stats post-RDP: mean={sl.mean():.1f}, median={np.median(sl):.0f}, p95={np.percentile(sl, 95):.0f}, max={sl.max()}")

set_rank_seed(CONFIG["seed"])


class QuickDrawStrokeDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "strokes": ex.sequence,
            "length": ex.length,
            "class_id": ex.class_id,
            "class_name": ex.class_name,
        }


def collate_fn(batch):
    strokes = [b["strokes"] for b in batch]
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    class_ids = torch.tensor([b["class_id"] for b in batch], dtype=torch.long)
    class_names = [b["class_name"] for b in batch]
    padded = pad_sequence(strokes, batch_first=True, padding_value=0.0)
    return {"strokes": padded, "lengths": lengths, "class_ids": class_ids, "class_names": class_names}


train_dataset = QuickDrawStrokeDataset(train_examples)
val_dataset = QuickDrawStrokeDataset(val_examples)
test_dataset = QuickDrawStrokeDataset(test_examples)

train_sampler = DistributedSampler(train_dataset, shuffle=True) if DISTRIBUTED else None
val_sampler = DistributedSampler(val_dataset, shuffle=False) if DISTRIBUTED else None
test_sampler = DistributedSampler(test_dataset, shuffle=False) if DISTRIBUTED else None

loader_kwargs = {
    "batch_size": CONFIG["batch_size_per_gpu"],
    "num_workers": CONFIG["num_workers"],
    "pin_memory": torch.cuda.is_available(),
    "persistent_workers": CONFIG["num_workers"] > 0,
    "collate_fn": collate_fn,
}

train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=(train_sampler is None), **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=val_sampler, shuffle=False, **loader_kwargs)
test_loader = DataLoader(test_dataset, sampler=test_sampler, shuffle=False, **loader_kwargs)


# =============================================================================
# MODEL — Encoder, VAE, Decoder, SketchModel
# =============================================================================
class Encoder(nn.Module):
    """BiLSTM encoder with class conditioning. Outputs mu and logvar of latent Gaussian."""
    def __init__(self, input_dim, hidden_dim, num_layers, latent_dim, num_classes, class_embed_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout,
            bidirectional=True,
        )
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
    """VAE reparameterization + KL divergence with FREE BITS."""
    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    @staticmethod
    def kl_divergence(mu, logvar):
        """Standard KL — kept for reference/comparison."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    @staticmethod
    def kl_divergence_free_bits(mu, logvar, free_bits=0.5):
        """
        IMPROVEMENT #5: Free bits prevent posterior collapse.

        Standard KL forces ALL latent dims toward N(0,1), so optimizer can collapse
        them to zero info. Free bits gives each dim `free_bits` nats "for free" —
        we only penalize KL ABOVE this threshold per dim. This guarantees each
        latent dim carries at least `free_bits` nats of useful information.
        """
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, D)
        kl_per_dim_avg = kl_per_dim.mean(dim=0)  # (D,)
        # Floor each dim at free_bits before summing
        kl_floored = torch.clamp(kl_per_dim_avg, min=free_bits)
        return kl_floored.sum()


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
    """Transformer decoder. Predicts (dx, dy) via MDN and pen state via classification."""
    def __init__(self, input_dim, d_model, num_heads, num_layers, ff_dim, dropout, latent_dim, num_classes, class_embed_dim, max_seq_len, num_mixtures):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.input_proj = nn.Linear(input_dim, d_model)
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        self.class_proj = nn.Linear(class_embed_dim, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.fuse = nn.Linear(d_model * 2, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_seq_len + 4)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        # MDN head: 6 params per mixture (pi, mu_x, mu_y, log_sx, log_sy, rho_raw)
        self.mdn_head = nn.Linear(d_model, num_mixtures * 6)
        self.pen_head = nn.Linear(d_model, 3)  # pen down / up / end

    def causal_mask(self, seq_len, device):
        m = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, decoder_inputs, class_ids, z):
        tok = self.input_proj(decoder_inputs)
        cls = self.class_proj(self.class_embedding(class_ids))
        ztok = self.latent_proj(z)
        cls_ctx = cls.unsqueeze(1).expand(-1, tok.size(1), -1)
        tok = self.fuse(torch.cat([tok, cls_ctx], dim=-1))

        # Prepend class and z as context tokens
        h = torch.cat([cls.unsqueeze(1), ztok.unsqueeze(1), tok], dim=1)
        h = self.pos(h)
        h = self.transformer(h, mask=self.causal_mask(h.size(1), h.device))
        h = self.norm(h)
        h = h[:, 2:, :]  # strip the two prepended context tokens
        return self.mdn_head(h), self.pen_head(h)


class SketchModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg["latent_dim"]
        self.num_mixtures = cfg["num_mixtures"]
        self.encoder = Encoder(5, cfg["encoder_hidden_dim"], cfg["encoder_layers"], cfg["latent_dim"], len(cfg["classes"]), cfg["class_embed_dim"], cfg["dropout"])
        self.vae = VAE()
        self.decoder = Decoder(5, cfg["d_model"], cfg["num_heads"], cfg["decoder_layers"], cfg["ff_dim"], cfg["dropout"], cfg["latent_dim"], len(cfg["classes"]), cfg["class_embed_dim"], cfg["max_seq_len"], cfg["num_mixtures"])
        self.register_buffer("start_token", START_TOKEN.clone(), persistent=False)

    def forward(self, strokes, lengths, class_ids, teacher_forcing_ratio=1.0):
        batch_size, seq_len, _ = strokes.shape
        mu, logvar = self.encoder(strokes, lengths, class_ids)
        z = self.vae(mu, logvar)
        history = self.start_token.view(1, 1, -1).expand(batch_size, 1, -1).clone()
        mdn_preds, pen_preds = [], []

        for step in range(seq_len):
            mdn_params, pen_logits = self.decoder(history, class_ids, z)
            next_mdn = mdn_params[:, -1, :]
            next_pen_logits = pen_logits[:, -1, :]
            mdn_preds.append(next_mdn.unsqueeze(1))
            pen_preds.append(next_pen_logits.unsqueeze(1))

            if self.training and step > 0 and teacher_forcing_ratio < 1.0:
                use_teacher = torch.rand(batch_size, device=strokes.device) < teacher_forcing_ratio
                pred_dxdy, pred_pen = mdn_expected_step(next_mdn, next_pen_logits, self.num_mixtures)
                pred_token = torch.cat([pred_dxdy.detach(), pred_pen.detach()], dim=-1).unsqueeze(1)
                teacher = strokes[:, step:step + 1, :]
                next_input = torch.where(use_teacher.view(batch_size, 1, 1), teacher, pred_token)
            else:
                next_input = strokes[:, step:step + 1, :]

            history = torch.cat([history, next_input], dim=1)

        return {
            "mdn_params": torch.cat(mdn_preds, dim=1),
            "pen_logits": torch.cat(pen_preds, dim=1),
            "mu": mu,
            "logvar": logvar,
        }

    @torch.no_grad()
    def generate_once(self, class_name, class_id, cfg, device):
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
                next_mdn,
                next_pen_logits,
                num_mixtures=self.num_mixtures,
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

    @torch.no_grad()
    def generate_best(self, class_name, class_id, cfg, device):
        """
        IMPROVEMENT #9: Combine heuristic geometry score with encoder-based score.
        Encoder score = how typical the generated sketch's latent encoding is.
        Well-formed sketches encode to high-density regions of the latent space.
        """
        best_seq = None
        best_score = -1e9
        for _ in range(cfg["generation_candidates"]):
            seq = self.generate_once(class_name, class_id, cfg, device)
            geom_score = score_generated_sequence(seq, class_name)
            enc_score = score_via_encoder(self, seq, class_id, device)
            combined = geom_score + 0.1 * enc_score
            if combined > best_score:
                best_score = combined
                best_seq = seq
        return best_seq


# =============================================================================
# EMA (unchanged)
# =============================================================================
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state):
        self.shadow.load_state_dict(state)


# =============================================================================
# MDN HELPERS (cleaned up)
# =============================================================================
def unpack_mdn_params(mdn_params, num_mixtures):
    pi_logits, mu_x, mu_y, log_sx, log_sy, rho_raw = torch.split(mdn_params, num_mixtures, dim=-1)
    sx = torch.exp(log_sx).clamp(min=1e-4, max=4.0)
    sy = torch.exp(log_sy).clamp(min=1e-4, max=4.0)
    rho = torch.tanh(rho_raw).clamp(min=-0.95, max=0.95)
    return pi_logits, mu_x, mu_y, sx, sy, rho


def bivariate_normal_nll(x, y, mdn_params, num_mixtures):
    """Negative log-likelihood under mixture of bivariate Gaussians.
    Uses logsumexp for numerical stability."""
    pi_logits, mu_x, mu_y, sx, sy, rho = unpack_mdn_params(mdn_params, num_mixtures)
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    nx = (x - mu_x) / sx
    ny = (y - mu_y) / sy
    z = nx ** 2 + ny ** 2 - 2 * rho * nx * ny
    one_minus_rho2 = 1 - rho ** 2 + 1e-6
    log_norm = -math.log(2 * math.pi) - torch.log(sx) - torch.log(sy) - 0.5 * torch.log(one_minus_rho2)
    log_kernel = -z / (2 * one_minus_rho2)
    log_pi = F.log_softmax(pi_logits, dim=-1)
    log_prob = log_norm + log_kernel + log_pi
    return -torch.logsumexp(log_prob, dim=-1)


def mdn_expected_step(mdn_params, pen_logits, num_mixtures):
    """Deterministic 'best' next step — used for teacher-forcing mixing."""
    pi_logits, mu_x, mu_y, _, _, _ = unpack_mdn_params(mdn_params, num_mixtures)
    mix_idx = torch.argmax(pi_logits, dim=-1)
    bi = torch.arange(mix_idx.size(0), device=mix_idx.device)
    dx = mu_x[bi, mix_idx]
    dy = mu_y[bi, mix_idx]
    pen_idx = torch.argmax(pen_logits, dim=-1)
    pen_onehot = F.one_hot(pen_idx, num_classes=3).float()
    return torch.stack([dx, dy], dim=-1), pen_onehot


def sample_mdn_step(mdn_params, pen_logits, num_mixtures, temperature, eos_bias, allow_eos, pen_mode, top_k_mixtures):
    """
    IMPROVEMENT #7: Cleaner sampling.

    Old code:
      - Applied temperature to BOTH mixture selection AND component sigma (double counting)
      - Hard-clamped step size, masking model errors instead of fixing them
    New:
      - Temperature only on categorical distributions (mixture choice + pen)
      - No step clamp — if model produces wild steps, that's a training problem to fix, not hide
    """
    pi_logits, mu_x, mu_y, sx, sy, rho = unpack_mdn_params(mdn_params, num_mixtures)

    # Top-k filter on mixture weights (focus on most confident components)
    if top_k_mixtures is not None and top_k_mixtures < num_mixtures:
        top_vals, top_idx = torch.topk(pi_logits, k=top_k_mixtures, dim=-1)
        masked = torch.full_like(pi_logits, float("-inf"))
        masked.scatter_(dim=-1, index=top_idx, src=top_vals)
        pi_logits = masked

    # Sample mixture component (temperature controls peakedness)
    pi = F.softmax(pi_logits / max(temperature, 1e-4), dim=-1)
    mix_idx = torch.distributions.Categorical(pi).sample()
    bi = torch.arange(mix_idx.size(0), device=mix_idx.device)

    mx = mu_x[bi, mix_idx]
    my = mu_y[bi, mix_idx]
    sx_sel = sx[bi, mix_idx]
    sy_sel = sy[bi, mix_idx]
    r = rho[bi, mix_idx]

    # Sample (dx, dy) from chosen bivariate Gaussian via Cholesky-style reparameterization
    eps_x = torch.randn_like(mx)
    eps_y = torch.randn_like(my)
    dx = mx + sx_sel * eps_x
    dy = my + sy_sel * (r * eps_x + torch.sqrt(1 - r ** 2 + 1e-6) * eps_y)

    # Pen sampling
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


def sequence_mask(lengths, max_len=None):
    max_len = max_len or int(lengths.max().item())
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


# =============================================================================
# LOSS — with weighted pen loss (IMPROVEMENT #2)
# =============================================================================
def sketch_loss(mdn_params, pen_logits, targets, lengths, num_mixtures, coord_weight=1.0, pen_weight=2.0):
    """
    IMPROVEMENT #2: Weighted combination.

    Coord NLL is typically ~3.0 in scale, pen CE is ~0.2. Adding directly means
    pen gets ~7% of total gradient signal. Pen errors are visually catastrophic
    (broken strokes, missing endings). Weighting pen higher fixes the imbalance.
    """
    target_dx = targets[:, :, 0]
    target_dy = targets[:, :, 1]
    target_pen = torch.argmax(targets[:, :, 2:5], dim=-1)
    mask = sequence_mask(lengths, targets.size(1)).float()

    coord_nll = bivariate_normal_nll(target_dx, target_dy, mdn_params, num_mixtures)
    coord_loss = (coord_nll * mask).sum() / mask.sum().clamp_min(1.0)

    flat_logits = pen_logits.reshape(-1, 3)
    flat_targets = target_pen.reshape(-1)
    flat_mask = mask.reshape(-1).bool()

    # Internal class weights: end-of-sketch is rare, weight it higher
    pen_class_weights = torch.tensor([1.0, 1.2, 2.0], device=targets.device)
    pen_loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask], weight=pen_class_weights)

    total = coord_weight * coord_loss + pen_weight * pen_loss
    return coord_loss, pen_loss, total


def current_kl_weight(epoch, cfg):
    scale = min(1.0, float(epoch) / float(max(1, cfg["kl_warmup_epochs"])))
    return cfg["beta"] * scale


def current_teacher_forcing_ratio(epoch, cfg):
    if cfg["epochs"] <= 1:
        return cfg["teacher_forcing_end"]
    alpha = (epoch - 1) / (cfg["epochs"] - 1)
    return cfg["teacher_forcing_start"] + alpha * (cfg["teacher_forcing_end"] - cfg["teacher_forcing_start"])


def get_model_ref(model):
    return model.module if hasattr(model, "module") else model


def move_batch(batch):
    return {
        "strokes": batch["strokes"].to(DEVICE, non_blocking=True),
        "lengths": batch["lengths"].to(DEVICE, non_blocking=True),
        "class_ids": batch["class_ids"].to(DEVICE, non_blocking=True),
        "class_names": batch["class_names"],
    }


def reduce_sum(x):
    if DISTRIBUTED:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


# =============================================================================
# EPOCH RUNNER — with per-class pen accuracy (IMPROVEMENT #3)
# =============================================================================
def run_epoch(model, loader, cfg, epoch_idx, optimizer=None, scaler=None, ema=None):
    is_train = optimizer is not None
    model.train(is_train)

    if is_train and DISTRIBUTED and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch_idx)

    kl_beta = current_kl_weight(epoch_idx, cfg)
    tf_ratio = current_teacher_forcing_ratio(epoch_idx, cfg)
    use_amp = cfg["amp"] and DEVICE.type == "cuda"

    total_sum = torch.zeros(1, device=DEVICE)
    recon_sum = torch.zeros(1, device=DEVICE)
    coord_sum = torch.zeros(1, device=DEVICE)
    pen_sum = torch.zeros(1, device=DEVICE)
    kl_sum = torch.zeros(1, device=DEVICE)
    sample_count = torch.zeros(1, device=DEVICE)

    # Per-class tracking for pen accuracy (IMPROVEMENT #3)
    pen_correct_per_class = torch.zeros(3, device=DEVICE)
    pen_total_per_class = torch.zeros(3, device=DEVICE)

    pbar = tqdm(loader, leave=False) if is_main() else loader

    for batch in pbar:
        batch = move_batch(batch)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp) if use_amp else nullcontext()
        with torch.set_grad_enabled(is_train):
            with autocast_ctx:
                out = model(batch["strokes"], batch["lengths"], batch["class_ids"], tf_ratio)
                coord_loss, pen_loss, recon_loss = sketch_loss(
                    out["mdn_params"], out["pen_logits"],
                    batch["strokes"], batch["lengths"], cfg["num_mixtures"],
                    coord_weight=cfg["coord_loss_weight"],
                    pen_weight=cfg["pen_loss_weight"],
                )
                # Use free-bits KL (IMPROVEMENT #5)
                kl_loss = get_model_ref(model).vae.kl_divergence_free_bits(
                    out["mu"], out["logvar"], free_bits=cfg["free_bits"]
                )
                loss = recon_loss + kl_beta * kl_loss

            if optimizer is not None:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    optimizer.step()
                if ema is not None:
                    ema.update(get_model_ref(model))

        bs = batch["strokes"].size(0)
        total_sum += loss.detach() * bs
        recon_sum += recon_loss.detach() * bs
        coord_sum += coord_loss.detach() * bs
        pen_sum += pen_loss.detach() * bs
        kl_sum += kl_loss.detach() * bs
        sample_count += bs

        # Per-class pen accuracy
        target_pen = torch.argmax(batch["strokes"][:, :, 2:5], dim=-1)
        mask = sequence_mask(batch["lengths"], batch["strokes"].size(1))
        pred_pen = out["pen_logits"].argmax(dim=-1)
        for c in range(3):
            class_mask = (target_pen == c) & mask
            pen_correct_per_class[c] += ((pred_pen == target_pen) & class_mask).sum()
            pen_total_per_class[c] += class_mask.sum()

        if is_main():
            pbar.set_postfix(
                loss=f"{(total_sum / sample_count.clamp_min(1)).item():.4f}",
                beta=f"{kl_beta:.4f}",
                tf=f"{tf_ratio:.2f}",
            )

    for t in [total_sum, recon_sum, coord_sum, pen_sum, kl_sum, sample_count]:
        reduce_sum(t)
    for t in [pen_correct_per_class, pen_total_per_class]:
        reduce_sum(t)

    pen_acc_overall = (pen_correct_per_class.sum() / pen_total_per_class.sum().clamp_min(1)).item()

    return {
        "loss": (total_sum / sample_count.clamp_min(1)).item(),
        "recon_loss": (recon_sum / sample_count.clamp_min(1)).item(),
        "coord_loss": (coord_sum / sample_count.clamp_min(1)).item(),
        "pen_loss": (pen_sum / sample_count.clamp_min(1)).item(),
        "kl_loss": (kl_sum / sample_count.clamp_min(1)).item(),
        "pen_acc": pen_acc_overall,
        "pen_acc_down": (pen_correct_per_class[0] / pen_total_per_class[0].clamp_min(1)).item(),
        "pen_acc_up": (pen_correct_per_class[1] / pen_total_per_class[1].clamp_min(1)).item(),
        "pen_acc_end": (pen_correct_per_class[2] / pen_total_per_class[2].clamp_min(1)).item(),
        "kl_beta": kl_beta,
        "teacher_forcing_ratio": tf_ratio,
    }


# =============================================================================
# IMPROVEMENT #4: AUTOREGRESSIVE VALIDATION
# =============================================================================
@torch.no_grad()
def autoregressive_validation(model_ref, loader, cfg, device, max_batches=8):
    """
    True generation-quality validation. Standard val uses 100% teacher forcing,
    which inflates accuracy because the model always sees ground-truth history.
    Real inference is autoregressive — model sees its own predictions.

    The gap between teacher-forced val and autoregressive val measures EXPOSURE BIAS.
    """
    model_ref.eval()
    total_coord_nll = 0.0
    total_pen_correct = 0
    total_count = 0
    n_batches = 0

    for batch in loader:
        if n_batches >= max_batches:
            break
        batch = move_batch(batch)

        # Use posterior mean (deterministic) for fair eval
        mu, _ = model_ref.encoder(batch["strokes"], batch["lengths"], batch["class_ids"])
        z = mu

        batch_size = batch["strokes"].size(0)
        seq_len = batch["strokes"].size(1)
        history = model_ref.start_token.view(1, 1, -1).expand(batch_size, 1, -1).clone().to(device)

        for step in range(seq_len):
            mdn_params, pen_logits = model_ref.decoder(history, batch["class_ids"], z)
            next_mdn = mdn_params[:, -1, :]
            next_pen_logits = pen_logits[:, -1, :]

            target = batch["strokes"][:, step, :]
            target_dx = target[:, 0]
            target_dy = target[:, 1]
            target_pen = torch.argmax(target[:, 2:5], dim=-1)

            step_mask = step < batch["lengths"]
            if step_mask.sum() == 0:
                break

            # Coord NLL on this step
            coord_nll = bivariate_normal_nll(
                target_dx.unsqueeze(1), target_dy.unsqueeze(1),
                next_mdn.unsqueeze(1), cfg["num_mixtures"]
            ).squeeze(1)
            total_coord_nll += (coord_nll * step_mask.float()).sum().item()

            # Pen accuracy
            pred_pen = next_pen_logits.argmax(dim=-1)
            total_pen_correct += ((pred_pen == target_pen) & step_mask).sum().item()
            total_count += step_mask.sum().item()

            # CRITICAL: feed model's OWN prediction back, not ground truth
            pred_dxdy, pred_pen_oh = mdn_expected_step(next_mdn, next_pen_logits, cfg["num_mixtures"])
            next_input = torch.cat([pred_dxdy, pred_pen_oh], dim=-1).unsqueeze(1)
            history = torch.cat([history, next_input], dim=1)

        n_batches += 1

    if total_count == 0:
        return {"ar_coord_nll": float("nan"), "ar_pen_acc": float("nan")}

    return {
        "ar_coord_nll": total_coord_nll / total_count,
        "ar_pen_acc": total_pen_correct / total_count,
    }


# =============================================================================
# RENDERING & GEOMETRY (mostly unchanged)
# =============================================================================
def stroke5_to_absolute_points(sequence):
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.detach().cpu().float().numpy()
    pts = []
    x, y = 0.0, 0.0
    for token in sequence:
        dx, dy = float(token[0]), float(token[1])
        pen_idx = int(np.argmax(token[2:5]))
        x += dx
        y += dy
        pts.append([x, y, pen_idx])
        if pen_idx == 2:
            break
    return np.asarray(pts, dtype=np.float32)


def geometry_features(sequence):
    pts = stroke5_to_absolute_points(sequence)
    pts = pts[pts[:, 2] != 2] if len(pts) > 0 else pts
    if len(pts) < 3:
        return {"length": len(pts), "closure": 1.0, "roundness": -1.0, "corner_count": 0}
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
        segs = segs[valid]
        ang = np.unwrap(np.arctan2(segs[:, 1], segs[:, 0]))
        dang = np.abs(np.diff(ang))
        corner_count = int((dang > 0.45).sum())
    return {"length": len(xy), "closure": float(closure), "roundness": float(roundness), "corner_count": int(corner_count)}


def score_generated_sequence(sequence, class_name):
    feat = geometry_features(sequence)
    n = feat["length"]
    closure = feat["closure"]
    roundness = feat["roundness"]
    corners = feat["corner_count"]
    if class_name == "circle":
        return 3.0 * roundness - 2.0 * closure - 0.03 * abs(n - 25)
    if class_name == "triangle":
        return -2.5 * closure - 0.35 * abs(corners - 3) - 0.02 * abs(n - 18)
    if class_name == "star":
        return -2.0 * closure - 0.20 * abs(corners - 5) - 0.02 * abs(n - 22)
    if class_name == "apple":
        return -2.0 * closure - 0.02 * abs(n - 22) + 0.2 * max(roundness, 0.0)
    return -closure


@torch.no_grad()
def score_via_encoder(model_ref, sequence, class_id, device):
    """
    IMPROVEMENT #9: Use encoder as a quality judge.
    A well-formed sketch should encode to a 'typical' point near the prior N(0,I).
    Higher score = more typical = more likely to be well-formed.
    """
    if len(sequence) < 3:
        return -1e9
    seq = sequence.unsqueeze(0).to(device).float()
    length = torch.tensor([len(sequence)], dtype=torch.long, device=device)
    cls = torch.tensor([class_id], dtype=torch.long, device=device)
    try:
        mu, _ = model_ref.encoder(seq, length, cls)
        return -mu.pow(2).sum(dim=-1).item()
    except Exception:
        return -1e9


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

    ani = animation.FuncAnimation(fig, update, frames=len(coords), init_func=init, interval=interval, blit=True, repeat=False)
    ani.save(save_path, writer="pillow")
    plt.close(fig)


# =============================================================================
# IMPROVEMENT #6: VALIDATION SAMPLE GENERATION (during training)
# =============================================================================
def generate_validation_samples(model_ref, cfg, device, epoch, output_dir):
    """Save generated samples each epoch so we can VISUALLY track progress.
    Without this, you only see loss curves and miss generation quality issues."""
    samples_dir = Path(output_dir) / "training_samples"
    samples_dir.mkdir(exist_ok=True)
    model_ref.eval()

    n_classes = len(cfg["classes"])
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]

    for col, class_name in enumerate(cfg["classes"]):
        try:
            seq = model_ref.generate_once(class_name, class_to_idx[class_name], cfg, device)
            render_sketch(seq.cpu(), axes[col], title=f"{class_name} @ ep{epoch}")
        except Exception as e:
            axes[col].set_title(f"FAIL: {str(e)[:30]}")
            axes[col].axis("off")

    plt.tight_layout()
    plt.savefig(samples_dir / f"epoch_{epoch:03d}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CHECKPOINT I/O
# =============================================================================
def get_paths(cfg):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "latest": out_dir / "latest.pt",
        "best": out_dir / "best.pt",
        "final": out_dir / "final.pt",
        "history": out_dir / "history.json",
        "summary": out_dir / "summary.json",
        "grid": out_dir / "results_grid.png",
    }


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_ckpt(path, model, ema, optimizer, scheduler, scaler, epoch, history, best_val, patience, cfg):
    payload = {
        "epoch": epoch,
        "model_state": get_model_ref(model).state_dict(),
        "ema_state": ema.state_dict() if ema is not None else None,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "history": history,
        "best_val_loss": best_val,
        "patience_counter": patience,
        "config": cfg,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }
    torch.save(payload, path)


def load_ckpt(path, model, ema=None, optimizer=None, scheduler=None, scaler=None):
    if not path.exists():
        return 1, {
            "train_loss": [], "val_loss": [], "val_pen_acc": [],
            "val_pen_acc_down": [], "val_pen_acc_up": [], "val_pen_acc_end": [],
            "ar_coord_nll": [], "ar_pen_acc": [],
        }, float("inf"), 0
    payload = torch.load(path, map_location="cpu")
    get_model_ref(model).load_state_dict(payload["model_state"])
    if ema is not None and payload.get("ema_state") is not None:
        ema.load_state_dict(payload["ema_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    history = payload["history"]
    # Backward-compat: ensure new keys exist
    for k in ["val_pen_acc_down", "val_pen_acc_up", "val_pen_acc_end", "ar_coord_nll", "ar_pen_acc"]:
        if k not in history:
            history[k] = []
    return (
        payload["epoch"] + 1,
        history,
        payload.get("best_val_loss", float("inf")),
        payload.get("patience_counter", 0),
    )


# =============================================================================
# MODEL INSTANTIATION
# =============================================================================
model = SketchModel(CONFIG).to(DEVICE)
if DISTRIBUTED:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,
        broadcast_buffers=False, find_unused_parameters=False,
    )

if is_main():
    n_params = sum(p.numel() for p in get_model_ref(model).parameters())
    n_trainable = sum(p.numel() for p in get_model_ref(model).parameters() if p.requires_grad)
    print(f"Model: {n_params/1e6:.2f}M params total, {n_trainable/1e6:.2f}M trainable")

ema = EMA(get_model_ref(model), CONFIG["ema_decay"])
optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["amp"] and DEVICE.type == "cuda")
paths = get_paths(CONFIG)


# =============================================================================
# MAIN TRAINING LOOP — wrapped in main() for proper DDP entry point
# =============================================================================
def train():
    global model, ema, optimizer, scheduler, scaler

    if paths["final"].exists():
        if is_main():
            print("Final checkpoint exists. Loading and skipping training.")
        start_epoch, history, best_val_loss, patience_counter = load_ckpt(paths["final"], model, ema)
        return history, best_val_loss

    start_epoch, history, best_val_loss, patience_counter = load_ckpt(
        paths["latest"], model, ema, optimizer, scheduler, scaler
    )
    if is_main() and start_epoch > 1:
        print(f"Resuming from epoch {start_epoch - 1}")

    barrier()

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, CONFIG, epoch, optimizer=optimizer, scaler=scaler, ema=ema)
        eval_model = ema.shadow.to(DEVICE)
        val_metrics = run_epoch(eval_model, val_loader, CONFIG, epoch, optimizer=None, scaler=None, ema=None)
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_pen_acc"].append(val_metrics["pen_acc"])
        history["val_pen_acc_down"].append(val_metrics["pen_acc_down"])
        history["val_pen_acc_up"].append(val_metrics["pen_acc_up"])
        history["val_pen_acc_end"].append(val_metrics["pen_acc_end"])

        # IMPROVEMENT #4: Run AR validation periodically (it's slow)
        if is_main() and (epoch % CONFIG["ar_eval_every_n_epochs"] == 0 or epoch == CONFIG["epochs"]):
            ar_metrics = autoregressive_validation(eval_model, val_loader, CONFIG, DEVICE, max_batches=8)
            history["ar_coord_nll"].append({"epoch": epoch, **ar_metrics})
            history["ar_pen_acc"].append({"epoch": epoch, "pen_acc": ar_metrics["ar_pen_acc"]})
            print(f"  [AR-VAL] coord_nll={ar_metrics['ar_coord_nll']:.4f} | pen_acc={ar_metrics['ar_pen_acc']:.4f}")

        # IMPROVEMENT #6: Save generated samples for visual tracking
        if is_main() and epoch % CONFIG["val_sample_every_n_epochs"] == 0:
            generate_validation_samples(eval_model, CONFIG, DEVICE, epoch, CONFIG["output_dir"])

        # IMPROVEMENT #8: Min-delta early stopping
        improved = val_metrics["loss"] < (best_val_loss - CONFIG["min_delta"])
        if improved:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
        else:
            patience_counter += 1

        if is_main():
            print(
                f"Epoch {epoch:02d} | train={train_metrics['loss']:.4f} | val={val_metrics['loss']:.4f} | "
                f"pen_acc={val_metrics['pen_acc']:.4f} (down={val_metrics['pen_acc_down']:.3f}, "
                f"up={val_metrics['pen_acc_up']:.3f}, end={val_metrics['pen_acc_end']:.3f}) | "
                f"beta={train_metrics['kl_beta']:.4f} | tf={train_metrics['teacher_forcing_ratio']:.2f} | "
                f"lr={optimizer.param_groups[0]['lr']:.6f} | patience={patience_counter}/{CONFIG['patience']}"
            )
            save_ckpt(paths["latest"], model, ema, optimizer, scheduler, scaler, epoch, history, best_val_loss, patience_counter, CONFIG)
            save_json(paths["history"], history)
            if improved:
                save_ckpt(paths["best"], model, ema, optimizer, scheduler, scaler, epoch, history, best_val_loss, patience_counter, CONFIG)

        barrier()

        if patience_counter >= CONFIG["patience"]:
            if is_main():
                print(f"Early stopping at epoch {epoch} (no improvement for {CONFIG['patience']} epochs).")
            break

    barrier()

    # Reload best for final eval
    best_path = paths["best"] if paths["best"].exists() else paths["latest"]
    _, history, best_val_loss, patience_counter = load_ckpt(best_path, model, ema, optimizer, scheduler, scaler)
    test_metrics = run_epoch(ema.shadow.to(DEVICE), test_loader, CONFIG, CONFIG["epochs"], optimizer=None, scaler=None, ema=None)

    if is_main():
        print("\nTest metrics:", test_metrics)
        # Final autoregressive eval on test set
        ar_test = autoregressive_validation(ema.shadow.to(DEVICE), test_loader, CONFIG, DEVICE, max_batches=20)
        print(f"AR Test: coord_nll={ar_test['ar_coord_nll']:.4f} | pen_acc={ar_test['ar_pen_acc']:.4f}")

        save_ckpt(paths["final"], model, ema, optimizer, scheduler, scaler, len(history["train_loss"]), history, best_val_loss, patience_counter, CONFIG)
        save_json(paths["summary"], {
            "best_val_loss": best_val_loss,
            "test_metrics": test_metrics,
            "ar_test_metrics": ar_test,
            "epochs_ran": len(history["train_loss"]),
            "config": CONFIG,
        })

    return history, best_val_loss


def generate_final_grid():
    """Generate the final visualization grid with real vs generated samples."""
    if not is_main():
        return

    num_classes = len(CONFIG["classes"])
    rows = 1 + CONFIG["samples_per_class"]
    fig, axes = plt.subplots(rows, num_classes, figsize=(4 * num_classes, 3.7 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_classes == 1:
        axes = np.expand_dims(axes, axis=1)

    diagnostics = {}
    ema_model = ema.shadow.to(DEVICE).eval()

    for col, class_name in enumerate(CONFIG["classes"]):
        real_candidates = [ex for ex in test_examples if ex.class_name == class_name]
        real_example = random.choice(real_candidates) if real_candidates else None
        if real_example is not None:
            render_sketch(real_example.sequence, axes[0, col], title=f"Real: {class_name}")
        else:
            axes[0, col].axis("off")
            axes[0, col].set_title(f"Real: {class_name}")

        diagnostics[class_name] = []
        for row in range(CONFIG["samples_per_class"]):
            gen = ema_model.generate_best(class_name, class_to_idx[class_name], CONFIG, DEVICE).cpu()
            render_sketch(gen, axes[row + 1, col], title=f"{class_name} sample {row + 1}")

            gif_path = Path(CONFIG["output_dir"]) / f"{class_name}_sample_{row + 1}.gif"
            if CONFIG["save_gifs"]:
                animate_sketch(gen, gif_path, title=f"{class_name} sample {row + 1}")

            gen_np = gen.numpy()
            pen_idx = np.argmax(gen_np[:, 2:5], axis=-1) if len(gen_np) > 0 else np.array([])
            unique_vals = sorted(set(pen_idx.tolist())) if len(gen_np) > 0 else []
            pen_counts = {int(v): int((pen_idx == v).sum()) for v in unique_vals}
            diagnostics[class_name].append({
                "length": len(gen_np),
                "pen_counts": pen_counts,
                "gif_path": str(gif_path) if CONFIG["save_gifs"] else "",
            })

    plt.tight_layout()
    plt.savefig(paths["grid"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved grid: {paths['grid']}")
    for class_name, samples in diagnostics.items():
        print(f"\nClass: {class_name}")
        for i, s in enumerate(samples, 1):
            print(f"  sample {i}: length={s['length']}, pen_counts={s['pen_counts']}")


def plot_loss_curves(history):
    if not is_main() or not history.get("train_loss"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val (TF)")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["val_pen_acc_down"], label="pen down")
    axes[1].plot(history["val_pen_acc_up"], label="pen up")
    axes[1].plot(history["val_pen_acc_end"], label="pen end")
    axes[1].set_title("Per-class pen accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(Path(CONFIG["output_dir"]) / "loss_curves.png", dpi=160, bbox_inches="tight")
    plt.close()


def main():
    try:
        if is_main():
            print("\n=== TRAINING (v2) ===")
            print(f"World Size: {WORLD_SIZE}, Rank: {RANK}, Local Rank: {LOCAL_RANK}")

        history, best_val_loss = train()

        # Final test eval (re-run to get on test set for fresh metrics)
        barrier()
        test_metrics = run_epoch(ema.shadow.to(DEVICE), test_loader, CONFIG, CONFIG["epochs"],
                                  optimizer=None, scaler=None, ema=None)
        if is_main():
            print("\nFinal test metrics:", test_metrics)
            generate_final_grid()
            plot_loss_curves(history)

    except Exception as e:
        print(f"\nERROR in process {RANK}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
