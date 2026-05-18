"""
extract_model.py

Slims final.pt (full training checkpoint with optimizer/scheduler/history/EMA)
into a small deployment file containing ONLY what's needed for inference:
  - EMA model weights (in FP16 to halve size)
  - Architecture config
  - class_to_idx mapping

Usage:
    python extract_model.py

Output:
    deploy/model_for_deploy.pt   (~5-15 MB, vs ~30-80 MB original)
"""

import os
import sys
import torch
from pathlib import Path

SOURCE_CHECKPOINT = "./outputs_v2/final.pt"         # adjust if yours is elsewhere
OUTPUT_DIR = "./deploy"
OUTPUT_FILE = "model_for_deploy.pt"


def main():
    src = Path(SOURCE_CHECKPOINT)
    if not src.exists():
        print(f"ERROR: {src} not found")
        # Try fallbacks
        for fallback in ["./outputs_v2_real/best.pt", "./outputs_v2_real/latest.pt"]:
            if Path(fallback).exists():
                print(f"  Using {fallback} instead")
                src = Path(fallback)
                break
        else:
            sys.exit(1)

    print(f"Loading: {src}")
    src_size_mb = src.stat().st_size / (1024 * 1024)
    print(f"Source size: {src_size_mb:.1f} MB")

    payload = torch.load(src, map_location="cpu", weights_only=False)

    # Pick weights — prefer EMA (used during training validation, generally better)
    if payload.get("ema_state") is not None:
        weights = payload["ema_state"]
        weight_source = "ema"
    else:
        weights = payload["model_state"]
        weight_source = "model"
    print(f"Using weights from: {weight_source}")

    # Convert to FP16 to halve size (still plenty precise for inference)
    weights_fp16 = {}
    for k, v in weights.items():
        if v.dtype == torch.float32:
            weights_fp16[k] = v.half()
        else:
            weights_fp16[k] = v

    # Get config from saved checkpoint
    cfg = payload.get("config", {})
    if not cfg:
        print("WARNING: no config in checkpoint; using defaults")
        cfg = {
            "classes": ["apple", "circle", "star", "triangle"],
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

    class_to_idx = payload.get("class_to_idx", {c: i for i, c in enumerate(cfg["classes"])})

    # Strip cfg to only architecture-relevant keys
    deploy_cfg = {
        "classes": cfg.get("classes", ["apple", "circle", "star", "triangle"]),
        "max_seq_len": cfg.get("max_seq_len", 80),
        "encoder_hidden_dim": cfg.get("encoder_hidden_dim", 192),
        "encoder_layers": cfg.get("encoder_layers", 1),
        "latent_dim": cfg.get("latent_dim", 128),
        "d_model": cfg.get("d_model", 192),
        "num_heads": cfg.get("num_heads", 6),
        "decoder_layers": cfg.get("decoder_layers", 4),
        "ff_dim": cfg.get("ff_dim", 384),
        "class_embed_dim": cfg.get("class_embed_dim", 128),
        "dropout": cfg.get("dropout", 0.12),
        "num_mixtures": cfg.get("num_mixtures", 16),
    }

    deploy_payload = {
        "weights": weights_fp16,
        "config": deploy_cfg,
        "class_to_idx": class_to_idx,
        "weight_source": weight_source,
    }

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_FILE
    torch.save(deploy_payload, out_path)

    out_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {out_path}")
    print(f"Output size: {out_size_mb:.1f} MB (was {src_size_mb:.1f} MB)")
    print(f"Reduction: {(1 - out_size_mb/src_size_mb)*100:.1f}%")

    if out_size_mb > 95:
        print(f"\nWARNING: Output is {out_size_mb:.1f} MB — close to GitHub's 100 MB hard limit.")
        print(f"You may need Git LFS to push to GitHub.")
    elif out_size_mb > 50:
        print(f"\nNOTE: Output is {out_size_mb:.1f} MB — over GitHub's 50 MB warning threshold.")
        print(f"Pushing will work but you'll get a warning.")
    else:
        print(f"\nOutput size is fine for direct GitHub push.")


if __name__ == "__main__":
    main()
