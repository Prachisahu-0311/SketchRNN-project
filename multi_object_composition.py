"""
multi_object_composition.py

Extension #2: Multi-object composition for the sketch generation model.

Takes a list of object class names, generates each individually using the trained
model, then composes them onto a single canvas using one of several layout
strategies.

This module is import-safe: it depends on the SketchModel class and generation
helpers from app.py but does NOT re-import them at module load. The composer
functions accept the model and helpers as arguments.
"""

import math
import random
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Data types
# =============================================================================
@dataclass
class PlacedObject:
    """One object placed onto the scene canvas."""
    class_name: str
    sequence: np.ndarray   # the stroke-5 sequence (numpy, on CPU)
    bbox: tuple            # (xmin, ymin, xmax, ymax) of the sketch in its own coords
    center: tuple          # (cx, cy) target placement on the scene canvas
    scale: float           # scaling factor applied during render


# =============================================================================
# Geometry helpers
# =============================================================================
def stroke5_to_points(sequence):
    """Convert stroke-5 to absolute (x, y, pen_state) points."""
    if hasattr(sequence, "detach"):
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


def compute_bbox(sequence):
    """Bounding box of a sketch's drawn points (ignoring pen-end token)."""
    pts = stroke5_to_points(sequence)
    if len(pts) < 1:
        return (0.0, 0.0, 1.0, 1.0)
    drawn = pts[pts[:, 2] != 2]
    if len(drawn) < 1:
        drawn = pts
    xs = drawn[:, 0]
    ys = drawn[:, 1]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


# =============================================================================
# Layout strategies
# =============================================================================
def layout_horizontal_row(n_objects, canvas_size=10.0, margin=0.3):
    """Place objects in a single horizontal row, evenly spaced."""
    if n_objects <= 0:
        return []
    usable = canvas_size * (1.0 - 2 * margin)
    spacing = usable / n_objects
    centers = []
    for i in range(n_objects):
        cx = -canvas_size / 2 + canvas_size * margin + spacing * (i + 0.5)
        cy = 0.0
        centers.append((cx, cy))
    return centers


def layout_grid(n_objects, canvas_size=10.0, margin=0.2):
    """Place objects in a roughly square grid."""
    if n_objects <= 0:
        return []
    cols = math.ceil(math.sqrt(n_objects))
    rows = math.ceil(n_objects / cols)
    usable = canvas_size * (1.0 - 2 * margin)
    x_spacing = usable / cols
    y_spacing = usable / rows
    centers = []
    for i in range(n_objects):
        r = i // cols
        c = i % cols
        cx = -canvas_size / 2 + canvas_size * margin + x_spacing * (c + 0.5)
        cy = canvas_size / 2 - canvas_size * margin - y_spacing * (r + 0.5)
        centers.append((cx, cy))
    return centers


def layout_random_non_overlapping(n_objects, canvas_size=10.0, min_dist=2.5, max_attempts=200):
    """Place objects randomly, retrying until no two centers are too close."""
    if n_objects <= 0:
        return []
    half = canvas_size / 2 - 1.0  # keep away from canvas edge
    centers = []
    for _ in range(n_objects):
        for attempt in range(max_attempts):
            cx = random.uniform(-half, half)
            cy = random.uniform(-half, half)
            ok = True
            for (px, py) in centers:
                if math.hypot(cx - px, cy - py) < min_dist:
                    ok = False
                    break
            if ok:
                centers.append((cx, cy))
                break
        else:
            # fall back to grid placement if random fails too many times
            return layout_grid(n_objects, canvas_size=canvas_size)
    return centers


def layout_circular(n_objects, canvas_size=10.0, radius_frac=0.35):
    """Arrange objects in a circle around the center."""
    if n_objects <= 0:
        return []
    if n_objects == 1:
        return [(0.0, 0.0)]
    radius = canvas_size * radius_frac
    centers = []
    for i in range(n_objects):
        angle = 2 * math.pi * i / n_objects - math.pi / 2  # start at top
        cx = radius * math.cos(angle)
        cy = radius * math.sin(angle)
        centers.append((cx, cy))
    return centers


LAYOUT_STRATEGIES = {
    "Horizontal row": layout_horizontal_row,
    "Grid": layout_grid,
    "Random (non-overlapping)": layout_random_non_overlapping,
    "Circular arrangement": layout_circular,
}


# =============================================================================
# Spatial relationship adjustments (extra points)
# =============================================================================
def apply_spatial_relation(centers_a, centers_b, relation, canvas_size=10.0):
    """Adjust the center of B relative to A based on a relation string.

    Returns adjusted (cx, cy) for B. centers_a is a single (cx, cy) tuple.
    Used for user-specified constraints like "apple LEFT_OF star".
    """
    ax, ay = centers_a
    offset = canvas_size * 0.25

    if relation == "LEFT_OF":
        return (ax - offset, ay)
    elif relation == "RIGHT_OF":
        return (ax + offset, ay)
    elif relation == "ABOVE":
        return (ax, ay + offset)
    elif relation == "BELOW":
        return (ax, ay - offset)
    else:
        return centers_b  # unknown relation, no change


# =============================================================================
# Composition pipeline
# =============================================================================
def compose_scene(
    class_names,
    generate_fn,
    layout="Grid",
    canvas_size=10.0,
    object_scale=1.4,
    n_candidates_per_object=4,
):
    """
    Main composition function.

    Args:
        class_names: list of strings, e.g. ["apple", "star", "triangle"].
                     Duplicates allowed.
        generate_fn: a callable (class_name) -> stroke5 numpy array of shape (T, 5).
                     This wraps best-of-N selection.
        layout: name of layout strategy (see LAYOUT_STRATEGIES).
        canvas_size: dimensions of the final scene canvas.
        object_scale: how large each object should appear on the canvas.
        n_candidates_per_object: how many candidates to generate per object
                                  (passed through to generate_fn implicitly).

    Returns:
        list of PlacedObject, ready to render.
    """
    n = len(class_names)
    if n == 0:
        return []

    # Step 1: generate each object's sketch
    sequences = []
    for class_name in class_names:
        seq = generate_fn(class_name)
        sequences.append(seq)

    # Step 2: compute each sketch's bbox in its own coordinates
    bboxes = [compute_bbox(seq) for seq in sequences]

    # Step 3: choose layout centers
    layout_fn = LAYOUT_STRATEGIES.get(layout, layout_grid)
    centers = layout_fn(n, canvas_size=canvas_size)

    # Step 4: pack into PlacedObject
    placed = []
    for i in range(n):
        placed.append(PlacedObject(
            class_name=class_names[i],
            sequence=sequences[i],
            bbox=bboxes[i],
            center=centers[i] if i < len(centers) else (0.0, 0.0),
            scale=object_scale,
        ))
    return placed


def render_scene(
    placed_objects,
    canvas_size=10.0,
    show_labels=True,
    figsize=(8, 8),
    line_color="black",
    line_width=2.0,
):
    """Render placed objects on a single matplotlib figure."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-canvas_size / 2, canvas_size / 2)
    ax.set_ylim(-canvas_size / 2, canvas_size / 2)
    ax.set_aspect("equal")
    ax.axis("off")

    for obj in placed_objects:
        pts = stroke5_to_points(obj.sequence)
        if len(pts) < 2:
            continue

        # Normalize: center the sketch on its bbox, scale, then translate to target center
        xmin, ymin, xmax, ymax = obj.bbox
        local_cx = (xmin + xmax) / 2
        local_cy = (ymin + ymax) / 2
        bbox_diag = max(xmax - xmin, ymax - ymin, 1e-6)
        scale = obj.scale / bbox_diag

        target_cx, target_cy = obj.center

        # Draw stroke by stroke
        xs, ys = [], []
        for x, y, pen_idx in pts:
            if pen_idx == 2:
                if len(xs) > 1:
                    ax.plot(xs, ys, color=line_color, linewidth=line_width)
                break
            # Transform: (local - local_center) * scale + target_center
            tx = (x - local_cx) * scale + target_cx
            ty = -(y - local_cy) * scale + target_cy  # flip Y for screen coords
            xs.append(tx)
            ys.append(ty)
            if pen_idx == 1:  # pen lift
                if len(xs) > 1:
                    ax.plot(xs, ys, color=line_color, linewidth=line_width)
                xs, ys = [], []
        if len(xs) > 1:
            ax.plot(xs, ys, color=line_color, linewidth=line_width)

        # Optional label
        if show_labels:
            bbox_world_size = obj.scale / 2 + 0.3
            ax.text(
                target_cx, target_cy - bbox_world_size,
                obj.class_name,
                ha="center", va="top", fontsize=9, color="gray",
            )

    plt.tight_layout()
    return fig


def describe_scene(placed_objects):
    """Generate a natural-language description of the composed scene."""
    if not placed_objects:
        return "Empty scene."

    # Count objects by class
    counts = {}
    for obj in placed_objects:
        counts[obj.class_name] = counts.get(obj.class_name, 0) + 1

    parts = []
    for cls, count in counts.items():
        if count == 1:
            parts.append(f"1 {cls}")
        else:
            parts.append(f"{count} {cls}s")

    if len(parts) == 1:
        items = parts[0]
    elif len(parts) == 2:
        items = f"{parts[0]} and {parts[1]}"
    else:
        items = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    return f"Scene with {items}."
