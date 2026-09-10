"""3-method correctness overlap: CSV + area-proportional Venn diagrams.

For each record in ``all_models.scored.jsonl`` (one record == one
(PICO, criterion) under one model), we have three 0/1 columns:

  - ``no_evidence_judgement_correctness``       -> method ``no_evidence``
  - ``rag_judgement_correctness``                -> method ``rag_evidence``
  - ``with_evidence_judgement_correctness``      -> method ``with_evidence``

This script answers: of the questions each method got right, how much
do the three methods overlap?

Two render modes (``--mode``):

  ``pico_criterion`` (default)
      Treat every (PICO, criterion) row as a question. One Venn for the
      whole model, saved to a single PNG.

  ``by_criterion``
      One Venn per criterion (12 sub-axes) on a single canvas. Same
      colors / set positions across sub-axes so visual comparison is
      direct.

Outputs land under ``dataset/preliminary_study/output/<model>/`` so
multiple models can coexist:

  - ``<model>_correctness_per_row.csv``       (raw 0/1 per question)
  - ``<model>_venn_counts_<mode>.csv``        (7 region counts)
  - ``<model>_venn_<mode>.png``               (figure)

Usage::

    python -m src.task.preliminary_study.methods_correctness_venn \\
        --model gpt-4o --mode pico_criterion

    python -m src.task.preliminary_study.methods_correctness_venn \\
        --model claude-opus-4-6-thinking --mode by_criterion
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

ETD_ROOT = Path(__file__).resolve().parents[3]
SCORED_JSONL = ETD_ROOT / "dataset/score_rubric/merge/output/all_models.scored.jsonl"
OUTPUT_DIR = ETD_ROOT / "dataset/preliminary_study/output"
SPLITS_FILE = ETD_ROOT / "dataset/train/sft/pico_splits.json"

# (display_name, source field in scored.jsonl, fixed color across all figures)
METHODS: tuple[tuple[str, str, str], ...] = (
    ("no_evidence",   "no_evidence_judgement_correctness",   "#2ca02c"),  # green
    ("rag_evidence",  "rag_judgement_correctness",           "#1f77b4"),  # blue
    ("with_evidence", "with_evidence_judgement_correctness", "#d62728"),  # red
)
METHOD_NAMES: tuple[str, ...] = tuple(m for m, _, _ in METHODS)
METHOD_COLORS: dict[str, str] = {m: c for m, _, c in METHODS}
VENN_FILL_ALPHA = 0.30  # semi-transparent fills so overlaps are visible
VENN_LEGEND_ALPHA = 0.45
# Outer frame: unit-area circle (pi * r^2 = 1) representing the full question set.
FRAME_RADIUS = 1.0 / math.sqrt(math.pi)
FRAME_EDGE_ALPHA = 0.35  # outer border transparency (lower = more transparent)

# Font sizes (pt) — shared across single-panel and 12-panel figures.
FONT_REGION = 12
FONT_NONE_CORRECT = 11
FONT_SET_LABEL = 14
FONT_TITLE = 15
FONT_TITLE_PANEL = 13
FONT_LEGEND = 14
FONT_SUPTITLE = 18

MODES = ("pico_criterion", "by_criterion")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_correctness(
    path: Path,
    model_name: str,
    *,
    splits_file: Path = SPLITS_FILE,
    split: str = "train",
) -> pd.DataFrame:
    """Read scored.jsonl, keep only ``model_name``, return one row per question.

    Columns: pico_source_file, section_index, criterion, no_evidence,
    rag_evidence, with_evidence (the latter three are 0/1).
    """
    split_payload = json.loads(splits_file.read_text(encoding="utf-8"))
    split_map = split_payload.get("split_map", split_payload)
    if not isinstance(split_map, dict):
        raise SystemExit(f"invalid PICO split map: {splits_file}")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("model_name") != model_name:
                continue
            if split_map.get(rec.get("pico_source_file")) != split:
                continue
            row = {
                "pico_source_file": rec.get("pico_source_file"),
                "section_index": rec.get("section_index"),
                "criterion": rec.get("criterion"),
            }
            for display, src_field, _ in METHODS:
                v = rec.get(src_field)
                row[display] = int(v) if isinstance(v, (int, float, bool)) and v is not None else 0
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows found for model_name={model_name!r} in {path}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Counting the 7 Venn regions
# --------------------------------------------------------------------------- #

# Order matters: matches matplotlib-venn's ``subsets`` tuple convention
# (Abc, aBc, ABc, abC, AbC, aBC, ABC) where A=no_evidence, B=rag_evidence,
# C=with_evidence.
REGION_KEYS: tuple[str, ...] = (
    "no_evidence_only",
    "rag_only",
    "no_and_rag_only",
    "with_only",
    "no_and_with_only",
    "rag_and_with_only",
    "all_three",
)
REGION_LABELS_FOR_PLOT: tuple[str, ...] = (
    "100", "010", "110", "001", "101", "011", "111",
)


def count_regions(df: pd.DataFrame) -> dict[str, int]:
    """Tally the 7 Venn regions for the three correctness columns.

    Rows where all three methods are wrong are NOT in any region and
    contribute only to ``none_correct``.
    """
    a = df["no_evidence"].to_numpy().astype(bool)
    b = df["rag_evidence"].to_numpy().astype(bool)
    c = df["with_evidence"].to_numpy().astype(bool)
    counts = {
        "no_evidence_only":   int(np.sum(a & ~b & ~c)),
        "rag_only":           int(np.sum(~a & b & ~c)),
        "no_and_rag_only":    int(np.sum(a & b & ~c)),
        "with_only":          int(np.sum(~a & ~b & c)),
        "no_and_with_only":   int(np.sum(a & ~b & c)),
        "rag_and_with_only":  int(np.sum(~a & b & c)),
        "all_three":          int(np.sum(a & b & c)),
        "none_correct":       int(np.sum(~a & ~b & ~c)),
        "total":              int(len(df)),
        "no_evidence_total":  int(np.sum(a)),
        "rag_total":          int(np.sum(b)),
        "with_total":         int(np.sum(c)),
    }
    return counts


# --------------------------------------------------------------------------- #
# Area-proportional Venn3 (no external dependency)
# --------------------------------------------------------------------------- #
#
# We render three filled circles. Each circle radius is proportional to
# sqrt(method_total) so that the *area* of each circle is proportional to
# the method's correct count. Pairwise center distances are chosen so the
# lens area between any two circles approximates their pairwise overlap.
#
# This is the same idea matplotlib-venn uses; the optimization is closed
# form for two circles and uses bisection for three. Code below is
# self-contained (no scipy needed beyond numpy).


def _circle_intersection_area(r1: float, r2: float, d: float) -> float:
    """Area of the lens formed by two circles with radii r1, r2 and center
    distance d. d, r1, r2 must be non-negative."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a1 = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    a2 = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    a3 = 0.5 * math.sqrt(
        (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    )
    return a1 + a2 - a3


def _solve_distance(r1: float, r2: float, target_overlap: float) -> float:
    """Bisect distance d in [|r1-r2|, r1+r2] so circle overlap matches target."""
    if target_overlap <= 0:
        return r1 + r2  # touch externally, no overlap
    max_overlap = math.pi * min(r1, r2) ** 2
    if target_overlap >= max_overlap:
        return abs(r1 - r2)  # one circle fully inside the other
    lo, hi = abs(r1 - r2), r1 + r2
    for _ in range(60):  # 60 iters of bisection -> ~1e-18 precision
        mid = 0.5 * (lo + hi)
        if _circle_intersection_area(r1, r2, mid) > target_overlap:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class _VennGeom:
    centers: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    radii: tuple[float, float, float]


def _compute_venn3_geometry(counts: dict[str, int]) -> _VennGeom:
    """Pick radii so circle *area* equals the method's correct count expressed
    as a fraction of ``total`` (the frame area is normalized to 1). Pairwise
    lens areas approximate the pairwise overlap fractions. Degenerate cases
    (empty sets) get a tiny fallback radius so nothing disappears."""
    # Set totals (any membership, used for radius)
    sA = counts["no_evidence_only"] + counts["no_and_rag_only"] + counts["no_and_with_only"] + counts["all_three"]
    sB = counts["rag_only"]         + counts["no_and_rag_only"] + counts["rag_and_with_only"] + counts["all_three"]
    sC = counts["with_only"]        + counts["no_and_with_only"] + counts["rag_and_with_only"] + counts["all_three"]
    # Pairwise overlaps (the *combined* AB, AC, BC counts including triple)
    nAB = counts["no_and_rag_only"]  + counts["all_three"]
    nAC = counts["no_and_with_only"] + counts["all_three"]
    nBC = counts["rag_and_with_only"] + counts["all_three"]

    # Frame is a unit-area circle (see FRAME_RADIUS) representing the full set.
    # A circle for a method with fraction f of correct answers gets area = f,
    # so radius = sqrt(f / pi). This makes circle size directly comparable
    # across figures (same fraction -> same circle).
    total = counts["total"] or 1
    def radius(s: int) -> float:
        if s <= 0:
            return 0.02  # tiny but visible
        return math.sqrt((s / total) / math.pi)

    rA, rB, rC = radius(sA), radius(sB), radius(sC)

    # Convert overlap *counts* to overlap *areas* on the same unit-frame scale.
    dAB = _solve_distance(rA, rB, nAB / total)
    dAC = _solve_distance(rA, rC, nAC / total)
    dBC = _solve_distance(rB, rC, nBC / total)

    # Place A at origin, B on x-axis at distance dAB, C by triangulating.
    # If the triangle inequality is violated (rare with extreme overlaps),
    # fall back to a regular triangle of unit-ish side.
    Ax, Ay = 0.0, 0.0
    Bx, By = dAB, 0.0
    if dAB <= 1e-9 or dAC + dBC <= dAB or dAC + dAB <= dBC or dBC + dAB <= dAC:
        # Equilateral fallback: center distance = max(rA+rB, rA+rC, rB+rC) / 2
        side = max(rA + rB, rA + rC, rB + rC) * 0.85
        Bx, By = side, 0.0
        Cx, Cy = side / 2, side * math.sqrt(3) / 2
    else:
        # Law of cosines: angle at A between AB (x-axis) and AC
        cos_a = (dAB * dAB + dAC * dAC - dBC * dBC) / (2 * dAB * dAC)
        cos_a = max(-1.0, min(1.0, cos_a))
        angle = math.acos(cos_a)
        Cx = dAC * math.cos(angle)
        Cy = dAC * math.sin(angle)
    return _VennGeom(centers=((Ax, Ay), (Bx, By), (Cx, Cy)), radii=(rA, rB, rC))


def _geom_bbox(geom: _VennGeom) -> tuple[float, float, float, float]:
    """Axis-aligned bbox of the three circles: (cx, cy, width, height)."""
    xs: list[float] = []
    ys: list[float] = []
    for (x, y), r in zip(geom.centers, geom.radii):
        xs.extend((x - r, x + r))
        ys.extend((y - r, y + r))
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin


def _max_reach_from_center(geom: _VennGeom, cx: float, cy: float) -> float:
    """Farthest distance from (cx, cy) to any point on the three circles."""
    return max(
        math.hypot(x - cx, y - cy) + r
        for (x, y), r in zip(geom.centers, geom.radii)
    )


def _fit_scale_factor(geom: _VennGeom, *, margin: float = 0.96) -> float:
    """Uniform scale so all three circles fit inside the unit-area frame circle."""
    cx, cy, _, _ = _geom_bbox(geom)
    reach = _max_reach_from_center(geom, cx, cy)
    limit = margin * FRAME_RADIUS
    if reach <= limit:
        return 1.0
    return limit / reach


def _scale_geom(geom: _VennGeom, scale: float) -> _VennGeom:
    """Scale centers and radii uniformly about the bbox center."""
    if scale == 1.0:
        return geom
    cx, cy, _, _ = _geom_bbox(geom)
    centers = tuple(
        (cx + scale * (x - cx), cy + scale * (y - cy))
        for x, y in geom.centers
    )
    radii = tuple(r * scale for r in geom.radii)
    return _VennGeom(centers=centers, radii=radii)


def _region_centroid(geom: _VennGeom, region: str) -> tuple[float, float]:
    """Heuristic placement for region count labels (not exact centroids
    of lens shapes, but visually reasonable)."""
    (Ax, Ay), (Bx, By), (Cx, Cy) = geom.centers
    rA, rB, rC = geom.radii
    # Helper: midpoint pull
    def mid(p1, p2, t=0.5):
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    centroid_ABC = ((Ax + Bx + Cx) / 3, (Ay + By + Cy) / 3)
    if region == "all_three":
        return centroid_ABC
    if region == "no_evidence_only":
        # Pull A away from BC midpoint
        bc = ((Bx + Cx) / 2, (By + Cy) / 2)
        dx, dy = Ax - bc[0], Ay - bc[1]
        norm = math.hypot(dx, dy) or 1.0
        return (Ax + dx / norm * rA * 0.55, Ay + dy / norm * rA * 0.55)
    if region == "rag_only":
        ac = ((Ax + Cx) / 2, (Ay + Cy) / 2)
        dx, dy = Bx - ac[0], By - ac[1]
        norm = math.hypot(dx, dy) or 1.0
        return (Bx + dx / norm * rB * 0.55, By + dy / norm * rB * 0.55)
    if region == "with_only":
        ab = ((Ax + Bx) / 2, (Ay + By) / 2)
        dx, dy = Cx - ab[0], Cy - ab[1]
        norm = math.hypot(dx, dy) or 1.0
        return (Cx + dx / norm * rC * 0.55, Cy + dy / norm * rC * 0.55)
    if region == "no_and_rag_only":
        return mid(centroid_ABC, ((Ax + Bx) / 2, (Ay + By) / 2), t=0.9)
    if region == "no_and_with_only":
        return mid(centroid_ABC, ((Ax + Cx) / 2, (Ay + Cy) / 2), t=0.9)
    if region == "rag_and_with_only":
        return mid(centroid_ABC, ((Bx + Cx) / 2, (By + Cy) / 2), t=0.9)
    raise ValueError(region)


def _draw_venn3(
    ax: plt.Axes,
    counts: dict[str, int],
    *,
    title: str,
    show_set_labels: bool = True,
    geom: _VennGeom | None = None,
    title_fontsize: int = FONT_TITLE,
) -> None:
    """Render an area-proportional 3-set Venn into one axes, framed by a
    unit-area circle representing the full question set (``total``). The
    blank space between the circles and the frame is the ``none_correct``
    region (questions every method got wrong).

    Pass ``geom`` when a shared layout scale was already applied (e.g.
    ``by_criterion`` uses one factor across all panels).
    """
    if geom is None:
        geom = _compute_venn3_geometry(counts)
    (Ax, Ay), (Bx, By), (Cx, Cy) = geom.centers
    rA, rB, rC = geom.radii

    # Frame: unit-area circle centered on the (possibly scaled) circle bbox.
    cx, cy, _, _ = _geom_bbox(geom)
    frame = Circle(
        (cx, cy), FRAME_RADIUS,
        facecolor="white",
        edgecolor=(0.4, 0.4, 0.4, FRAME_EDGE_ALPHA),
        linewidth=1.2,
        zorder=0,
    )
    ax.add_patch(frame)

    # Draw filled, semi-transparent circles in fixed method colors.
    for (x, y), r, method in zip(geom.centers, geom.radii, METHOD_NAMES):
        circle = Circle(
            (x, y), r,
            facecolor=METHOD_COLORS[method],
            edgecolor="none",
            alpha=VENN_FILL_ALPHA,
            linewidth=0,
            zorder=1,
        )
        ax.add_patch(circle)

    # Region count labels
    total = counts["total"] or 1
    for region in REGION_KEYS:
        n = counts[region]
        if n == 0:
            continue
        x, y = _region_centroid(geom, region)
        ax.text(
            x, y, f"{n / total:.0%}",
            ha="center", va="center",
            fontsize=FONT_REGION, fontweight="bold",
            zorder=2,
        )

    # none_correct label in the upper-left inside the frame circle.
    n_none = counts.get("none_correct", 0)
    if n_none > 0:
        label_r = FRAME_RADIUS * 0.82
        label_angle = 3 * math.pi / 4  # upper-left
        ax.text(
            cx + label_r * math.cos(label_angle),
            cy + label_r * math.sin(label_angle),
            f"none correct: {n_none / total:.0%}",
            ha="left", va="top",
            fontsize=FONT_NONE_CORRECT, color="#555555", style="italic",
            zorder=2,
        )

    # Set labels next to each circle
    if show_set_labels:
        label_offset = 0.04
        for (x, y), r, method in zip(geom.centers, geom.radii, METHOD_NAMES):
            if y >= cy:
                ax.text(x, y + r + label_offset, method,
                        ha="center", va="bottom",
                        fontsize=FONT_SET_LABEL, color=METHOD_COLORS[method],
                        fontweight="bold", zorder=2)
            else:
                ax.text(x, y - r - label_offset, method,
                        ha="center", va="top",
                        fontsize=FONT_SET_LABEL, color=METHOD_COLORS[method],
                        fontweight="bold", zorder=2)

    # Tidy axes: pad just enough around the frame circle to fit set labels.
    pad = 0.14
    ax.set_xlim(cx - FRAME_RADIUS - pad, cx + FRAME_RADIUS + pad)
    ax.set_ylim(cy - FRAME_RADIUS - pad, cy + FRAME_RADIUS + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=title_fontsize)


# --------------------------------------------------------------------------- #
# Per-mode drivers
# --------------------------------------------------------------------------- #


def _save_counts_csv(counts: dict[str, int], path: Path, scope: str) -> None:
    """Write a single-row CSV of the 7+ counts under a ``scope`` label."""
    row = {"scope": scope, **counts}
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"saved: {path}")


def _save_counts_csv_many(
    rows: Iterable[dict], path: Path
) -> None:
    df = pd.DataFrame(list(rows))
    df.to_csv(path, index=False)
    print(f"saved: {path}")


def render_single(
    df: pd.DataFrame, model: str, out_dir: Path, mode: str
) -> None:
    """Single Venn over all (PICO, criterion) rows (mode == pico_criterion)."""
    counts = count_regions(df)
    csv_path = out_dir / f"{model}_venn_counts_{mode}.csv"
    _save_counts_csv(counts, csv_path, scope=mode)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    title = f"{model} — methods correctness overlap ({mode})\nn = {counts['total']}, all-wrong = {counts['none_correct']}"
    geom = _compute_venn3_geometry(counts)
    geom = _scale_geom(geom, _fit_scale_factor(geom))
    _draw_venn3(ax, counts, title=title, show_set_labels=True, geom=geom)
    fig.tight_layout()
    png_path = out_dir / f"{model}_venn_{mode}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {png_path}")


def render_by_criterion(df: pd.DataFrame, model: str, out_dir: Path) -> None:
    criteria = sorted(df["criterion"].dropna().unique().tolist())
    if not criteria:
        raise SystemExit("No criteria found")

    # 12-panel grid (3 x 4). If more / fewer criteria, recompute layout.
    n = len(criteria)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows)
    )
    axes_flat = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

    panel_data: list[tuple[str, dict[str, int], _VennGeom]] = []
    for crit in criteria:
        sub = df[df["criterion"] == crit]
        counts = count_regions(sub)
        panel_data.append((crit, counts, _compute_venn3_geometry(counts)))

    # One scale for all panels so circle areas stay comparable across criteria.
    global_scale = min(_fit_scale_factor(g) for _, _, g in panel_data)

    rows_for_csv: list[dict] = []
    for i, (crit, counts, geom) in enumerate(panel_data):
        rows_for_csv.append({"scope": "by_criterion", "criterion": crit, **counts})
        _draw_venn3(
            axes_flat[i], counts,
            title=f"{crit}  (n={counts['total']})",
            show_set_labels=(i == 0),  # only label sets once to reduce clutter
            geom=_scale_geom(geom, global_scale),
            title_fontsize=FONT_TITLE_PANEL,
        )

    # Hide unused panels (if any)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_axis_off()

    # Shared legend so the color/method mapping is visible across all panels.
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=METHOD_COLORS[m], markeredgecolor="none",
                   markersize=14, alpha=VENN_LEGEND_ALPHA, label=m)
        for m in METHOD_NAMES
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=3, frameon=False, fontsize=FONT_LEGEND, bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"{model} — methods correctness overlap by criterion",
        fontsize=FONT_SUPTITLE, y=1.0,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])

    png_path = out_dir / f"{model}_venn_by_criterion.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {png_path}")

    csv_path = out_dir / f"{model}_venn_counts_by_criterion.csv"
    _save_counts_csv_many(rows_for_csv, csv_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="gpt-4o",
        help="model_name to filter on (e.g. gpt-4o, gpt-5.5, claude-opus-4-6-thinking).",
    )
    parser.add_argument(
        "--mode", choices=MODES, default="pico_criterion",
        help=(
            "pico_criterion: one Venn over all (PICO, criterion) rows; "
            "by_criterion: 12 panels on one canvas."
        ),
    )
    parser.add_argument(
        "--input", type=Path, default=SCORED_JSONL,
        help=f"Path to scored.jsonl (default: {SCORED_JSONL}).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Where to write outputs. Default: "
            "dataset/preliminary_study/output/<model>/"
        ),
    )
    parser.add_argument("--splits-file", type=Path, default=SPLITS_FILE)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
        help="Split to analyze (default: train; paper preliminary uses train only).",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or (OUTPUT_DIR / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_correctness(
        args.input,
        args.model,
        splits_file=args.splits_file,
        split=args.split,
    )
    per_row_csv = out_dir / f"{args.model}_correctness_per_row.csv"
    df.to_csv(per_row_csv, index=False)
    print(f"saved: {per_row_csv}  (n={len(df)} rows)")

    if args.mode == "pico_criterion":
        render_single(df, args.model, out_dir, mode=args.mode)
    elif args.mode == "by_criterion":
        render_by_criterion(df, args.model, out_dir)
    else:  # pragma: no cover
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
