#!/usr/bin/env python3
"""
efa_items.py — Pseudo Factor Analysis for Psychometric Item Pools
=================================================================
Accepts a CSV with items and scale labels, computes embeddings,
runs PAF + Oblimin EFA, evaluates factor structure, and produces
semantic discrimination scores.

Based on the embedding-EFA approach in:
  mistr3ated/AI-Psychometrics-Nigel (MIT License)
  https://github.com/mistr3ated/AI-Psychometrics-Nigel

INPUT CSV FORMAT (default):
    Column 1 — scale name   e.g. "Perseverance of Effort"
    Column 2 — item text    e.g. "I finish what I start."

USAGE:
    python efa_items.py items.csv
    python efa_items.py items.csv --scale-col Scale --item-col Item
    python efa_items.py items.csv --openai --output-dir results/
    python efa_items.py items.csv --definitions defs.csv --top-k 15
    python efa_items.py items.csv --no-header          # CSV has no header row

DEFINITIONS CSV (optional, --definitions):
    Column 1 — scale name  (must match scale names in items CSV)
    Column 2 — definition  (text description of the construct)
    If omitted, the scale name itself is used as the definition.

OUTPUTS (saved to --output-dir, default: efa_results_<timestamp>/):
    summary.txt              — full text report
    loadings.csv             — factor loadings per item
    assignments.csv          — item assignments + scores
    topk_assignments.csv     — assignments after filtering to top-K items
    similarity_heatmap.jpeg  — cosine similarity matrix (items × items)
    loading_plot.jpeg        — factor loading scatter plot
    embedding_pca.jpeg       — 2-D PCA of item embeddings
    discrimination_plot.jpeg — semantic discrimination score boxplot

REQUIREMENTS:
    pip install sentence-transformers scikit-learn matplotlib pandas numpy
    pip install factor_analyzer          # for Rotator (oblimin)
    pip install openai python-dotenv     # only if --openai is used
"""

import argparse
import sys
import warnings
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ── Factor Analysis ───────────────────────────────────────────────────────────

def principal_axis_factoring(R, n_factors: int, max_iter: int = 300, tol: float = 1e-8):
    """Iterative PAF on a correlation matrix R. Returns unrotated loadings (n × n_factors)."""
    n = len(R)
    R_mod = R.copy()
    try:
        R_inv = np.linalg.inv(R)
        smc = np.clip(1 - 1 / np.diag(R_inv), 0, 1)
    except np.linalg.LinAlgError:
        smc = np.full(n, 0.5)

    for _ in range(max_iter):
        np.fill_diagonal(R_mod, smc)
        eigvals, eigvecs = np.linalg.eigh(R_mod)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        eig_pos = np.maximum(eigvals[:n_factors], 0)
        loadings = eigvecs[:, :n_factors] * np.sqrt(eig_pos)
        new_smc = np.clip(np.sum(loadings ** 2, axis=1), 0, 1)
        if np.max(np.abs(new_smc - smc)) < tol:
            break
        smc = new_smc
    return loadings


def run_efa(R, n_factors: int):
    """PAF extraction + Oblimin rotation. Returns rotated pattern loadings."""
    from factor_analyzer.rotator import Rotator
    L = principal_axis_factoring(R, n_factors)
    return Rotator(method="oblimin").fit_transform(L)


def regularise_corr(mat):
    """Ensure positive-definiteness by shifting eigenvalues and renormalising diagonal."""
    min_eig = np.linalg.eigvalsh(mat).min()
    if min_eig < 0.001:
        mat = mat + (abs(min_eig) + 0.01) * np.eye(len(mat))
        d = np.sqrt(np.diag(mat))
        mat = mat / np.outer(d, d)
    return mat


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_sbert(texts: list[str], model_name: str = "all-mpnet-base-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True)
    return embs.astype(np.float32)


def embed_openai(texts: list[str], model_name: str = "text-embedding-3-small") -> np.ndarray:
    import os
    from openai import OpenAI
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    all_embs = []
    batch = 100
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        resp = client.embeddings.create(model=model_name, input=chunk)
        all_embs.extend([d.embedding for d in resp.data])
        print(f"  Embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    embs = np.array(all_embs, dtype=np.float32)
    embs = normalize(embs)
    return embs


# ── Plotting ──────────────────────────────────────────────────────────────────

# Auto colour palette for up to 10 scales
_PALETTE = [
    "#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
    "#FF9800", "#00BCD4", "#E91E63", "#795548",
    "#607D8B", "#CDDC39",
]


def scale_color_map(scale_names):
    return {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(sorted(set(scale_names)))}


def plot_similarity_heatmap(sim, labels, scale_order, out_path, item_texts=None):
    """Items × items cosine similarity, reordered by scale, with item labels."""
    from matplotlib.patches import Patch
    idx        = np.argsort([scale_order.index(l) for l in labels])
    sim_ord    = sim[np.ix_(idx, idx)]
    ord_labels = labels[idx]
    color_map  = scale_color_map(scale_order)

    boundaries = []
    cur = ord_labels[0]
    for i, l in enumerate(ord_labels):
        if l != cur:
            boundaries.append(i)
            cur = l

    n     = len(idx)
    fig_w = max(9, n * 0.58)
    fig_h = max(7, n * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
    im = ax.imshow(sim_ord, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    for b in boundaries:
        ax.axhline(b - 0.5, color="red", lw=1.5)
        ax.axvline(b - 0.5, color="red", lw=1.5)

    # Tick labels: truncated item text, or fallback to index
    if item_texts is not None:
        tick_labels = [t[:32] + "…" if len(t) > 32 else t for t in item_texts[idx]]
    else:
        tick_labels = [f"item {i}" for i in idx]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)

    # Color each tick label by its scale
    for tick, lbl in zip(ax.get_xticklabels(), ord_labels):
        tick.set_color(color_map[lbl])
    for tick, lbl in zip(ax.get_yticklabels(), ord_labels):
        tick.set_color(color_map[lbl])

    plt.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.6)

    legend_handles = [Patch(color=color_map[s], label=s) for s in scale_order]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.35, 1.0), fontsize=8, title="Scale")

    ax.set_title("Item Embedding Similarity Matrix\n"
                 "(items reordered by scale — red lines = scale boundaries)",
                 fontsize=10, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loadings(loadings, labels, cross, weak, scale_names, factor_labels, cross_thr, out_path):
    """Factor loading scatter (works for 2 factors)."""
    colors = scale_color_map(scale_names)
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=120)

    for i in range(len(loadings)):
        color  = colors[labels[i]]
        marker = "x" if cross[i] else ("s" if weak[i] else "o")
        ax.scatter(loadings[i, 0], loadings[i, 1],
                   c=color, marker=marker, s=65, alpha=0.75, linewidths=1.5)

    for v in [0, cross_thr, -cross_thr]:
        ax.axhline(v, color="grey", lw=0.5 if v != 0 else 0.8,
                   ls="--" if v != 0 else "-", alpha=0.7)
        ax.axvline(v, color="grey", lw=0.5 if v != 0 else 0.8,
                   ls="--" if v != 0 else "-", alpha=0.7)

    handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=c,
               markersize=9, label=s)
        for s, c in colors.items()
    ] + [
        Line2D([0],[0], marker="x", color="grey", markersize=9, lw=1.5,
               label=f"cross-loading (|l|≥{cross_thr})"),
        Line2D([0],[0], marker="s", color="grey", markersize=8,
               label=f"weak (max|l|<{cross_thr})"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    ax.set_xlabel(f"{factor_labels[0]} loading", fontsize=10)
    ax.set_ylabel(f"{factor_labels[1]} loading", fontsize=10)
    ax.set_title("Factor Loadings (PAF + Oblimin)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pca(embs, labels, scale_names, out_path):
    """2-D PCA of item embeddings coloured by scale."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embs)
    colors = scale_color_map(scale_names)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    for scale in sorted(set(labels)):
        mask = np.array(labels) == scale
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[scale], label=scale, s=50, alpha=0.7)
    ax.legend(fontsize=8, loc="best")
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=10)
    ax.set_title("Item Embeddings — PCA Projection", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_discrimination(disc_df, scale_names, out_path):
    """Boxplot of discrimination scores per scale."""
    colors = scale_color_map(scale_names)
    scales = sorted(disc_df["scale"].unique())
    fig, ax = plt.subplots(figsize=(max(5, len(scales) * 1.8), 4), dpi=120)
    data = [disc_df[disc_df["scale"] == s]["disc_mean"].values for s in scales]
    bp = ax.boxplot(data, labels=scales, patch_artist=True, notch=False)
    for patch, s in zip(bp["boxes"], scales):
        patch.set_facecolor(colors[s])
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Discrimination score\n(sim(target) − sim(other constructs))", fontsize=9)
    ax.set_title("Semantic Discrimination by Scale", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pseudo Factor Analysis for psychometric item pools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("csv", help="Input CSV file")
    parser.add_argument("--scale-col",  default=None,
                        help="Column name (or 0-based index) for scale labels. Default: first column.")
    parser.add_argument("--item-col",   default=None,
                        help="Column name (or 0-based index) for item text. Default: second column.")
    parser.add_argument("--definitions", default=None,
                        help="Optional CSV: col1=scale_name, col2=construct definition text.")
    parser.add_argument("--openai",     action="store_true",
                        help="Use OpenAI text-embedding-3-small instead of sentence-transformers.")
    parser.add_argument("--sbert-model", default="all-mpnet-base-v2",
                        help="sentence-transformers model name (default: all-mpnet-base-v2).")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory. Default: efa_results_<timestamp>/")
    parser.add_argument("--n-factors",  type=int, default=None,
                        help="Number of EFA factors. Default: number of unique scales.")
    parser.add_argument("--cross-thr",  type=float, default=0.30,
                        help="Absolute loading threshold for cross-loading / weak flag (default 0.30).")
    parser.add_argument("--top-k",      type=int, default=20,
                        help="Items per scale kept for filtered EFA re-run (default 20).")
    parser.add_argument("--no-header",  action="store_true",
                        help="CSV has no header row (first row is data). "
                             "Columns are selected by position (0=scale, 1=item).")
    args = parser.parse_args()

    # ── Output directory ─────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"efa_results_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutputs → {out_dir}/")

    # ── Step 0: Load data ────────────────────────────────────────────────────
    print("\n── Step 0: Load data ──")
    if args.no_header:
        df = pd.read_csv(args.csv, header=None)
        df.columns = [str(c) for c in df.columns]   # 0, 1, 2, ...
        # override column selectors to positional
        if args.scale_col is None: args.scale_col = "0"
        if args.item_col  is None: args.item_col  = "1"
    else:
        df = pd.read_csv(args.csv)

    def resolve_col(spec, position, df):
        if spec is None:
            return df.columns[position]
        try:
            return df.columns[int(spec)]
        except (ValueError, IndexError):
            return spec

    scale_col = resolve_col(args.scale_col, 0, df)
    item_col  = resolve_col(args.item_col,  1, df)

    if scale_col not in df.columns:
        sys.exit(f"ERROR: scale column '{scale_col}' not found. Columns: {df.columns.tolist()}")
    if item_col not in df.columns:
        sys.exit(f"ERROR: item column '{item_col}' not found. Columns: {df.columns.tolist()}")

    df = df[[scale_col, item_col]].dropna().reset_index(drop=True)
    df.columns = ["scale", "item_text"]
    df["scale"]     = df["scale"].str.strip()
    # collapse embedded newlines / multiple spaces to a single space
    df["item_text"] = (df["item_text"].str.strip()
                                      .str.replace(r"\s+", " ", regex=True))

    scale_names = sorted(df["scale"].unique().tolist())
    n_factors   = args.n_factors if args.n_factors else len(scale_names)
    n_items     = len(df)

    print(f"  Items loaded: {n_items}")
    print(f"  Scales ({len(scale_names)}): {scale_names}")
    print(f"  Items per scale:")
    for s, cnt in df["scale"].value_counts().sort_index().items():
        print(f"    {s}: {cnt}")
    print(f"  EFA factors: {n_factors}")

    # ── Step 1: Compute embeddings ───────────────────────────────────────────
    print("\n── Step 1: Compute embeddings ──")
    texts = df["item_text"].tolist()
    if args.openai:
        print("  Using OpenAI text-embedding-3-small …")
        embs = embed_openai(texts)
    else:
        print(f"  Using sentence-transformers ({args.sbert_model}) …")
        embs = embed_sbert(texts, args.sbert_model)
    print(f"  Embeddings: {embs.shape}")

    # Construct definition embeddings
    if args.definitions:
        defs_df = pd.read_csv(args.definitions, header=None)
        defs_df.columns = ["scale", "definition"]
        defs_df["scale"] = defs_df["scale"].str.strip()
        def_texts = [
            defs_df.loc[defs_df["scale"] == s, "definition"].iloc[0]
            if s in defs_df["scale"].values else s
            for s in scale_names
        ]
    else:
        def_texts = scale_names   # use scale name as proxy definition

    print(f"  Embedding {len(def_texts)} construct definitions …")
    if args.openai:
        def_embs = embed_openai(def_texts)
    else:
        def_embs = embed_sbert(def_texts, args.sbert_model)
    # def_embs[i] corresponds to scale_names[i]

    # ── Step 2: Similarity matrix ────────────────────────────────────────────
    print("\n── Step 2: Similarity matrix ──")
    sim = cosine_similarity(embs)
    print(f"  sim range: [{sim.min():.3f}, {sim.max():.3f}]  mean off-diag: {(sim.sum()-n_items)/(n_items*(n_items-1)):.3f}")

    # Within- vs between-scale similarity
    labels = df["scale"].values
    within, between = [], []
    for i in range(n_items):
        for j in range(i+1, n_items):
            (within if labels[i] == labels[j] else between).append(sim[i, j])
    print(f"  Within-scale sim: mean={np.mean(within):.3f}  Between-scale sim: mean={np.mean(between):.3f}")

    # Plot heatmap
    scale_order = scale_names
    plot_similarity_heatmap(sim, labels, scale_order, out_dir / "similarity_heatmap.jpeg",
                            item_texts=df["item_text"].values)
    print("  Saved: similarity_heatmap.jpeg")

    # PCA plot
    plot_pca(embs, labels, scale_names, out_dir / "embedding_pca.jpeg")
    print("  Saved: embedding_pca.jpeg")

    # ── Step 3: EFA ──────────────────────────────────────────────────────────
    print("\n── Step 3: EFA (PAF + Oblimin) ──")
    sim_reg = regularise_corr(sim)
    try:
        loadings = run_efa(sim_reg, n_factors)
        print(f"  Loadings shape: {loadings.shape}")
    except Exception as e:
        sys.exit(f"  EFA failed: {e}")

    # ── Step 4: Factor–construct alignment ───────────────────────────────────
    print("\n── Step 4: Factor–construct alignment ──")
    abs_load = np.abs(loadings)   # (n_items, n_factors)

    # For each factor, find which scale has the highest average absolute loading on it
    factor_to_scale = {}
    used_factors = set()
    for scale in scale_names:
        mask = labels == scale
        avg_per_factor = abs_load[mask].mean(axis=0)  # (n_factors,)
        # assign factor greedily (highest average, not yet taken)
        ranked = np.argsort(avg_per_factor)[::-1]
        for f in ranked:
            if f not in used_factors:
                factor_to_scale[f] = scale
                used_factors.add(f)
                print(f"  Factor {f+1} → {scale}  (avg |l| = {avg_per_factor[f]:.3f})")
                break
    # If there are more factors than scales, label remaining
    for f in range(n_factors):
        if f not in factor_to_scale:
            factor_to_scale[f] = f"Factor_{f+1}"

    factor_labels = [f"F{f+1}={factor_to_scale[f][:15]}" for f in range(n_factors)]
    scale_to_factor = {v: k for k, v in factor_to_scale.items()}

    # ── Step 5: Item assignment ───────────────────────────────────────────────
    print("\n── Step 5: Item assignment (argmax + DAAL) ──")

    # Argmax: item goes to factor with highest absolute loading
    argmax_factor = np.argmax(abs_load, axis=1)
    argmax_scale  = np.array([factor_to_scale.get(f, f"Factor_{f+1}") for f in argmax_factor])

    # DAAL: normalise each item's loadings by the mean of confirmed items on each factor
    daal_means = {}
    for f, s in factor_to_scale.items():
        mask = labels == s
        daal_means[f] = abs_load[mask, f].mean() + 1e-9

    daal_scale = []
    for i in range(n_items):
        normed = {f: abs_load[i, f] / daal_means[f] for f in factor_to_scale}
        best_f = max(normed, key=normed.get)
        daal_scale.append(factor_to_scale[best_f])
    daal_scale = np.array(daal_scale)

    pct_argmax = (argmax_scale == labels).mean() * 100
    pct_daal   = (daal_scale   == labels).mean() * 100
    print(f"  % correctly assigned (argmax): {pct_argmax:.1f}%")
    print(f"  % correctly assigned (DAAL):   {pct_daal:.1f}%")

    # Problematic items
    cross_loading = np.all(abs_load >= args.cross_thr, axis=1)  # ≥threshold on ALL factors
    weak          = abs_load.max(axis=1) < args.cross_thr
    print(f"  Cross-loading items (|l|≥{args.cross_thr} on all factors): {cross_loading.sum()}")
    print(f"  Weak items (max|l|<{args.cross_thr}): {weak.sum()}")

    # ── Step 6: Residual analysis ─────────────────────────────────────────────
    print("\n── Step 6: Residual analysis ──")
    reproduced    = loadings @ loadings.T
    residuals     = sim - reproduced
    np.fill_diagonal(residuals, 0)
    rmse = float(np.sqrt((residuals ** 2).mean()))
    print(f"  Residual RMSE: {rmse:.4f}")

    # ── Step 7: Semantic discrimination ──────────────────────────────────────
    print("\n── Step 7: Semantic discrimination ──")
    # sim(item, each construct def) → shape (n_items, n_scales)
    all_def_sims = embs @ def_embs.T   # (n_items, n_scales)

    disc_rows = []
    disc_scores = []
    for i, row in df.iterrows():
        target_idx  = scale_names.index(row["scale"])
        target_sim  = all_def_sims[i, target_idx]
        other_sims  = np.delete(all_def_sims[i], target_idx)
        disc1 = target_sim - other_sims.max()   # vs hardest competitor
        disc2 = target_sim - other_sims.mean()  # vs mean of others
        disc_rows.append({"item_idx": i, "scale": row["scale"],
                          "target_sim": round(float(target_sim), 4),
                          "disc_max":   round(float(disc1), 4),
                          "disc_mean":  round(float(disc2), 4)})
        disc_scores.append(disc2)

    disc_df = pd.DataFrame(disc_rows)
    for s in scale_names:
        sub = disc_df[disc_df["scale"] == s]["disc_mean"]
        print(f"  {s}: mean_disc={sub.mean():.4f}  sd={sub.std():.4f}")

    plot_discrimination(disc_df, scale_names, out_dir / "discrimination_plot.jpeg")
    print("  Saved: discrimination_plot.jpeg")

    # ── Step 8: Top-K filtered EFA re-run ────────────────────────────────────
    print(f"\n── Step 8: Filtered EFA (top-{args.top_k} per scale by disc_mean) ──")
    sel_idx = []
    for s in scale_names:
        mask = np.where(labels == s)[0]
        disc_for_scale = disc_df.loc[disc_df["scale"] == s, "disc_mean"].values
        top = mask[np.argsort(disc_for_scale)[-args.top_k:]]
        sel_idx.extend(top.tolist())
    sel_idx = np.array(sel_idx)

    try:
        sub_sim  = regularise_corr(sim[np.ix_(sel_idx, sel_idx)])
        ld2      = run_efa(sub_sim, n_factors)
        abs2     = np.abs(ld2)
        # Re-align factors for filtered set
        true_sel = labels[sel_idx]
        factor_to_scale_2 = {}
        used2 = set()
        for s in scale_names:
            mask2 = true_sel == s
            avg2  = abs2[mask2].mean(axis=0)
            for f in np.argsort(avg2)[::-1]:
                if f not in used2:
                    factor_to_scale_2[f] = s
                    used2.add(f)
                    break
        argmax2 = np.argmax(abs2, axis=1)
        asgn2   = np.array([factor_to_scale_2.get(f, f"Factor_{f+1}") for f in argmax2])
        pct2    = (asgn2 == true_sel).mean() * 100
        print(f"  % correct on top-{args.top_k} per scale: {pct2:.1f}%")
    except Exception as e:
        pct2, ld2, asgn2, sel_idx = None, None, None, np.array([])
        print(f"  Filtered re-run failed: {e}")

    # ── Step 9: Plots ─────────────────────────────────────────────────────────
    print("\n── Step 9: Save plots ──")
    if n_factors == 2:
        ci_fac = scale_to_factor.get(scale_names[0], 0)
        pe_fac = 1 - ci_fac
        ld_plot = np.column_stack([loadings[:, ci_fac], loadings[:, pe_fac]])
        fl = [factor_labels[ci_fac], factor_labels[pe_fac]]
        plot_loadings(ld_plot, labels, cross_loading, weak, scale_names,
                      fl, args.cross_thr, out_dir / "loading_plot.jpeg")
        print("  Saved: loading_plot.jpeg")
    else:
        print(f"  Loading plot skipped (only supported for 2 factors; you have {n_factors}).")

    # ── Step 10: Save data files ──────────────────────────────────────────────
    print("\n── Step 10: Save data files ──")

    # loadings.csv
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f"F{f+1}_{factor_to_scale.get(f,'?')[:12]}" for f in range(n_factors)]
    )
    loadings_df.insert(0, "item_text",      df["item_text"].values)
    loadings_df.insert(0, "scale_true",     labels)
    loadings_df["assigned_argmax"] = argmax_scale
    loadings_df["assigned_DAAL"]   = daal_scale
    loadings_df["cross_loading"]   = cross_loading
    loadings_df["weak"]            = weak
    loadings_df.to_csv(out_dir / "loadings.csv", index=False)
    print(f"  Saved: loadings.csv  ({len(loadings_df)} rows)")

    # assignments.csv (merged with discrimination)
    assign_df = loadings_df[["scale_true","item_text","assigned_argmax",
                              "assigned_DAAL","cross_loading","weak"]].copy()
    assign_df["correct_argmax"] = assign_df["assigned_argmax"] == assign_df["scale_true"]
    assign_df["correct_DAAL"]   = assign_df["assigned_DAAL"]   == assign_df["scale_true"]
    assign_df = assign_df.merge(
        disc_df[["item_idx","target_sim","disc_max","disc_mean"]],
        left_index=True, right_on="item_idx", how="left"
    ).drop(columns="item_idx")
    assign_df.to_csv(out_dir / "assignments.csv", index=False)
    print(f"  Saved: assignments.csv  ({len(assign_df)} rows)")

    # embeddings.csv — AI-GENIE format: rows=dims, cols=items (for aigenie_items.R)
    item_names = [f"item_{i}" for i in range(n_items)]
    emb_df = pd.DataFrame(
        embs.T,                                              # (dim, n_items)
        index   = [f"dim_{d}" for d in range(embs.shape[1])],
        columns = item_names,
    )
    emb_df.to_csv(out_dir / "embeddings.csv")

    # item_labels.csv — companion index for aigenie_items.R
    scale_to_id = {s: i+1 for i, s in enumerate(scale_names)}  # sorted → 1, 2, ...
    pd.DataFrame({
        "item_index":           item_names,
        "item_text":            df["item_text"].values,
        "original_dimension":   df["scale"].values,
        "community_id":         df["scale"].map(scale_to_id).values,
    }).to_csv(out_dir / "item_labels.csv", index=False)
    print(f"  Saved: embeddings.csv  ({embs.shape[1]}D × {n_items} items)  ← input for aigenie_items.R")
    print(f"  Saved: item_labels.csv")

    # topk_assignments.csv
    if len(sel_idx) > 0 and ld2 is not None:
        topk_df = df.iloc[sel_idx][["scale","item_text"]].copy().reset_index(drop=True)
        topk_df["assigned_DAAL"] = asgn2
        topk_df["correct"] = asgn2 == topk_df["scale"].values
        topk_df = topk_df.merge(
            disc_df.iloc[sel_idx][["disc_mean"]].reset_index(drop=True),
            left_index=True, right_index=True, how="left"
        )
        topk_df.to_csv(out_dir / "topk_assignments.csv", index=False)
        print(f"  Saved: topk_assignments.csv  ({len(topk_df)} rows)")

    # summary.json
    summary = {
        "input_file":          str(args.csv),
        "n_items":             n_items,
        "scales":              scale_names,
        "n_factors":           n_factors,
        "embedding":           "openai" if args.openai else args.sbert_model,
        "within_scale_sim":    round(float(np.mean(within)), 4),
        "between_scale_sim":   round(float(np.mean(between)), 4),
        "pct_correct_argmax":  round(pct_argmax, 1),
        "pct_correct_DAAL":    round(pct_daal, 1),
        "n_cross_loading":     int(cross_loading.sum()),
        "n_weak":              int(weak.sum()),
        "residual_rmse":       round(rmse, 4),
        "pct_correct_topK":    round(pct2, 1) if pct2 is not None else None,
        "mean_disc_by_scale":  {
            s: round(float(disc_df[disc_df["scale"]==s]["disc_mean"].mean()), 4)
            for s in scale_names
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # summary.txt
    lines = [
        "=" * 60,
        "PSEUDO FACTOR ANALYSIS — SUMMARY",
        "=" * 60,
        f"Input:      {args.csv}",
        f"Scales:     {', '.join(scale_names)}",
        f"Items:      {n_items}  ({', '.join(f'{s}={c}' for s,c in df['scale'].value_counts().sort_index().items())})",
        f"Embedding:  {'openai' if args.openai else args.sbert_model}  (dim={embs.shape[1]})",
        f"EFA:        PAF + Oblimin  ({n_factors} factor{'s' if n_factors>1 else ''})",
        "",
        "── Embedding similarity ──",
        f"  Within-scale  mean: {np.mean(within):.3f}",
        f"  Between-scale mean: {np.mean(between):.3f}",
        f"  Separation ratio:   {np.mean(within)/np.mean(between):.2f}x",
        "",
        "── Factor alignment ──",
    ] + [
        f"  Factor {f+1} → {factor_to_scale.get(f,'?')}"
        for f in range(n_factors)
    ] + [
        "",
        "── Assignment accuracy ──",
        f"  Argmax:  {pct_argmax:.1f}%",
        f"  DAAL:    {pct_daal:.1f}%",
        f"  Top-{args.top_k}:  {pct2:.1f}%" if pct2 is not None else f"  Top-{args.top_k}: n/a",
        "",
        "── Item flags ──",
        f"  Cross-loading (|l|≥{args.cross_thr} on all factors): {cross_loading.sum()}",
        f"  Weak  (max|l|<{args.cross_thr}):                     {weak.sum()}",
        f"  Residual RMSE:                                  {rmse:.4f}",
        "",
        "── Semantic discrimination (disc_mean) ──",
    ] + [
        f"  {s}: {disc_df[disc_df['scale']==s]['disc_mean'].mean():.4f} ± {disc_df[disc_df['scale']==s]['disc_mean'].std():.4f}"
        for s in scale_names
    ] + [
        "",
        "── Output files ──",
        f"  {out_dir}/",
        "    loadings.csv           — factor loadings per item",
        "    assignments.csv        — assignments + discrimination scores",
        "    topk_assignments.csv   — top-K filtered items",
        "    similarity_heatmap.jpeg",
        "    loading_plot.jpeg",
        "    embedding_pca.jpeg",
        "    discrimination_plot.jpeg",
        "    summary.json",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(report + "\n")

    print(f"\n✓ Done. Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
