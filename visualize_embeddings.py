"""
visualize_embeddings.py
───────────────────────
Load the saved CLIP checkpoint, compute both chemical and proteome embeddings
for every matched compound, then project to 2D with UMAP and colour-code nodes
by the Louvain community labels from cluster_compounds.py.

Produces two plots:
  embeddings_umap_chemical.png   — ChemBERTa projections coloured by community
  embeddings_umap_proteome.png   — Proteome MLP projections coloured by community
  embeddings_umap_both.png       — Side-by-side comparison

Usage:
  pixi run python visualize_embeddings.py
"""

import json
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import umap

# ── import model components from training script ──────────────────────────────
from train_contrastive_clip import (
    TrainConfig,
    ProteomeCLIP,
    ProteomeCLIPDataset,
    load_fc_matrix,
    load_smiles_table,
    load_ppi_graph,
    make_collate,
    set_seed,
)

CHECKPOINT  = Path("best_proteome_clip.pt")
CLUSTER_FILE = Path("data/cluster_results.json")

# ── community colour palette (same as cluster_compounds.py) ──────────────────
BASE_COLORS = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#A65628", "#F781BF", "#999999", "#66C2A5", "#FC8D62",
    "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494",
    "#B3B3B3", "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
    "#66A61E", "#E6AB02", "#A6761D", "#666666", "#8DD3C7",
]
UNMATCHED_COLOR = "#CCCCCC"


def load_checkpoint():
    """Load model + metadata from checkpoint."""
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    print(f"  Epoch {ckpt['epoch']}  |  Top-1 {ckpt['top1']:.3f}  |  Top-5 {ckpt['top5']:.3f}")
    return ckpt


def load_community_map(fc_keys: list[str]) -> dict[str, int]:
    """
    Build a compound_fc_key → community_rank mapping from the Louvain partition.

    The cluster_results.json stores partition keys as raw compound column names
    (same as fc_keys).  Community rank 0 = largest community.
    Returns -1 for unmatched compounds.
    """
    print(f"Loading community assignments: {CLUSTER_FILE}")
    with open(CLUSTER_FILE) as f:
        cr = json.load(f)

    partition: dict[str, int] = cr["partition"]   # fc_key → community rank
    comm_map = {k: partition.get(k, -1) for k in fc_keys}
    n_matched   = sum(1 for v in comm_map.values() if v >= 0)
    n_community = cr["n_communities"]
    print(f"  {n_matched}/{len(fc_keys)} compounds matched to {n_community} communities")
    return comm_map, n_community


@torch.no_grad()
def compute_embeddings(
    model:       ProteomeCLIP,
    dataset:     ProteomeCLIPDataset,
    ppi_graph,
    encoder_type: str,
    batch_size:   int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a full forward pass over *all* items (no train/val split).
    Returns (z_chem, z_prot) — both (N, D) numpy arrays, L2-normalised.
    """
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    collate_fn = make_collate(ppi_graph, encoder_type)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False, collate_fn=collate_fn,
    )

    all_zc, all_zp = [], []
    for smiles_batch, proteome_data in loader:
        # Tokenise SMILES
        enc        = model.chem_tower.tokenize(list(smiles_batch))
        input_ids  = enc["input_ids"].to(device)
        attn_mask  = enc["attention_mask"].to(device)
        if not isinstance(proteome_data, torch.Tensor):
            proteome_data = proteome_data.to(device)
        else:
            proteome_data = proteome_data.to(device)

        zc, zp = model(input_ids, attn_mask, proteome_data)
        all_zc.append(zc.cpu().numpy())
        all_zp.append(zp.cpu().numpy())

    return np.concatenate(all_zc), np.concatenate(all_zp)


def run_umap(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    """Project (N, D) embeddings → (N, 2) with UMAP."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",       # matches the CLIP cosine similarity
        random_state=seed,
        verbose=False,
    )
    return reducer.fit_transform(embeddings)


def _build_colors_and_legend(
    fc_keys:     list[str],
    comm_map:    dict[str, int],
    n_communities: int,
) -> tuple[list[str], list[mpatches.Patch]]:
    colors_ext = (BASE_COLORS * math.ceil(n_communities / len(BASE_COLORS)))[:n_communities]

    node_colors = []
    for k in fc_keys:
        rank = comm_map.get(k, -1)
        node_colors.append(colors_ext[rank] if rank >= 0 else UNMATCHED_COLOR)

    # Legend: top communities by size (max 15 shown)
    from train_contrastive_clip import cfg
    with open(CLUSTER_FILE) as f:
        cr = json.load(f)
    communities = sorted(cr["communities"], key=lambda c: c["rank"])

    patches = []
    for comm in communities[:15]:
        rank  = comm["rank"]
        size  = comm["size"]
        color = colors_ext[rank]
        patches.append(mpatches.Patch(color=color, label=f"Community {rank+1}  ({size} cpds)"))
    if any(c == UNMATCHED_COLOR for c in node_colors):
        patches.append(mpatches.Patch(color=UNMATCHED_COLOR, label="No community"))

    return node_colors, patches


def plot_umap(
    xy:           np.ndarray,
    fc_keys:      list[str],
    comm_map:     dict[str, int],
    n_communities: int,
    title:        str,
    out_path:     str,
):
    node_colors, patches = _build_colors_and_legend(fc_keys, comm_map, n_communities)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        xy[:, 0], xy[:, 1],
        c=node_colors,
        s=18,
        alpha=0.75,
        linewidths=0.2,
        edgecolors="white",
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.legend(
        handles=patches,
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
        title="Louvain communities",
        title_fontsize=8,
        ncol=1,
        handlelength=1.2,
    )
    ax.axis("on")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_umap_side_by_side(
    xy_chem:      np.ndarray,
    xy_prot:      np.ndarray,
    fc_keys:      list[str],
    comm_map:     dict[str, int],
    n_communities: int,
    out_path:     str,
):
    node_colors, patches = _build_colors_and_legend(fc_keys, comm_map, n_communities)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, xy, title in zip(
        axes,
        [xy_chem, xy_prot],
        ["Chemical Tower (ChemBERTa)", "Proteome Tower (MLP)"],
    ):
        ax.scatter(
            xy[:, 0], xy[:, 1],
            c=node_colors,
            s=18,
            alpha=0.75,
            linewidths=0.2,
            edgecolors="white",
        )
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)

    # Shared legend on the right
    fig.legend(
        handles=patches,
        loc="center right",
        fontsize=7,
        framealpha=0.85,
        title="Louvain\ncommunities",
        title_fontsize=8,
        ncol=1,
        handlelength=1.2,
        bbox_to_anchor=(1.0, 0.5),
    )
    fig.suptitle(
        "CLIP Embedding Space  —  coloured by Louvain compound communities",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main():
    # ── load checkpoint ───────────────────────────────────────────────────────
    ckpt = load_checkpoint()
    encoder_type = ckpt["encoder_type"]
    prot_order   = ckpt["prot_order"]
    valid_smiles = ckpt["valid_smiles"]
    valid_fc_keys = ckpt["valid_fc_keys"]

    set_seed(42)

    # ── rebuild FC matrix + PPI graph (needed for collate / dataset) ──────────
    fc_df      = load_fc_matrix()
    ppi_graph, _ = load_ppi_graph(fc_df)

    # ── dataset over ALL compounds (no train/val split) ───────────────────────
    print(f"\nBuilding dataset: {len(valid_smiles)} compounds, {len(prot_order)} proteins")
    dataset = ProteomeCLIPDataset(valid_smiles, valid_fc_keys, fc_df, prot_order)

    # ── restore model ─────────────────────────────────────────────────────────
    print("\nRestoring model …")
    # Temporarily override cfg so ProteomeCLIP uses the right encoder
    from train_contrastive_clip import cfg
    cfg.encoder_type = encoder_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = ProteomeCLIP(n_proteins=len(prot_order)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── compute embeddings ────────────────────────────────────────────────────
    print("\nComputing embeddings …")
    z_chem, z_prot = compute_embeddings(
        model, dataset, ppi_graph, encoder_type, batch_size=128,
    )
    print(f"  z_chem: {z_chem.shape}  z_prot: {z_prot.shape}")

    # ── load community assignments ────────────────────────────────────────────
    comm_map, n_communities = load_community_map(valid_fc_keys)

    # ── UMAP projections ──────────────────────────────────────────────────────
    print("\nRunning UMAP (chemical) …")
    xy_chem = run_umap(z_chem)
    print("Running UMAP (proteome) …")
    xy_prot = run_umap(z_prot)

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nPlotting …")
    epoch    = ckpt["epoch"]
    top1     = ckpt["top1"]
    subtitle = f"epoch {epoch}, top-1 {top1:.3f}, encoder={encoder_type}, n={len(valid_smiles)}"

    plot_umap(
        xy_chem, valid_fc_keys, comm_map, n_communities,
        title=f"Chemical embeddings (ChemBERTa)\n{subtitle}",
        out_path="embeddings_umap_chemical.png",
    )
    plot_umap(
        xy_prot, valid_fc_keys, comm_map, n_communities,
        title=f"Proteome embeddings (MLP)\n{subtitle}",
        out_path="embeddings_umap_proteome.png",
    )
    plot_umap_side_by_side(
        xy_chem, xy_prot, valid_fc_keys, comm_map, n_communities,
        out_path="embeddings_umap_both.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
