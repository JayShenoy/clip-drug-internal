"""
visualize_embeddings.py
───────────────────────
Load the saved CLIP checkpoint, compute both chemical and proteome embeddings
for all matched compounds, then project to 2D with UMAP and colour-code nodes
by the Louvain community labels from cluster_compounds.py.

Produces six plots (all compounds + Louvain-only subset):
  embeddings_umap_chemical.png          — all compounds, ChemBERTa embeddings
  embeddings_umap_proteome.png          — all compounds, Proteome MLP embeddings
  embeddings_umap_both.png              — all compounds, side-by-side
  louvain_umap_chemical.png             — Louvain members only, ChemBERTa
  louvain_umap_proteome.png             — Louvain members only, Proteome MLP
  louvain_umap_both.png                 — Louvain members only, side-by-side

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
import torch
import umap
from torch.utils.data import DataLoader

from train_contrastive_clip import (
    ProteomeCLIP,
    ProteomeCLIPDataset,
    cfg,
    load_fc_matrix,
    load_ppi_graph,
    make_collate,
    set_seed,
)

CHECKPOINT   = Path("best_proteome_clip.pt")
CLUSTER_FILE = Path("data/cluster_results.json")

BASE_COLORS = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#A65628", "#F781BF", "#999999", "#66C2A5", "#FC8D62",
    "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494",
    "#B3B3B3", "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
    "#66A61E", "#E6AB02", "#A6761D", "#666666", "#8DD3C7",
]
UNMATCHED_COLOR = "#CCCCCC"


# ── helpers ───────────────────────────────────────────────────────────────────

def _colors_ext(n: int) -> list[str]:
    return (BASE_COLORS * math.ceil(n / len(BASE_COLORS)))[:n]


def _build_colors_and_legend(
    fc_keys: list[str],
    partition: dict[str, int],
    communities: list[dict],
    n_communities: int,
    include_unmatched: bool,
) -> tuple[list[str], list[mpatches.Patch]]:
    ext = _colors_ext(n_communities)
    node_colors = [
        ext[partition[k]] if k in partition else UNMATCHED_COLOR
        for k in fc_keys
    ]
    patches = [
        mpatches.Patch(
            color=ext[c["rank"]],
            label=f"C{c['rank']+1}  ({c['size']} cpds)",
        )
        for c in communities[:15]
    ]
    if include_unmatched and any(col == UNMATCHED_COLOR for col in node_colors):
        patches.append(mpatches.Patch(color=UNMATCHED_COLOR, label="No community"))
    return node_colors, patches


def run_umap(embeddings: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    return reducer.fit_transform(embeddings)


def _scatter(ax, xy, node_colors, title, point_size=18):
    ax.scatter(
        xy[:, 0], xy[:, 1],
        c=node_colors,
        s=point_size,
        alpha=0.80,
        linewidths=0.2,
        edgecolors="white",
    )
    ax.set_title(title, fontsize=11, pad=7)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)


def save_single(xy, node_colors, patches, title, out_path, point_size=18):
    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter(ax, xy, node_colors, title, point_size)
    ax.legend(
        handles=patches, loc="upper right", fontsize=7,
        framealpha=0.85, title="Louvain communities",
        title_fontsize=8, handlelength=1.2,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def save_side_by_side(
    xy_chem, xy_prot, node_colors, patches, suptitle, out_path, point_size=18,
):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    _scatter(axes[0], xy_chem, node_colors, "Chemical Tower (ChemBERTa)", point_size)
    _scatter(axes[1], xy_prot, node_colors, "Proteome Tower (MLP)", point_size)
    fig.legend(
        handles=patches, loc="center right", fontsize=7,
        framealpha=0.85, title="Louvain\ncommunities",
        title_fontsize=8, handlelength=1.2,
        bbox_to_anchor=(1.0, 0.5),
    )
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── checkpoint ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    encoder_type  = ckpt["encoder_type"]
    prot_order    = ckpt["prot_order"]
    valid_smiles  = ckpt["valid_smiles"]
    valid_fc_keys = ckpt["valid_fc_keys"]
    print(f"  Epoch {ckpt['epoch']}  |  Top-1 {ckpt['top1']:.3f}  |  encoder={encoder_type}")

    set_seed(42)

    # ── community data ────────────────────────────────────────────────────────
    print(f"Loading community results: {CLUSTER_FILE}")
    with open(CLUSTER_FILE) as f:
        cr = json.load(f)
    partition: dict[str, int] = cr["partition"]
    communities = sorted(cr["communities"], key=lambda c: c["rank"])
    n_communities = cr["n_communities"]
    n_matched = sum(1 for k in valid_fc_keys if k in partition)
    print(f"  {n_matched}/{len(valid_fc_keys)} compounds matched to {n_communities} communities")

    # ── data ──────────────────────────────────────────────────────────────────
    fc_df = load_fc_matrix()
    ppi_graph, _ = load_ppi_graph(fc_df)

    print(f"\nBuilding dataset: {len(valid_smiles)} compounds, {len(prot_order)} proteins")
    dataset = ProteomeCLIPDataset(valid_smiles, valid_fc_keys, fc_df, prot_order)

    # ── model ─────────────────────────────────────────────────────────────────
    print("Restoring model …")
    cfg.encoder_type = encoder_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = ProteomeCLIP(n_proteins=len(prot_order)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── embeddings (all compounds) ────────────────────────────────────────────
    print("\nComputing embeddings …")
    collate_fn = make_collate(ppi_graph, encoder_type)
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False,
        num_workers=0, drop_last=False, collate_fn=collate_fn,
    )
    all_zc, all_zp = [], []
    with torch.no_grad():
        for smiles_batch, proteome_data in loader:
            enc       = model.chem_tower.tokenize(list(smiles_batch))
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            proteome_data = proteome_data.to(device)
            zc, zp = model(input_ids, attn_mask, proteome_data)
            all_zc.append(zc.cpu().numpy())
            all_zp.append(zp.cpu().numpy())
    z_chem = np.concatenate(all_zc)   # (N, 128)
    z_prot = np.concatenate(all_zp)
    print(f"  z_chem: {z_chem.shape}  z_prot: {z_prot.shape}")

    # ── shared subtitle ───────────────────────────────────────────────────────
    sub = f"epoch {ckpt['epoch']}, top-1 {ckpt['top1']:.3f}, encoder={encoder_type}"

    # ══════════════════════════════════════════════════════════════════════════
    # Plot set 1: ALL compounds
    # ══════════════════════════════════════════════════════════════════════════
    print("\nRunning UMAP on all compounds …")
    xy_chem_all = run_umap(z_chem)
    xy_prot_all = run_umap(z_prot)

    colors_all, patches_all = _build_colors_and_legend(
        valid_fc_keys, partition, communities, n_communities, include_unmatched=True,
    )

    print("Plotting (all compounds) …")
    save_single(
        xy_chem_all, colors_all, patches_all,
        f"Chemical embeddings — all compounds\n{sub}, n={len(valid_smiles)}",
        "embeddings_umap_chemical.png",
    )
    save_single(
        xy_prot_all, colors_all, patches_all,
        f"Proteome embeddings — all compounds\n{sub}, n={len(valid_smiles)}",
        "embeddings_umap_proteome.png",
    )
    save_side_by_side(
        xy_chem_all, xy_prot_all, colors_all, patches_all,
        f"CLIP embeddings — all compounds\n{sub}, n={len(valid_smiles)}",
        "embeddings_umap_both.png",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Plot set 2: Louvain members only
    # ══════════════════════════════════════════════════════════════════════════
    louvain_mask = [k in partition for k in valid_fc_keys]
    fc_keys_lv   = [k for k, m in zip(valid_fc_keys, louvain_mask) if m]
    z_chem_lv    = z_chem[[m for m in louvain_mask]]
    z_prot_lv    = z_prot[[m for m in louvain_mask]]

    # rebuild boolean array properly
    idx_lv       = [i for i, m in enumerate(louvain_mask) if m]
    z_chem_lv    = z_chem[idx_lv]
    z_prot_lv    = z_prot[idx_lv]

    print(f"\nRunning UMAP on {len(fc_keys_lv)} Louvain compounds …")
    xy_chem_lv = run_umap(z_chem_lv)
    xy_prot_lv = run_umap(z_prot_lv)

    colors_lv, patches_lv = _build_colors_and_legend(
        fc_keys_lv, partition, communities, n_communities, include_unmatched=False,
    )

    print("Plotting (Louvain members) …")
    save_single(
        xy_chem_lv, colors_lv, patches_lv,
        f"Chemical embeddings — Louvain members\n{sub}, n={len(fc_keys_lv)}",
        "louvain_umap_chemical.png",
        point_size=30,
    )
    save_single(
        xy_prot_lv, colors_lv, patches_lv,
        f"Proteome embeddings — Louvain members\n{sub}, n={len(fc_keys_lv)}",
        "louvain_umap_proteome.png",
        point_size=30,
    )
    save_side_by_side(
        xy_chem_lv, xy_prot_lv, colors_lv, patches_lv,
        f"CLIP embeddings — Louvain members only\n{sub}, n={len(fc_keys_lv)}",
        "louvain_umap_both.png",
        point_size=30,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
