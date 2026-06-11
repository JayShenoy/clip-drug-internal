"""
visualize_louvain_embeddings.py
────────────────────────────────
UMAP visualization restricted to the 242 compounds that appear in the
Louvain community detection results (r >= 0.38 correlation network).

Each point is coloured by its Louvain community rank (same palette as
compound_network_clusters.png).  Produces three plots:
  louvain_umap_chemical.png   — ChemBERTa embeddings
  louvain_umap_proteome.png   — Proteome MLP embeddings
  louvain_umap_both.png       — side-by-side comparison

Usage:
  pixi run python visualize_louvain_embeddings.py
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

from train_contrastive_clip import (
    ProteomeCLIP,
    ProteomeCLIPDataset,
    load_fc_matrix,
    load_ppi_graph,
    make_collate,
    set_seed,
    cfg,
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


def main():
    # ── checkpoint ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    encoder_type  = ckpt["encoder_type"]
    prot_order    = ckpt["prot_order"]
    all_smiles    = ckpt["valid_smiles"]
    all_fc_keys   = ckpt["valid_fc_keys"]
    print(f"  Epoch {ckpt['epoch']}  |  Top-1 {ckpt['top1']:.3f}  |  encoder={encoder_type}")

    # ── community partition ───────────────────────────────────────────────────
    print(f"\nLoading community results: {CLUSTER_FILE}")
    with open(CLUSTER_FILE) as f:
        cr = json.load(f)
    partition: dict[str, int] = cr["partition"]   # fc_key → community rank
    communities = sorted(cr["communities"], key=lambda c: c["rank"])
    n_communities = cr["n_communities"]
    print(f"  {cr['n_nodes']} compounds in {n_communities} communities")

    # ── filter to Louvain members only ────────────────────────────────────────
    louvain_set = set(partition.keys())
    mask = [k in louvain_set for k in all_fc_keys]
    smiles_sub  = [s for s, m in zip(all_smiles,  mask) if m]
    fc_keys_sub = [k for k, m in zip(all_fc_keys, mask) if m]
    comm_ranks  = [partition[k] for k in fc_keys_sub]
    print(f"  Kept {len(smiles_sub)} / {len(all_smiles)} compounds")

    # colour and label per point
    colors_ext = (BASE_COLORS * math.ceil(n_communities / len(BASE_COLORS)))[:n_communities]
    node_colors = [colors_ext[r] for r in comm_ranks]

    # legend patches — one per community, labelled with hub compound
    hub_of = {}
    for comm in communities:
        # hub = member named first in the list (already hub-ordered from cluster script)
        # fall back to first member
        hub_of[comm["rank"]] = comm["members"][0].split("_")[0]

    patches = [
        mpatches.Patch(
            color=colors_ext[comm["rank"]],
            label=f"C{comm['rank']+1}  ({comm['size']} cpds)  hub: {hub_of[comm['rank']]}",
        )
        for comm in communities
    ]

    # ── rebuild FC matrix + dataset ───────────────────────────────────────────
    set_seed(42)
    fc_df = load_fc_matrix()
    ppi_graph, _ = load_ppi_graph(fc_df)

    print(f"\nBuilding dataset over {len(smiles_sub)} Louvain compounds …")
    dataset = ProteomeCLIPDataset(smiles_sub, fc_keys_sub, fc_df, prot_order)

    # ── restore model ─────────────────────────────────────────────────────────
    print("Restoring model …")
    cfg.encoder_type = encoder_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = ProteomeCLIP(n_proteins=len(prot_order)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── embeddings ────────────────────────────────────────────────────────────
    from torch.utils.data import DataLoader
    collate_fn = make_collate(ppi_graph, encoder_type)
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False,
        num_workers=0, drop_last=False, collate_fn=collate_fn,
    )

    print("Computing embeddings …")
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

    z_chem = np.concatenate(all_zc)   # (242, 128)
    z_prot = np.concatenate(all_zp)
    print(f"  z_chem: {z_chem.shape}  z_prot: {z_prot.shape}")

    # ── UMAP ──────────────────────────────────────────────────────────────────
    def run_umap(emb: np.ndarray) -> np.ndarray:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
            verbose=False,
        )
        return reducer.fit_transform(emb)

    print("Running UMAP (chemical) …")
    xy_chem = run_umap(z_chem)
    print("Running UMAP (proteome) …")
    xy_prot = run_umap(z_prot)

    # ── plots ─────────────────────────────────────────────────────────────────
    subtitle = (
        f"epoch {ckpt['epoch']}, top-1 {ckpt['top1']:.3f}, "
        f"encoder={encoder_type}, n={len(smiles_sub)} (Louvain members only)"
    )

    def _scatter(ax, xy, title):
        ax.scatter(
            xy[:, 0], xy[:, 1],
            c=node_colors,
            s=30,
            alpha=0.85,
            linewidths=0.3,
            edgecolors="white",
        )
        ax.set_title(title, fontsize=11, pad=7)
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)

    print("\nPlotting individual panels …")
    for xy, tag, label in [
        (xy_chem, "chemical", "Chemical Tower (ChemBERTa)"),
        (xy_prot, "proteome", "Proteome Tower (MLP)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 7))
        _scatter(ax, xy, f"{label}\n{subtitle}")
        ax.legend(
            handles=patches, loc="upper right", fontsize=7,
            framealpha=0.85, title="Louvain communities",
            title_fontsize=8, handlelength=1.2,
        )
        plt.tight_layout()
        out = f"louvain_umap_{tag}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")

    print("Plotting side-by-side …")
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    _scatter(axes[0], xy_chem, "Chemical Tower (ChemBERTa)")
    _scatter(axes[1], xy_prot, "Proteome Tower (MLP)")
    fig.legend(
        handles=patches,
        loc="center right",
        fontsize=7,
        framealpha=0.85,
        title="Louvain\ncommunities",
        title_fontsize=8,
        handlelength=1.2,
        bbox_to_anchor=(1.0, 0.5),
    )
    fig.suptitle(
        f"CLIP Embedding Space  —  Louvain members only\n{subtitle}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out = "louvain_umap_both.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
