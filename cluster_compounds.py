"""
Replication of compound-compound correlation network clustering from:
Mitchell et al. "A proteome-wide atlas of drug mechanism of action" Nat Biotechnol 2023

Algorithm (Fig. 4a + Methods):
1. Load log2FC matrix (proteins x compounds)
2. Compute all-vs-all Pearson correlation between compound profiles
3. Filter edges at r >= 0.38 (0.1% FDR threshold stated in Methods/website)
4. Detect communities using Louvain algorithm
5. Visualize network with node colors = community, edge width = correlation strength
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import networkx as nx
import community as community_louvain  # python-louvain
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

PEARSON_CACHE = Path("data/pearson_cache.npz")
CLUSTER_CACHE = Path("data/cluster_results.json")

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(
    "data/41587_2022_1539_MOESM3_ESM.csv",
    encoding="latin-1",
    index_col=0,
)
# Drop UniprotID column, keep only compound columns
df = df.drop(columns=["UniprotID"], errors="ignore")
df = df.apply(pd.to_numeric, errors="coerce")

# Replace 0 with NaN (0 = not quantified in this dataset)
df = df.replace(0, np.nan)

print(f"  Matrix shape: {df.shape[0]} proteins x {df.shape[1]} compounds")

# ── 2. Pearson correlation between compounds ──────────────────────────────────
# Paper: correlations calculated on proteins quantified in >= 200 compounds
# (stated as using rcorr from Hmisc in R, pairwise complete obs)

compounds = df.columns.tolist()
n = len(compounds)
mat = df.values  # shape: proteins x compounds
mat_f = mat.astype(np.float32)

if PEARSON_CACHE.exists():
    print("Loading cached Pearson matrices...")
    cache = np.load(PEARSON_CACHE, allow_pickle=True)
    cached_compounds = cache["compounds"].tolist()
    if cached_compounds == compounds:
        corr_matrix = cache["corr_matrix"]
        count_matrix = cache["count_matrix"]
        print("  Done (from cache).")
    else:
        print("  Cache compound list mismatch — recomputing ...")
        PEARSON_CACHE.unlink()
        corr_matrix = None
else:
    corr_matrix = None

if corr_matrix is None:
    print("Computing pairwise Pearson correlations (pairwise complete obs)...")
    corr_matrix = np.full((n, n), np.nan)
    count_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        xi = mat_f[:, i]
        for j in range(i, n):
            xj = mat_f[:, j]
            mask = ~(np.isnan(xi) | np.isnan(xj))
            cnt = mask.sum()
            count_matrix[i, j] = count_matrix[j, i] = cnt
            if cnt >= 3:
                r, _ = stats.pearsonr(xi[mask], xj[mask])
                corr_matrix[i, j] = corr_matrix[j, i] = r
            else:
                corr_matrix[i, j] = corr_matrix[j, i] = np.nan
        if i % 100 == 0:
            print(f"  {i}/{n}...")

    print("  Done.")
    np.savez_compressed(
        PEARSON_CACHE,
        corr_matrix=corr_matrix.astype(np.float32),
        count_matrix=count_matrix.astype(np.int32),
        compounds=np.array(compounds),
    )
    print(f"  Pearson matrices cached → {PEARSON_CACHE}")

# ── 3. Filter: r >= 0.38 (0.1% FDR), remove pairs with < 200 shared proteins ─
# Paper Methods: "correlations using fewer than 200 datapoints were removed"
FDR_THRESHOLD = 0.38
MIN_SHARED = 200

print(f"Building network (r >= {FDR_THRESHOLD}, shared proteins >= {MIN_SHARED})...")
G = nx.Graph()
G.add_nodes_from(compounds)

for i in range(n):
    for j in range(i + 1, n):
        r = corr_matrix[i, j]
        cnt = count_matrix[i, j]
        if not np.isnan(r) and r >= FDR_THRESHOLD and cnt >= MIN_SHARED:
            G.add_edge(compounds[i], compounds[j], weight=r)

print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# Keep only the largest connected component for layout (mirrors Fig 4a)
largest_cc = max(nx.connected_components(G), key=len)
G_main = G.subgraph(largest_cc).copy()
print(f"  Largest component: {G_main.number_of_nodes()} nodes, {G_main.number_of_edges()} edges")

# ── 4. Community detection (Louvain) ─────────────────────────────────────────
print("Detecting communities (Louvain)...")
np.random.seed(42)
partition = community_louvain.best_partition(G_main, weight="weight", random_state=42)

communities = {}
for node, comm_id in partition.items():
    communities.setdefault(comm_id, []).append(node)

# Sort communities by size descending
sorted_comms = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
print(f"  Found {len(sorted_comms)} communities")
for cid, members in sorted_comms[:10]:
    print(f"    Community {cid}: {len(members)} compounds")

# Remap community IDs to size-rank order
id_remap = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_comms)}
partition_ranked = {node: id_remap[cid] for node, cid in partition.items()}

# Save quantitative results for cross-checking and fast replot
cluster_results = {
    "n_nodes": G_main.number_of_nodes(),
    "n_edges": G_main.number_of_edges(),
    "n_communities": len(sorted_comms),
    "fdr_threshold": FDR_THRESHOLD,
    "min_shared": MIN_SHARED,
    "communities": [
        {
            "rank": rank,
            "size": len(members),
            "members": members,
        }
        for rank, (_, members) in enumerate(sorted_comms)
    ],
    "partition": partition_ranked,
    "edge_list": [
        {"u": u, "v": v, "weight": float(G_main[u][v]["weight"])}
        for u, v in G_main.edges()
    ],
}
with open(CLUSTER_CACHE, "w") as f:
    json.dump(cluster_results, f)
print(f"  Cluster results saved → {CLUSTER_CACHE}")

# ── 5. Layout ─────────────────────────────────────────────────────────────────
print("Computing layout (community-aware spring)...")
# Build a condensed graph of community hubs, then lay out each community
# around its centroid — gives clearer cluster separation like the paper.

# 5a. Compress graph: one super-node per community
comm_of = partition_ranked  # node -> community rank index
community_nodes = {}
for node, cid in comm_of.items():
    community_nodes.setdefault(cid, []).append(node)

n_comms_main = len(community_nodes)
# Place community centroids on a circle
comm_centroids = {}
for i, (cid, members) in enumerate(sorted(community_nodes.items(), key=lambda x: -len(x[1]))):
    angle = 2 * np.pi * i / n_comms_main
    # Radius proportional to community size
    radius = 2.5 + 0.3 * len(members)
    comm_centroids[cid] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

# 5b. Within each community do a spring layout centred on the centroid
pos = {}
rng = np.random.default_rng(42)
for cid, members in sorted(community_nodes.items()):
    if len(members) == 1:
        pos[members[0]] = comm_centroids[cid] + rng.uniform(-0.1, 0.1, 2)
        continue
    subg = G_main.subgraph(members)
    sub_pos = nx.spring_layout(subg, weight="weight", seed=42,
                               k=1.0 / np.sqrt(len(members)), iterations=80)
    # Scale and shift to centroid
    coords = np.array(list(sub_pos.values()))
    if len(coords) > 1:
        coords = (coords - coords.mean(axis=0))
        scale = max(0.5, np.sqrt(len(members)) * 0.15)
        coords = coords / max(np.abs(coords).max(), 1e-6) * scale
    for node, coord in zip(sub_pos.keys(), coords):
        pos[node] = comm_centroids[cid] + coord

# ── 5c. Fine-tune positions with a global pass that preserves community structure
# One final spring relaxation over the full graph starting from community positions
pos = nx.spring_layout(G_main, pos=pos, weight="weight", seed=42,
                       k=0.8, iterations=50, fixed=None)

# ── 6. Colour scheme ──────────────────────────────────────────────────────────
n_comms = len(sorted_comms)

# Use a qualitative palette; cycle if > 20 communities
base_colors = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#A65628", "#F781BF", "#999999", "#66C2A5", "#FC8D62",
    "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494",
    "#B3B3B3", "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
    "#66A61E", "#E6AB02", "#A6761D", "#666666", "#8DD3C7",
]
while len(base_colors) < n_comms:
    base_colors += base_colors

comm_color = {cid: base_colors[i] for i, (cid, _) in enumerate(sorted_comms)}
node_colors = [comm_color[partition_ranked[n]] for n in G_main.nodes()]

# Edge colours by correlation strength (matches paper: yellow=0.4 → purple=0.9)
edge_cmap = plt.cm.plasma
edge_weights = [G_main[u][v]["weight"] for u, v in G_main.edges()]
edge_norm = mcolors.Normalize(vmin=0.4, vmax=1.0)
edge_colors = [edge_cmap(edge_norm(w)) for w in edge_weights]
edge_widths = [0.2 + 2.5 * max(0, (w - 0.38)) / (1.0 - 0.38) for w in edge_weights]

# ── 7. Plot ───────────────────────────────────────────────────────────────────
print("Plotting...")
fig, ax = plt.subplots(figsize=(18, 16))

# Draw edges: intra-community with higher alpha, inter-community faded
intra_edges = [(u, v) for u, v in G_main.edges()
               if partition_ranked[u] == partition_ranked[v]]
inter_edges = [(u, v) for u, v in G_main.edges()
               if partition_ranked[u] != partition_ranked[v]]

def edge_idx(edge_list):
    edge_set = {(u, v): i for i, (u, v) in enumerate(G_main.edges())}
    edge_set.update({(v, u): i for i, (u, v) in enumerate(G_main.edges())})
    return [edge_set[(u, v)] for u, v in edge_list]

intra_idx = edge_idx(intra_edges)
inter_idx = edge_idx(inter_edges)

nx.draw_networkx_edges(
    G_main, pos, ax=ax,
    edgelist=intra_edges,
    edge_color=[edge_colors[i] for i in intra_idx],
    width=[edge_widths[i] for i in intra_idx],
    alpha=0.75,
)
nx.draw_networkx_edges(
    G_main, pos, ax=ax,
    edgelist=inter_edges,
    edge_color=[edge_colors[i] for i in inter_idx],
    width=[edge_widths[i] * 0.4 for i in inter_idx],
    alpha=0.2,
)

# Draw nodes
node_sizes = [30 + 8 * G_main.degree(n) for n in G_main.nodes()]
nx.draw_networkx_nodes(
    G_main, pos, ax=ax,
    node_color=node_colors,
    node_size=node_sizes,
    linewidths=0.3,
    edgecolors="white",
)

# Label the top 12 communities with their most-connected member
for rank, (old_cid, members) in enumerate(sorted_comms[:12]):
    # Find the node in this community with highest degree
    subg = G_main.subgraph(members)
    if len(subg) == 0:
        continue
    hub = max(subg.nodes(), key=lambda n: G_main.degree(n))
    x, y = pos[hub]
    # Strip the _TARGET suffix for a cleaner label
    label = hub.split("_")[0] if "_" in hub else hub
    ax.annotate(
        label,
        xy=(x, y),
        fontsize=7,
        fontweight="bold",
        color="black",
        ha="center",
        va="bottom",
        xytext=(0, 6),
        textcoords="offset points",
    )

# Colorbar for edge correlation
sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.35, pad=0.01, aspect=20)
cbar.set_label("Pearson r", fontsize=11)
cbar.set_ticks([0.4, 0.6, 0.8, 1.0])

# Legend for top communities (show top 12)
legend_patches = []
for rank, (old_cid, members) in enumerate(sorted_comms[:12]):
    color = comm_color[rank]
    # Name the community by its hub compound (highest degree node)
    subg = G_main.subgraph(members)
    if not subg.nodes():
        continue
    hub = max(subg.nodes(), key=lambda n: G_main.degree(n))
    hub_label = hub.split("_")[0] if "_" in hub else hub
    patch = mpatches.Patch(
        color=color,
        label=f"Community {rank + 1} ({len(members)} cpds, hub: {hub_label})",
    )
    legend_patches.append(patch)

leg = ax.legend(
    handles=legend_patches,
    loc="upper left",
    fontsize=7,
    framealpha=0.85,
    title="Top 12 communities",
    title_fontsize=8,
    ncol=1,
    handlelength=1.2,
)

ax.set_title(
    "Compound–Compound Correlation Network\n"
    "Pearson r ≥ 0.38 (0.1% FDR), Louvain community detection\n"
    f"n = {G_main.number_of_nodes()} compounds, {G_main.number_of_edges()} edges",
    fontsize=13,
    pad=12,
)
ax.axis("off")

plt.tight_layout()
out_path = "compound_network_clusters.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved → {out_path}")
