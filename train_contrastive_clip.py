"""
train_contrastive_clip.py
─────────────────────────
Dual-Tower CLIP for aligning small-molecule SMILES with cellular proteome traces.
Dataset:  Mitchell et al. Nat Biotechnol 2023  ("A proteome-wide atlas of drug MOA")

Architecture
────────────
Chemical Tower  : frozen ChemBERTa-77M-MTR  →  2-layer MLP head  →  D=128
Proteome Tower  : Trace Encoder — each compound's log2FC values mapped as node
                  features onto the protein-protein interactome (MOESM6, Pearson
                  correlations across all 875 compounds, 35 936 edges, 2 387
                  proteins).  The continuous, un-thresholded Pearson r is the edge
                  weight; no hard cutoffs.  A 3-layer multi-head GAT
                  (torch_geometric.nn.GATConv) aggregates the signal across the
                  interactome, learning to weight edges natively during
                  backpropagation.  Global mean pool over all protein nodes
                  →  2-layer MLP Projection Head  →  D=128
Loss            : symmetric InfoNCE (CLIP-style) with learnable temperature
Optimiser       : AdamW + CosineAnnealingLR
Evaluation      : Top-1 / Top-5 retrieval accuracy on validation set
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import math
import os
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import openpyxl
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoTokenizer

from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from torch_geometric.nn import GATConv, global_mean_pool

# ────────────────────────────────────────────────────────────────────────────
# 0.  Config
# ────────────────────────────────────────────────────────────────────────────
SEED = 42
LATENT_DIM = 128           # shared embedding dimension D
MLP_HIDDEN = 512           # hidden width of both projection heads
BATCH_SIZE = 64
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-2
VAL_FRAC = 0.20
CHECKPOINT_PATH = "best_proteome_clip.pt"

# ChemBERTa checkpoint
CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MTR"
MAX_SMILES_LEN = 128

# Protein-protein interactome (MOESM6)
PPI_FILE = "data/41587_2022_1539_MOESM6_ESM.xlsx"

# GAT dims — node features are scalar (1-dim FC value per protein)
GNN_IN_DIM = 1
GNN_HIDDEN = 256
GNN_HEADS = 4              # GAT attention heads

# ─── device: prefer Apple-Silicon MPS, then CUDA, else CPU ──────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


# ────────────────────────────────────────────────────────────────────────────
# 1.  Reproducibility
# ────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ────────────────────────────────────────────────────────────────────────────
# 2.  Data loading  (identical pipeline to cluster_compounds.py)
# ────────────────────────────────────────────────────────────────────────────

def load_fc_matrix() -> pd.DataFrame:
    """Load log2FC matrix (proteins × compounds). 0 → NaN as in cluster script."""
    print("Loading fold-change matrix …")
    df = pd.read_csv(
        "data/41587_2022_1539_MOESM3_ESM.csv",
        encoding="latin-1",
        index_col=0,
    )
    df = df.drop(columns=["UniprotID"], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace(0, np.nan)
    # Normalise column names: strip non-breaking spaces (3 columns have \xa0)
    df.columns = [c.replace("\xa0", " ").strip() for c in df.columns]
    print(f"  FC matrix: {df.shape[0]} proteins × {df.shape[1]} compounds")
    return df


def load_smiles_table() -> pd.DataFrame:
    """Load supplementary compound table (MOESM4) that contains SMILES."""
    print("Loading SMILES table …")
    wb = openpyxl.load_workbook("data/41587_2022_1539_MOESM4_ESM.xlsx", read_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    # Reconstruct the FC-matrix column key: "CompoundName_PrimaryTarget"
    df["fc_key"] = (
        df["Compound Name"].str.strip() + "_" + df["Primary Target"].str.strip()
    )
    df = df.dropna(subset=["SMILES"])
    print(f"  SMILES table: {len(df)} compounds with valid SMILES")
    return df


PEARSON_CACHE = Path("data/pearson_cache.npz")


def build_pearson_graph(
    fc_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Compute pairwise Pearson correlations with pairwise-complete observations.
    Mirrors the exact loop in cluster_compounds.py.

    Results are cached to PEARSON_CACHE and reloaded on subsequent runs.

    Returns
    -------
    corr_matrix  : (N, N) float32
    count_matrix : (N, N) int32
    mat_f        : (P, N) float32  —  raw FC values used for node features
    compounds    : list of column names (length N)
    """
    print("Building Pearson correlation graph (mirrors cluster_compounds.py) …")
    compounds = fc_df.columns.tolist()
    mat_f = fc_df.values.astype(np.float32)  # proteins × compounds

    if PEARSON_CACHE.exists():
        print(f"  Loading cached Pearson matrices from {PEARSON_CACHE} …")
        cache = np.load(PEARSON_CACHE, allow_pickle=True)
        cached_compounds = cache["compounds"].tolist()
        if cached_compounds == compounds:
            corr_matrix = cache["corr_matrix"]
            count_matrix = cache["count_matrix"]
            print("  Done (from cache).")
            return corr_matrix, count_matrix, mat_f, compounds
        print("  Cache compound list mismatch — recomputing …")

    n = len(compounds)
    corr_matrix = np.full((n, n), np.nan, dtype=np.float32)
    count_matrix = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        xi = mat_f[:, i]
        for j in range(i, n):
            xj = mat_f[:, j]
            mask = ~(np.isnan(xi) | np.isnan(xj))
            cnt = int(mask.sum())
            count_matrix[i, j] = count_matrix[j, i] = cnt
            if cnt >= 3:
                r, _ = stats.pearsonr(xi[mask], xj[mask])
                corr_matrix[i, j] = corr_matrix[j, i] = float(r)
        if i % 100 == 0:
            print(f"  Pearson {i}/{n} …")

    print("  Done.")
    np.savez_compressed(
        PEARSON_CACHE,
        corr_matrix=corr_matrix,
        count_matrix=count_matrix,
        compounds=np.array(compounds),
    )
    print(f"  Pearson matrices cached → {PEARSON_CACHE}")
    return corr_matrix, count_matrix, mat_f, compounds


# ────────────────────────────────────────────────────────────────────────────
# 3.  Build the protein-protein interactome graph (fixed topology, MOESM6)
# ────────────────────────────────────────────────────────────────────────────

def load_ppi_graph(fc_df: pd.DataFrame) -> tuple[PyGData, list[str]]:
    """
    Load the protein-protein Pearson correlation network (MOESM6) and
    intersect with proteins present in the FC matrix.

    Returns
    -------
    ppi_graph      : PyGData with placeholder zero node features (shape N×1);
                     real features are filled per-compound at training time.
    prot_order     : list of protein names (length N) — the canonical node order.
    """
    print("Loading protein-protein interactome (MOESM6) …")
    wb = openpyxl.load_workbook(PPI_FILE, read_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    fc_proteins = set(fc_df.index.tolist())
    src_list, dst_list, w_list = [], [], []
    prot_set: set[str] = set()
    for row in rows[1:]:
        p1, p2, r = str(row[0]), str(row[1]), float(row[2])
        if p1 in fc_proteins and p2 in fc_proteins:
            prot_set.add(p1)
            prot_set.add(p2)
            src_list.append(p1)
            dst_list.append(p2)
            w_list.append(r)

    # Canonical protein ordering (sorted for reproducibility)
    prot_order = sorted(prot_set)
    prot_idx = {p: i for i, p in enumerate(prot_order)}
    n = len(prot_order)

    # Build bidirectional edge index
    srcs = [prot_idx[p] for p in src_list]
    dsts = [prot_idx[p] for p in dst_list]
    srcs_bi = srcs + dsts
    dsts_bi = dsts + srcs
    ws_bi = w_list + w_list

    edge_index = torch.tensor([srcs_bi, dsts_bi], dtype=torch.long)
    edge_attr  = torch.tensor(ws_bi, dtype=torch.float32).unsqueeze(1)  # (E, 1)
    x          = torch.zeros(n, 1, dtype=torch.float32)                 # placeholder

    ppi_graph = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr)
    ppi_graph.num_nodes = n
    print(
        f"  PPI graph: {n} proteins (FC∩PPI overlap), "
        f"{len(w_list)} undirected edges, continuous Pearson r weights"
    )
    return ppi_graph, prot_order


# ────────────────────────────────────────────────────────────────────────────
# 4.  Dataset
# ────────────────────────────────────────────────────────────────────────────

class ProteomeCLIPDataset(Dataset):
    """
    Each item is a (smiles_str, fc_node_feats) pair where fc_node_feats is a
    (N_prot, 1) float32 tensor — the z-scored log2FC value for each protein
    node in the PPI graph for that compound (NaN → 0).
    """

    def __init__(
        self,
        smiles_list: list[str],
        compound_keys: list[str],      # FC matrix column names for each compound
        fc_df: pd.DataFrame,           # proteins × compounds, NaN for missing
        prot_order: list[str],         # canonical protein node order (length N_prot)
    ):
        self.smiles = smiles_list

        # Subset FC matrix to PPI proteins in canonical order, fill NaN→0
        fc_sub = fc_df.reindex(index=prot_order)   # (N_prot, N_compounds)
        fc_sub = fc_sub[compound_keys]              # (N_prot, B_compounds)

        # Z-score each protein row across the compounds present in this dataset
        fc_np = fc_sub.values.astype(np.float32)   # (N_prot, B)
        mu    = np.nanmean(fc_np, axis=1, keepdims=True)
        sigma = np.nanstd(fc_np, axis=1, keepdims=True) + 1e-8
        fc_z  = (fc_np - mu) / sigma                # (N_prot, B)
        fc_z  = np.nan_to_num(fc_z, nan=0.0)

        # Shape: (B, N_prot, 1) — one feature vector per compound
        fc_z_T = fc_z.T[:, :, np.newaxis]          # (B, N_prot, 1)
        self.fc_tensors = torch.from_numpy(fc_z_T.astype(np.float32))  # (B, N_prot, 1)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        return self.smiles[idx], self.fc_tensors[idx]  # (smiles, (N_prot, 1))


# ────────────────────────────────────────────────────────────────────────────
# 5.  Chemical Tower: frozen ChemBERTa + MLP projection head
# ────────────────────────────────────────────────────────────────────────────

class ChemicalTower(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, hidden_dim: int = MLP_HIDDEN):
        super().__init__()
        print(f"  Loading ChemBERTa from '{CHEMBERTA_MODEL}' …")
        self.tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
        self.encoder = AutoModel.from_pretrained(CHEMBERTA_MODEL)

        # Freeze all base weights completely
        for param in self.encoder.parameters():
            param.requires_grad = False

        bert_hidden = self.encoder.config.hidden_size  # 384 for 77M-MTR

        # Learnable 2-layer MLP projection head
        self.projector = nn.Sequential(
            nn.Linear(bert_hidden, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def tokenize(self, smiles_batch: list[str]) -> dict:
        return self.tokenizer(
            smiles_batch,
            padding=True,
            truncation=True,
            max_length=MAX_SMILES_LEN,
            return_tensors="pt",
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # encoder is frozen: run in no_grad to save memory
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token representation  (B, hidden)
        cls = out.last_hidden_state[:, 0, :]
        return self.projector(cls)          # (B, D)


# ────────────────────────────────────────────────────────────────────────────
# 6.  Proteome Tower: multi-layer GAT on the compound graph + global pool + MLP
# ────────────────────────────────────────────────────────────────────────────

class ProteomeTower(nn.Module):
    """
    Trace Encoder: three-layer GAT over the protein-protein interactome.

    For each compound, node features are the scalar log2FC value at each
    protein (z-scored, NaN→0).  The fixed graph topology comes from the
    MOESM6 protein-protein Pearson correlation network; edge weights are
    continuous Pearson r with no hard threshold.  The GAT uses multi-head
    attention to dynamically weight edges during backpropagation.  After
    message-passing, global mean pooling over all protein nodes collapses
    the graph to a single compound embedding, projected to D=128.
    """

    def __init__(
        self,
        in_dim: int = GNN_IN_DIM,  # 1 — scalar FC value per protein
        hidden_dim: int = GNN_HIDDEN,
        heads: int = GNN_HEADS,
        latent_dim: int = LATENT_DIM,
        mlp_hidden: int = MLP_HIDDEN,
    ):
        super().__init__()
        self.conv1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=1,
            concat=True,
            dropout=0.1,
        )
        self.conv2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=1,
            concat=True,
            dropout=0.1,
        )
        self.conv3 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            edge_dim=1,
            concat=False,
            dropout=0.1,
        )
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim * heads)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # 2-layer MLP projection head
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Linear(mlp_hidden, latent_dim),
        )

    def forward(
        self,
        x: torch.Tensor,            # (N_graph, P)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr: torch.Tensor,    # (E, 1)
        batch: torch.Tensor,        # (N_graph,) batch assignment vector
    ) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.conv3(h, edge_index, edge_attr=edge_attr)
        h = self.norm3(h)
        h = F.gelu(h)
        # Global mean pool: (N_graph, hidden) → (B, hidden)
        h_pooled = global_mean_pool(h, batch)
        return self.projector(h_pooled)    # (B, D)


# ────────────────────────────────────────────────────────────────────────────
# 7.  CLIP model: ties both towers + learnable temperature
# ────────────────────────────────────────────────────────────────────────────

class ProteomeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.chem_tower = ChemicalTower()
        self.prot_tower = ProteomeTower()
        # Learnable log-temperature: initialised to log(1/0.07) ≈ 2.659
        self.log_temp = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    @property
    def temperature(self) -> torch.Tensor:
        # Cap at log(100) = 4.605 to prevent instability
        return torch.exp(torch.clamp(self.log_temp, max=math.log(100.0)))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_chem = self.chem_tower(input_ids, attention_mask)        # (B, D)
        z_prot = self.prot_tower(
            graph_x, graph_edge_index, graph_edge_attr, batch
        )  # (B, D)
        # L2-normalise both embeddings
        z_chem = F.normalize(z_chem, dim=-1)
        z_prot = F.normalize(z_prot, dim=-1)
        return z_chem, z_prot


# ────────────────────────────────────────────────────────────────────────────
# 8.  Symmetric InfoNCE loss
# ────────────────────────────────────────────────────────────────────────────

def info_nce_loss(
    z_chem: torch.Tensor,   # (B, D)  L2-normalised
    z_prot: torch.Tensor,   # (B, D)  L2-normalised
    temperature: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric InfoNCE / CLIP contrastive loss.
    Each (z_chem[i], z_prot[i]) is a positive pair; all off-diagonal
    entries within the batch are negatives.
    """
    B = z_chem.shape[0]
    # Scaled cosine similarity matrix  (B, B)
    logits = (z_chem @ z_prot.T) * temperature
    labels = torch.arange(B, device=z_chem.device)
    loss_chem2prot = F.cross_entropy(logits, labels)
    loss_prot2chem = F.cross_entropy(logits.T, labels)
    return (loss_chem2prot + loss_prot2chem) / 2.0


# ────────────────────────────────────────────────────────────────────────────
# 9.  Retrieval evaluation  (Top-K accuracy)
# ────────────────────────────────────────────────────────────────────────────

def make_collate(ppi_graph: PyGData):
    """
    Returns a collate_fn that stamps per-compound FC vectors onto the fixed
    PPI graph topology, producing a PyGBatch of B protein graphs.
    """
    ei = ppi_graph.edge_index   # shared, will be replicated by PyGBatch
    ea = ppi_graph.edge_attr

    def collate(batch: list) -> tuple[list[str], PyGBatch]:
        smiles_list = [item[0] for item in batch]
        graphs = [
            PyGData(x=item[1], edge_index=ei, edge_attr=ea)
            for item in batch
        ]
        return smiles_list, PyGBatch.from_data_list(graphs)

    return collate


@torch.no_grad()
def retrieval_accuracy(
    model: ProteomeCLIP,
    loader: DataLoader,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """
    Chemistry-to-proteome retrieval: for each compound in the val set,
    rank all val-set proteome embeddings by cosine similarity to the
    compound's chemistry embedding, and report hit-rate at rank K.
    """
    model.eval()
    all_z_chem, all_z_prot = [], []
    for smiles_batch, pyg_batch in loader:
        enc = model.chem_tower.tokenize(list(smiles_batch))
        input_ids = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)
        pyg_batch = pyg_batch.to(DEVICE)
        z_c, z_p = model(
            input_ids, attn_mask,
            pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr, pyg_batch.batch,
        )
        all_z_chem.append(z_c.cpu())
        all_z_prot.append(z_p.cpu())

    zc = torch.cat(all_z_chem)   # (N_val, D)
    zp = torch.cat(all_z_prot)   # (N_val, D)
    sim = zc @ zp.T               # (N_val, N_val)
    labels = torch.arange(len(zc))
    results = {}
    for k in ks:
        top_k = sim.topk(k, dim=1).indices           # (N_val, k)
        hit = (top_k == labels.unsqueeze(1)).any(1)  # (N_val,)
        results[f"top{k}"] = hit.float().mean().item()
    return results


# ────────────────────────────────────────────────────────────────────────────
# 10.  Training loop
# ────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: ProteomeCLIP,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for smiles_batch, pyg_batch in loader:
        enc = model.chem_tower.tokenize(list(smiles_batch))
        input_ids = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)
        pyg_batch = pyg_batch.to(DEVICE)
        z_c, z_p = model(
            input_ids, attn_mask,
            pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr, pyg_batch.batch,
        )
        loss = info_nce_loss(z_c, z_p, model.temperature)
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        b = len(smiles_batch)
        total_loss += loss.item() * b
        n_samples += b
    return total_loss / n_samples


@torch.no_grad()
def eval_one_epoch(
    model: ProteomeCLIP,
    loader: DataLoader,
) -> float:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    for smiles_batch, pyg_batch in loader:
        enc = model.chem_tower.tokenize(list(smiles_batch))
        input_ids = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)
        pyg_batch = pyg_batch.to(DEVICE)
        z_c, z_p = model(
            input_ids, attn_mask,
            pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr, pyg_batch.batch,
        )
        loss = info_nce_loss(z_c, z_p, model.temperature)
        b = len(smiles_batch)
        total_loss += loss.item() * b
        n_samples += b
    return total_loss / n_samples


# ────────────────────────────────────────────────────────────────────────────
# 11.  Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)

    # ── 11a. Load raw data ───────────────────────────────────────────────────
    fc_df = load_fc_matrix()
    smiles_df = load_smiles_table()

    # ── 11b. Load protein-protein interactome ────────────────────────────────
    ppi_graph, prot_order = load_ppi_graph(fc_df)

    # ── 11c. Match SMILES ↔ FC compounds ────────────────────────────────────
    fc_col_set = set(fc_df.columns.tolist())
    valid_smiles, valid_fc_keys = [], []
    for _, row in smiles_df.iterrows():
        key = row["fc_key"]
        if key in fc_col_set:
            smi = str(row["SMILES"]).strip()
            if smi and smi.lower() != "nan":
                valid_smiles.append(smi)
                valid_fc_keys.append(key)

    print(f"\nMatched {len(valid_smiles)} SMILES ↔ FC profiles")

    # ── 11d. Dataset + 80/20 split ───────────────────────────────────────────
    dataset = ProteomeCLIPDataset(valid_smiles, valid_fc_keys, fc_df, prot_order)
    n_total = len(dataset)
    n_val = max(1, int(n_total * VAL_FRAC))
    n_train = n_total - n_val
    indices_all = list(range(n_total))
    random.shuffle(indices_all)
    train_idx, val_idx = indices_all[:n_train], indices_all[n_train:]
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    print(f"  Train: {len(train_set)}  |  Val: {len(val_set)}")

    collate_fn = make_collate(ppi_graph)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, drop_last=False, collate_fn=collate_fn,
    )

    # ── 11e. Model ───────────────────────────────────────────────────────────
    print("\nBuilding model …")
    model = ProteomeCLIP().to(DEVICE)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.chem_tower.encoder.parameters())
    print(f"  Trainable params : {n_trainable:,}")
    print(f"  Frozen ChemBERTa : {n_frozen:,}")

    # ── 11g. Optimiser + scheduler ───────────────────────────────────────────
    # Only pass trainable parameters to the optimiser
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=EPOCHS, eta_min=LR * 0.01,
    )

    # ── 11h. Training loop ───────────────────────────────────────────────────
    best_val_loss = float("inf")
    history = []
    print(f"\nTraining for {EPOCHS} epochs …\n")
    header = f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Top-1':>6}  {'Top-5':>6}  {'Temp':>6}"
    print(header)
    print("─" * len(header))

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser)
        val_loss = eval_one_epoch(model, val_loader)
        retrieval = retrieval_accuracy(model, val_loader, ks=(1, 5))
        scheduler.step()

        temp_val = model.temperature.item()
        history.append(
            dict(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                top1=retrieval["top1"],
                top5=retrieval["top5"],
                temperature=temp_val,
            )
        )
        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
            f"{retrieval['top1']:>6.3f}  {retrieval['top5']:>6.3f}  "
            f"{temp_val:>6.3f}"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "val_loss": val_loss,
                    "top1": retrieval["top1"],
                    "top5": retrieval["top5"],
                    "temperature": temp_val,
                    "prot_order": prot_order,
                    "valid_smiles": valid_smiles,
                    "valid_fc_keys": valid_fc_keys,
                },
                CHECKPOINT_PATH,
            )
            print(f"  ✓ New best checkpoint saved → {CHECKPOINT_PATH}")

    # ── 11i. Final summary ───────────────────────────────────────────────────
    best = min(history, key=lambda r: r["val_loss"])
    print(f"\n{'─'*60}")
    print(f"Best epoch  : {best['epoch']}")
    print(f"Val loss    : {best['val_loss']:.4f}")
    print(f"Top-1 acc   : {best['top1']:.3f}")
    print(f"Top-5 acc   : {best['top5']:.3f}")
    print(f"Checkpoint  : {CHECKPOINT_PATH}")

    # Save history CSV
    pd.DataFrame(history).to_csv("training_history.csv", index=False)
    print("Training history saved → training_history.csv")


if __name__ == "__main__":
    main()
