"""
Shared utilities for the BACE1 reviewer-revision experiments.

The code intentionally keeps the revised analyses separate from the submitted
results. It supports:
- chemically standardized identity handling,
- scaffold-disjoint train/validation/test splits,
- classical fingerprint baselines,
- the original Edge-GNN / PI-EGNN architecture,
- ablation modes for the path component,
- manuscript-ready metrics and reproducibility outputs.
"""
import os
import math
import json
import random
import copy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors, Crippen, Lipinski, QED, rdMolDescriptors,
    rdFingerprintGenerator
)
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import softmax
    HAS_PYG = True
except Exception:
    HAS_PYG = False


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if HAS_PYG:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def canonical_smiles(smi: str, isomeric=True) -> Optional[str]:
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def scaffold_smiles(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_group_split(df: pd.DataFrame, seed: int, test_size=0.15, val_size=0.15):
    """
    Randomized Bemis-Murcko scaffold holdout.

    No scaffold appears in more than one partition. The function uses
    GroupShuffleSplit twice, so repeated seeds genuinely produce different
    scaffold-disjoint partitions. This is preferable for repeated uncertainty
    assessment to the submitted greedy split, whose random seed mainly affects
    ties between equally sized scaffold groups.
    """
    groups = np.array([scaffold_smiles(s) for s in df["smiles"].astype(str)])
    y = df["Class"].astype(int).values

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(np.arange(len(df)), y, groups))

    rel_val = val_size / (1.0 - test_size)
    trainval_groups = groups[trainval_idx]
    trainval_y = y[trainval_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 10000)
    tr_rel, va_rel = next(gss_val.split(trainval_idx, trainval_y, trainval_groups))
    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[va_rel]

    # Hard leakage check.
    gs = {
        "train": set(groups[train_idx]),
        "val": set(groups[val_idx]),
        "test": set(groups[test_idx]),
    }
    assert not (gs["train"] & gs["val"])
    assert not (gs["train"] & gs["test"])
    assert not (gs["val"] & gs["test"])

    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def split_audit_table(df, train_idx, val_idx, test_idx):
    rows = []
    for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        y = df.iloc[idx]["Class"].astype(int).values
        rows.append({
            "split": name,
            "n": len(idx),
            "active": int(y.sum()),
            "inactive": int(len(y) - y.sum()),
            "active_fraction": float(y.mean()) if len(y) else np.nan,
            "n_scaffolds": len(set(scaffold_smiles(s) for s in df.iloc[idx]["smiles"]))
        })
    return pd.DataFrame(rows)


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_test": len(y_true),
        "test_active_fraction": float(y_true.mean()),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    except Exception:
        out["pr_auc"] = np.nan
    return out


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def majority_baseline(y_true):
    y_true = np.asarray(y_true).astype(int)
    maj = int(y_true.mean() >= 0.5)
    # Constant score: AUROC=0.5 where both classes are present; AP=prevalence.
    score = np.full(len(y_true), float(y_true.mean()))
    pred = np.full(len(y_true), maj)
    m = classification_metrics(y_true, score, threshold=0.5)
    # Ensure thresholded output is exactly majority class if prevalence <0.5.
    if maj == 0:
        score[:] = 0.0
        m = classification_metrics(y_true, score, threshold=0.5)
    m["model"] = "Majority_class"
    return m


# -------------------------
# Fingerprints
# -------------------------
def _fp_to_np(fp, nbits):
    arr = np.zeros((nbits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def fingerprint_matrix(smiles, kind="morgan", nbits=2048, radius=2):
    kind = kind.lower()
    if kind == "morgan":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    elif kind in {"rdk", "rdkpath", "path"}:
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(minPath=1, maxPath=7, fpSize=nbits)
    elif kind in {"atompair", "atom_pair"}:
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nbits)
    elif kind in {"torsion", "topologicaltorsion", "topological_torsion"}:
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=nbits)
    else:
        raise ValueError(f"Unknown fingerprint kind: {kind}")

    X = np.zeros((len(smiles), nbits), dtype=np.uint8)
    valid = np.zeros(len(smiles), dtype=bool)
    fps = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            fps.append(None)
            continue
        fp = gen.GetFingerprint(mol)
        X[i] = _fp_to_np(fp, nbits)
        valid[i] = True
        fps.append(fp)
    return X, fps, valid


def train_rf_classifier(X, y, seed):
    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
        min_samples_leaf=1,
    ).fit(X, y)


# -------------------------
# Molecular graph
# -------------------------
ATOM_LIST = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Se", "other"]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
]


def one_hot(x, choices):
    return [int(x == c) for c in choices]


def atom_features(atom):
    symbol = atom.GetSymbol()
    if symbol not in ATOM_LIST:
        symbol = "other"
    feats = []
    feats += one_hot(symbol, ATOM_LIST)
    feats += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    feats += one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    feats += one_hot(str(atom.GetHybridization()), ["SP", "SP2", "SP3", "SP3D", "SP3D2"])
    feats += [
        int(atom.GetIsAromatic()), int(atom.IsInRing()),
        atom.GetTotalNumHs(), atom.GetTotalValence(), atom.GetMass() / 100.0
    ]
    return feats


def bond_features(bond):
    feats = one_hot(bond.GetBondType(), BOND_LIST)
    feats += [int(bond.GetIsConjugated()), int(bond.IsInRing()),
              int(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)]
    return feats


def enumerate_simple_paths(mol, max_len=2, max_paths_per_pair=8):
    n = mol.GetNumAtoms()
    adj = [[] for _ in range(n)]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    paths = []
    for start in range(n):
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) > 1:
                paths.append(path[:])
            if len(path) - 1 >= max_len:
                continue
            for nb in adj[node]:
                if nb not in path:
                    stack.append((nb, path + [nb]))

    pair_count, kept = {}, []
    for p in paths:
        key = (p[0], p[-1])
        if pair_count.get(key, 0) < max_paths_per_pair:
            kept.append(p)
            pair_count[key] = pair_count.get(key, 0) + 1
    return kept


def mol_to_graph(smi, y_cls=None, y_reg=None, max_path_len=2):
    if not HAS_PYG:
        raise ImportError("torch-geometric is required.")
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float)

    paths = enumerate_simple_paths(mol, max_len=max_path_len)
    srcs, dsts, pfeats = [], [], []
    for p in paths:
        srcs.append(p[0]); dsts.append(p[-1])
        length = len(p) - 1
        aromatic_atoms = sum(int(mol.GetAtomWithIdx(i).GetIsAromatic()) for i in p) / len(p)
        ring_atoms = sum(int(mol.GetAtomWithIdx(i).IsInRing()) for i in p) / len(p)
        hetero_atoms = sum(int(mol.GetAtomWithIdx(i).GetSymbol() not in ["C", "H"]) for i in p) / len(p)
        conj = arom = 0
        for a, b in zip(p[:-1], p[1:]):
            bond = mol.GetBondBetweenAtoms(a, b)
            if bond is not None:
                conj += int(bond.GetIsConjugated())
                arom += int(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC)
        pfeats.append([
            length, aromatic_atoms, ring_atoms, hetero_atoms,
            conj / max(length, 1), arom / max(length, 1)
        ])
    if srcs:
        path_index = torch.tensor([srcs, dsts], dtype=torch.long)
        path_attr = torch.tensor(pfeats, dtype=torch.float)
    else:
        path_index = torch.empty((2, 0), dtype=torch.long)
        path_attr = torch.empty((0, 6), dtype=torch.float)

    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
             path_index=path_index, path_attr=path_attr)
    if y_cls is not None:
        d.y_cls = torch.tensor([int(y_cls)], dtype=torch.long)

    # PyG batching requires every Data object in the same batch to expose the
    # same attribute keys.  The revised classification dataset can legitimately
    # contain compounds supported only by censored IC50 evidence, so pIC50 may
    # be missing for some rows.  Store NaN rather than omitting y_reg; the
    # classification loss never reads y_reg, while the dedicated regression
    # dataset contains finite pIC50 values.
    if y_reg is None or pd.isna(y_reg):
        d.y_reg = torch.tensor([float("nan")], dtype=torch.float)
    else:
        d.y_reg = torch.tensor([float(y_reg)], dtype=torch.float)

    d.smiles = str(smi)
    return d


def build_graph_dataset(df, max_path_len=2):
    data = []
    kept_rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        d = mol_to_graph(row["smiles"], row.get("Class", None), row.get("pIC50", None),
                         max_path_len=max_path_len)
        if d is not None:
            data.append(d)
            kept_rows.append(i)
    if len(data) != len(df):
        raise RuntimeError(
            f"{len(df)-len(data)} molecules failed graph conversion. "
            "Re-curate the dataset rather than silently shifting indices."
        )
    return data


if HAS_PYG:
    class EdgeAttentionLayer(nn.Module):
        def __init__(self, hidden_dim, edge_dim, dropout=0.1):
            super().__init__()
            self.msg = nn.Linear(hidden_dim + edge_dim, hidden_dim)
            self.att = nn.Linear(2 * hidden_dim + edge_dim, 1)
            self.self_lin = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = dropout

        def forward(self, x, edge_index, edge_attr):
            if edge_index.numel() == 0:
                return self.self_lin(x)
            src, dst = edge_index
            score = F.leaky_relu(self.att(torch.cat([x[src], x[dst], edge_attr], -1))).squeeze(-1)
            alpha = softmax(score, dst)
            msg = self.msg(torch.cat([x[src], edge_attr], -1)) * alpha.unsqueeze(-1)
            out = torch.zeros_like(x)
            out.index_add_(0, dst, msg)
            return F.dropout(F.relu(out + self.self_lin(x)), p=self.dropout, training=self.training)

    class PathAggregationLayer(nn.Module):
        def __init__(self, hidden_dim, path_dim, dropout=0.1, mode="learned"):
            super().__init__()
            if mode not in {"learned", "uniform"}:
                raise ValueError("Path mode must be 'learned' or 'uniform'.")
            self.mode = mode
            self.action = nn.Sequential(
                nn.Linear(2 * hidden_dim + path_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.path_msg = nn.Sequential(
                nn.Linear(hidden_dim + path_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.tau = nn.Parameter(torch.tensor(1.0))
            self.dropout = dropout

        def forward(self, x, path_index, path_attr):
            if path_index.numel() == 0:
                return torch.zeros_like(x)
            src, dst = path_index
            if self.mode == "learned":
                tau = torch.clamp(F.softplus(self.tau), min=0.05, max=10.0)
                S = self.action(torch.cat([x[src], x[dst], path_attr], -1)).squeeze(-1)
                omega = softmax(-S / tau, dst)
            else:
                # Equal contribution among all paths terminating at each destination.
                zeros = torch.zeros(dst.shape[0], device=x.device, dtype=x.dtype)
                omega = softmax(zeros, dst)

            msg = self.path_msg(torch.cat([x[src], path_attr], -1)) * omega.unsqueeze(-1)
            out = torch.zeros_like(x)
            out.index_add_(0, dst, msg)
            return F.dropout(out, p=self.dropout, training=self.training)

    class PIEGNN(nn.Module):
        def __init__(self, node_dim, edge_dim, path_dim, hidden_dim=128, layers=3,
                     dropout=0.35, task="classification", path_mode="learned"):
            super().__init__()
            if path_mode not in {"learned", "uniform", "none"}:
                raise ValueError("path_mode: learned, uniform, or none")
            self.task = task
            self.path_mode = path_mode
            self.node_enc = nn.Linear(node_dim, hidden_dim)
            self.edge_enc = nn.Linear(edge_dim, hidden_dim // 2)
            self.path_enc = nn.Linear(path_dim, hidden_dim // 2)
            self.local_layers = nn.ModuleList([
                EdgeAttentionLayer(hidden_dim, hidden_dim // 2, dropout)
                for _ in range(layers)
            ])
            self.path_layers = nn.ModuleList([
                PathAggregationLayer(hidden_dim, hidden_dim // 2, dropout,
                                     mode="learned" if path_mode == "learned" else "uniform")
                for _ in range(layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
            self.readout_att = nn.Linear(hidden_dim, 1)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
            )

        def forward(self, data):
            x = F.relu(self.node_enc(data.x))
            ea = (F.relu(self.edge_enc(data.edge_attr)) if data.edge_attr.numel()
                  else data.edge_attr.new_empty((0, self.edge_enc.out_features)))
            pa = (F.relu(self.path_enc(data.path_attr)) if data.path_attr.numel()
                  else data.path_attr.new_empty((0, self.path_enc.out_features)))

            for local, path_layer, norm in zip(self.local_layers, self.path_layers, self.norms):
                h_local = local(x, data.edge_index, ea)
                if self.path_mode == "none":
                    x = norm(x + h_local)
                else:
                    h_path = path_layer(x, data.path_index, pa)
                    x = norm(x + h_local + h_path)

            batch = data.batch
            score = self.readout_att(x).squeeze(-1)
            beta = softmax(score, batch)
            graph_emb = torch.zeros(int(batch.max().item()) + 1, x.size(-1), device=x.device)
            graph_emb.index_add_(0, batch, x * beta.unsqueeze(-1))
            return self.head(graph_emb).squeeze(-1)


def _evaluate_gnn(model, loader, task, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            if task == "classification":
                ps.append(torch.sigmoid(out).cpu().numpy())
                ys.append(batch.y_cls.view(-1).cpu().numpy())
            else:
                ps.append(out.cpu().numpy())
                ys.append(batch.y_reg.view(-1).cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return y, p


def train_gnn_once(df, train_idx, val_idx, test_idx, cfg, seed,
                   path_mode="learned", max_path_len=None, task="classification",
                   save_dir=None, model_tag="PI_EGNN"):
    if not HAS_PYG:
        raise ImportError("Install torch and torch-geometric.")
    set_seed(seed)
    max_path_len = cfg.max_path_len if max_path_len is None else max_path_len
    data = build_graph_dataset(df, max_path_len=max_path_len)

    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    test_data = [data[i] for i in test_idx]
    tr_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    va_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False)
    te_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)

    sample = data[0]
    model = PIEGNN(
        node_dim=sample.x.size(-1), edge_dim=sample.edge_attr.size(-1),
        path_dim=sample.path_attr.size(-1), hidden_dim=cfg.hidden_dim,
        layers=cfg.layers, dropout=cfg.dropout, task=task, path_mode=path_mode
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if task == "classification":
        ytr = np.array([int(d.y_cls.item()) for d in train_data])
        pos_weight = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(pos_weight), dtype=torch.float, device=device)
        )
        monitor = "roc_auc"
    else:
        criterion = nn.MSELoss()
        monitor = "r2"

    best_score = -np.inf
    best_state = None
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in tr_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            target = (batch.y_cls.float().view(-1) if task == "classification"
                      else batch.y_reg.float().view(-1))
            loss = criterion(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        vy, vp = _evaluate_gnn(model, va_loader, task, device)
        vm = classification_metrics(vy, vp) if task == "classification" else regression_metrics(vy, vp)
        score = vm.get(monitor, -np.inf)
        if np.isfinite(score) and score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        history.append({"epoch": epoch, "loss": np.mean(losses), **{f"val_{k}": v for k, v in vm.items()}})
        if wait >= cfg.early_stopping_patience:
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    ty, tp = _evaluate_gnn(model, te_loader, task, device)
    metrics = classification_metrics(ty, tp) if task == "classification" else regression_metrics(ty, tp)
    metrics.update({
        "model": model_tag, "seed": seed, "path_mode": path_mode,
        "max_path_len": max_path_len, "best_epoch": best_epoch
    })

    if save_dir:
        ensure_dir(save_dir)
        pd.DataFrame(history).to_csv(Path(save_dir) / f"history_{model_tag}_seed{seed}.csv", index=False)
        pred = pd.DataFrame({
            "row_index": test_idx,
            "smiles": df.iloc[test_idx]["smiles"].values,
            "y_true": ty,
            "y_prob_or_pred": tp,
        })
        pred.to_csv(Path(save_dir) / f"pred_{model_tag}_seed{seed}.csv", index=False)
        torch.save({
            "model_state": best_state,
            "seed": seed,
            "path_mode": path_mode,
            "max_path_len": max_path_len,
            "config": vars(cfg),
            "metrics": metrics
        }, Path(save_dir) / f"{model_tag}_seed{seed}.pt")

    return metrics, ty, tp


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=123):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)])
    return np.quantile(means, alpha/2), np.quantile(means, 1-alpha/2)


def summarize_repeated(results_df, metric_cols=None):
    if metric_cols is None:
        metric_cols = [
            "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
            "f1", "mcc", "roc_auc", "pr_auc"
        ]
    rows = []
    for model, g in results_df.groupby("model"):
        row = {"model": model, "n_runs": len(g)}
        for m in metric_cols:
            if m not in g:
                continue
            vals = pd.to_numeric(g[m], errors="coerce").dropna().values
            if len(vals):
                lo, hi = bootstrap_ci(vals)
                row[f"{m}_mean"] = vals.mean()
                row[f"{m}_sd"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
                row[f"{m}_ci95_low"] = lo
                row[f"{m}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_wilcoxon_table(results_df, reference="PI_EGNN", metrics=("f1", "mcc", "roc_auc")):
    rows = []
    if not HAS_SCIPY:
        return pd.DataFrame([{"warning": "scipy not installed"}])
    ref = results_df[results_df["model"] == reference].set_index("seed")
    for model in sorted(results_df["model"].unique()):
        if model == reference:
            continue
        other = results_df[results_df["model"] == model].set_index("seed")
        common = sorted(set(ref.index) & set(other.index))
        for metric in metrics:
            if metric not in ref or metric not in other or len(common) < 2:
                continue
            a = ref.loc[common, metric].astype(float).values
            b = other.loc[common, metric].astype(float).values
            try:
                stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            except Exception:
                stat, p = np.nan, np.nan
            rows.append({
                "reference": reference, "comparison": model, "metric": metric,
                "n_paired": len(common), "reference_mean": np.mean(a),
                "comparison_mean": np.mean(b), "mean_difference": np.mean(a-b),
                "wilcoxon_stat": stat, "p_value": p
            })
    return pd.DataFrame(rows)
