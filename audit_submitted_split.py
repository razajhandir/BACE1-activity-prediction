"""
Reproduces and documents the submitted scaffold split, including the
82% active test-set issue raised by Reviewer 2.

This script uses the ORIGINAL merged dataset and the original greedy
Bemis-Murcko splitting logic so the response letter can state exactly what
happened in the submitted analysis.
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from revision_config import CFG
from revision_core import ensure_dir, majority_baseline

OUT = Path(CFG.outdir) / "02_submitted_split_audit"
ensure_dir(OUT)

def canonicalize(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)

def load_original():
    df = pd.read_csv(CFG.original_merged_csv)

    # The submitted merged dataset may store molecular structures in either
    # a MoleculeNet-style "mol" column or a generic "smiles" column.
    if "smiles" not in df.columns:
        if "mol" in df.columns:
            df["smiles"] = df["mol"]
        else:
            raise ValueError(
                "No molecular structure column found. Expected 'smiles' or 'mol'. "
                f"Available columns: {df.columns.tolist()}"
            )

    df["smiles"] = df["smiles"].map(canonicalize)
    df = df.dropna(subset=["smiles", "Class", "pIC50"]).copy()
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
    df["pIC50"] = pd.to_numeric(df["pIC50"], errors="coerce")
    df = df.dropna(subset=["Class", "pIC50"])
    df["Class"] = df["Class"].astype(int)

    # Reproduce the submitted pipeline's duplicate handling.
    df = df.groupby("smiles", as_index=False).agg(
        {"Class": lambda x: int(round(float(np.mean(x)))), "pIC50": "median"}
    )
    return df

def original_scaffold_split(df, test_size=0.15, val_size=0.15, seed=42):
    scaffolds = {}
    for idx, smi in enumerate(df["smiles"].values):
        mol = Chem.MolFromSmiles(smi)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        ) if mol else ""
        scaffolds.setdefault(scaffold, []).append(idx)

    rng = random.Random(seed)
    scaffold_sets = list(scaffolds.values())
    rng.shuffle(scaffold_sets)
    scaffold_sets = sorted(scaffold_sets, key=len, reverse=True)

    n = len(df)
    n_test = int(test_size * n)
    n_val = int(val_size * n)
    train_idx, val_idx, test_idx = [], [], []
    for s in scaffold_sets:
        if len(test_idx) + len(s) <= n_test:
            test_idx.extend(s)
        elif len(val_idx) + len(s) <= n_val:
            val_idx.extend(s)
        else:
            train_idx.extend(s)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

def main():
    df = load_original()
    tr, va, te = original_scaffold_split(df, 0.15, 0.15, 42)
    rows = []
    for name, idx in [("train", tr), ("validation", va), ("test", te)]:
        y = df.iloc[idx]["Class"].values
        rows.append({
            "split": name, "n": len(y), "active": int(y.sum()),
            "inactive": int(len(y)-y.sum()), "active_fraction": y.mean()
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "submitted_split_class_composition.csv", index=False)

    maj = majority_baseline(df.iloc[te]["Class"].values)
    pd.DataFrame([maj]).to_csv(OUT / "submitted_test_majority_baseline.csv", index=False)

    split = pd.DataFrame({"row_index": np.arange(len(df)), "smiles": df["smiles"], "Class": df["Class"]})
    split["split"] = "train"
    split.loc[va, "split"] = "validation"
    split.loc[te, "split"] = "test"
    split.to_csv(OUT / "submitted_split_indices.csv", index=False)

    print(tab.to_string(index=False))
    print("\nMajority baseline on submitted test set:")
    print(pd.DataFrame([maj]).to_string(index=False))

if __name__ == "__main__":
    main()
