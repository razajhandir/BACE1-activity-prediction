"""
Fingerprint-based applicability-domain (AD) analysis.

Definition used here:
- Morgan/Tanimoto representation.
- For every training molecule, compute its nearest OTHER training molecule.
- AD threshold = configurable lower percentile (default 5th percentile)
  of these nearest-neighbor similarities.
- A screened candidate is 'in-domain' when its maximum similarity to the
  training set is >= that threshold.

Also reports exact Bemis-Murcko scaffold match and the k nearest training
neighbors with labels/pIC50 where available.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from revision_config import CFG
from revision_core import ensure_dir, scaffold_smiles

TRAIN_CSV = Path(CFG.outdir) / "01_dataset" / "bace_classification_revised.csv"
OUT = Path(CFG.outdir) / "05_applicability_domain"
ensure_dir(OUT)

def read_candidates():
    choices = [
        Path(CFG.outdir) / "05b_zinc_rescreen" / "zinc_revision_screened_with_ids.csv",
        Path(CFG.original_screened_zinc_csv),
        Path(CFG.original_top20_csv),
    ]
    for p in choices:
        if p.exists():
            df = pd.read_csv(p)
            smi_col = next((c for c in ["smiles", "SMILES", "Smiles"] if c in df), None)
            if smi_col:
                df = df.rename(columns={smi_col: "smiles"})
                return df, str(p)
    raise FileNotFoundError(
        "No ZINC candidate file found. Set original_screened_zinc_csv or original_top20_csv "
        "in revision_config.py."
    )

def mol_fps(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=CFG.morgan_radius, fpSize=CFG.fp_bits)
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        fps.append(gen.GetFingerprint(m) if m else None)
    return fps

def main():
    train = pd.read_csv(TRAIN_CSV).dropna(subset=["smiles", "Class"]).reset_index(drop=True)
    cand, source = read_candidates()
    cand = cand.dropna(subset=["smiles"]).reset_index(drop=True)

    tfps = mol_fps(train["smiles"])
    cfps = mol_fps(cand["smiles"])
    if any(x is None for x in tfps):
        raise RuntimeError("Invalid molecule in training set.")

    # Full training nearest-neighbor calibration unless explicitly capped.
    n = len(tfps)
    use_idx = np.arange(n)
    if n > CFG.ad_calibration_max_n:
        rng = np.random.default_rng(42)
        use_idx = np.sort(rng.choice(n, CFG.ad_calibration_max_n, replace=False))

    nn_train = []
    for ii, i in enumerate(use_idx):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(tfps[i], tfps))
        sims[i] = -1.0
        nn_train.append(float(sims.max()))
        if (ii + 1) % 500 == 0:
            print("AD calibration:", ii + 1, "/", len(use_idx))
    threshold = float(np.percentile(nn_train, CFG.ad_percentile))

    with open(OUT / "ad_definition.json", "w") as f:
        json.dump({
            "representation": f"Morgan radius={CFG.morgan_radius}, nBits={CFG.fp_bits}",
            "similarity": "Tanimoto",
            "threshold_rule": f"{CFG.ad_percentile}th percentile of training nearest-neighbor similarity",
            "threshold": threshold,
            "n_training": len(train),
            "n_calibration": len(use_idx),
            "candidate_source": source,
        }, f, indent=2)

    train_scaf = [scaffold_smiles(s) for s in train["smiles"]]
    train_scaf_set = set(train_scaf)

    summary_rows = []
    neighbor_rows = []
    for ci, (smi, fp) in enumerate(zip(cand["smiles"], cfps)):
        if fp is None:
            summary_rows.append({"candidate_index": ci, "smiles": smi, "valid": False})
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, tfps))
        order = np.argsort(-sims)[:CFG.nearest_neighbors_k]
        cscaf = scaffold_smiles(smi)
        row = {
            "candidate_index": ci,
            "smiles": smi,
            "max_tanimoto": float(sims[order[0]]),
            "ad_threshold": threshold,
            "in_domain": bool(sims[order[0]] >= threshold),
            "exact_scaffold_in_training": bool(cscaf in train_scaf_set),
            "candidate_scaffold": cscaf,
        }
        # Preserve any existing ID / model-score columns.
        for col in cand.columns:
            if col != "smiles" and col not in row and np.isscalar(cand.loc[ci, col]):
                row[col] = cand.loc[ci, col]
        summary_rows.append(row)

        for rank, ti in enumerate(order, 1):
            nr = {
                "candidate_index": ci, "candidate_smiles": smi,
                "neighbor_rank": rank, "tanimoto": float(sims[ti]),
                "training_smiles": train.loc[ti, "smiles"],
                "training_Class": train.loc[ti, "Class"],
                "training_pIC50": train.loc[ti, "pIC50"] if "pIC50" in train else np.nan,
                "training_sources": train.loc[ti, "sources"] if "sources" in train else "",
            }
            neighbor_rows.append(nr)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "candidate_applicability_domain.csv", index=False)
    pd.DataFrame(neighbor_rows).to_csv(OUT / "candidate_nearest_training_neighbors.csv", index=False)
    pd.DataFrame({"training_nn_similarity": nn_train}).to_csv(
        OUT / "training_nn_similarity_distribution.csv", index=False
    )

    if "in_domain" in summary:
        print("AD threshold:", threshold)
        print("Candidate in-domain fraction:", summary["in_domain"].mean())
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
