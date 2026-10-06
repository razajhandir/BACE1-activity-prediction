"""
Reviewer-response dataset reconstruction.

Addresses:
- exact ChEMBL filtering criteria,
- salts/fragments, charges, tautomer and stereochemistry handling,
- repeated measurements,
- censored IC50 values,
- consistent pIC50 thresholding,
- separate classification and regression datasets,
- assay-scope reporting and within-compound variability.

Important:
The classification dataset may include unambiguous censored measurements.
The regression dataset uses exact '=' IC50 values only.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from revision_config import CFG
from revision_core import ensure_dir

OUT = Path(CFG.outdir) / "01_dataset"
ensure_dir(OUT)

def clean_relation(x):
    if pd.isna(x):
        return ""
    return str(x).strip().replace("'", "").replace('"', "")

def standardize_identity(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        if CFG.uncharge_for_identity:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        if CFG.canonicalize_tautomers:
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=CFG.keep_stereochemistry
        )
    except Exception:
        return None

def classify_relation(rel, value_nm):
    """
    Return 0/1 only when the relation proves the class at the 1000 nM threshold.
    Otherwise return NaN.
    """
    t = CFG.activity_threshold_nm
    if not np.isfinite(value_nm) or value_nm <= 0:
        return np.nan
    if rel == "=":
        return int(value_nm <= t)
    if rel in {">", ">="}:
        return 0 if value_nm >= t else np.nan
    if rel in {"<", "<="}:
        return 1 if value_nm <= t else np.nan
    return np.nan

def prepare_chembl():
    raw = pd.read_csv(CFG.chembl_raw_csv, sep=";", quotechar='"', low_memory=False)
    counts = [{"stage": "raw_export", "n_records": len(raw)}]

    q = raw.copy()
    q = q[q["Target ChEMBL ID"].astype(str) == CFG.target_chembl_id]
    counts.append({"stage": "target_id", "n_records": len(q)})
    q = q[q["Target Organism"].astype(str) == CFG.target_organism]
    counts.append({"stage": "human_target", "n_records": len(q)})
    q = q[q["Target Type"].astype(str) == CFG.target_type]
    counts.append({"stage": "single_protein_target", "n_records": len(q)})
    q = q[q["Standard Type"].astype(str) == "IC50"]
    counts.append({"stage": "IC50_only", "n_records": len(q)})
    q = q[q["Standard Units"].astype(str) == "nM"]
    counts.append({"stage": "nM_only", "n_records": len(q)})
    q["Standard Value"] = pd.to_numeric(q["Standard Value"], errors="coerce")
    q = q[q["Standard Value"].notna() & (q["Standard Value"] > 0)]
    counts.append({"stage": "positive_numeric_value", "n_records": len(q)})

    if CFG.exclude_nonempty_validity_comment and "Data Validity Comment" in q:
        q = q[q["Data Validity Comment"].isna() | (q["Data Validity Comment"].astype(str).str.strip() == "")]
        counts.append({"stage": "validity_comment_filtered", "n_records": len(q)})

    if "Assay Type" in q:
        q = q[q["Assay Type"].astype(str) == "B"]
        counts.append({"stage": "binding_assay_type_B", "n_records": len(q)})

    if CFG.assay_scope == "single_protein" and "BAO Label" in q:
        q = q[q["BAO Label"].astype(str).str.lower() == "single protein format"]
        counts.append({"stage": "BAO_single_protein_format", "n_records": len(q)})

    q["relation"] = q["Standard Relation"].map(clean_relation)
    q = q[q["relation"].isin(["=", ">", "<", ">=", "<="])]
    q["smiles_std"] = q["Smiles"].map(standardize_identity)
    q = q.dropna(subset=["smiles_std"])
    counts.append({"stage": "valid_standardized_structure", "n_records": len(q)})

    q["class_evidence"] = [
        classify_relation(r, v) for r, v in zip(q["relation"], q["Standard Value"])
    ]
    q["pIC50_exact"] = np.where(
        q["relation"].eq("="),
        -np.log10(q["Standard Value"].astype(float) * 1e-9),
        np.nan
    )

    keep_cols = [
        "Molecule ChEMBL ID", "smiles_std", "relation", "Standard Value",
        "pIC50_exact", "class_evidence", "Assay ChEMBL ID", "Assay Description",
        "Assay Type", "BAO Label", "Target ChEMBL ID"
    ]
    q[keep_cols].to_csv(OUT / "chembl_record_level_curated.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "chembl_filtering_counts.csv", index=False)
    return q

def prepare_moleculenet():
    df = pd.read_csv(CFG.bace_csv)
    if "mol" not in df.columns:
        raise ValueError("MoleculeNet BACE file must contain column 'mol'.")
    df["pIC50"] = pd.to_numeric(df["pIC50"], errors="coerce")
    df["smiles_std"] = df["mol"].map(standardize_identity)
    df = df.dropna(subset=["smiles_std", "pIC50"]).copy()
    # Recompute the binary label from the same threshold used for ChEMBL.
    df["class_recomputed"] = (df["pIC50"] >= -np.log10(CFG.activity_threshold_nm * 1e-9)).astype(int)
    if "Class" in df:
        df["original_Class"] = pd.to_numeric(df["Class"], errors="coerce")
        df["label_changed_by_rethresholding"] = (df["original_Class"] != df["class_recomputed"]).astype(int)
    df.to_csv(OUT / "moleculenet_standardized.csv", index=False)
    return df

def aggregate():
    chembl = prepare_chembl()
    molnet = prepare_moleculenet()

    # -------- Classification evidence --------
    evid = []
    for _, r in chembl[chembl["class_evidence"].notna()].iterrows():
        evid.append({
            "smiles": r["smiles_std"],
            "Class": int(r["class_evidence"]),
            "source": "ChEMBL",
            "evidence_type": f"IC50{r['relation']}{r['Standard Value']} nM",
            "pIC50": r["pIC50_exact"] if np.isfinite(r["pIC50_exact"]) else np.nan,
        })
    for _, r in molnet.iterrows():
        evid.append({
            "smiles": r["smiles_std"],
            "Class": int(r["class_recomputed"]),
            "source": "MoleculeNet_BACE",
            "evidence_type": "exact_pIC50",
            "pIC50": float(r["pIC50"]),
        })
    ev = pd.DataFrame(evid)
    ev.to_csv(OUT / "classification_evidence_all_records.csv", index=False)

    class_rows = []
    conflict_rows = []
    for smi, g in ev.groupby("smiles", sort=True):
        labels = sorted(g["Class"].dropna().astype(int).unique().tolist())
        exact_pic = pd.to_numeric(g["pIC50"], errors="coerce").dropna()
        if len(labels) != 1:
            conflict_rows.append({
                "smiles": smi,
                "labels_observed": ";".join(map(str, labels)),
                "n_evidence": len(g),
                "sources": ";".join(sorted(set(g["source"]))),
            })
            continue
        class_rows.append({
            "smiles": smi,
            "Class": labels[0],
            "pIC50": float(exact_pic.median()) if len(exact_pic) else np.nan,
            "n_class_evidence": len(g),
            "n_exact_values": len(exact_pic),
            "sources": ";".join(sorted(set(g["source"]))),
        })

    cls = pd.DataFrame(class_rows)
    conflicts = pd.DataFrame(conflict_rows)
    cls.to_csv(OUT / "bace_classification_revised.csv", index=False)
    conflicts.to_csv(OUT / "classification_conflicts_excluded.csv", index=False)

    # -------- Regression exact measurements only --------
    reg_records = []
    exact_ch = chembl[chembl["relation"].eq("=") & chembl["pIC50_exact"].notna()]
    for _, r in exact_ch.iterrows():
        reg_records.append({
            "smiles": r["smiles_std"], "pIC50": float(r["pIC50_exact"]),
            "source": "ChEMBL", "assay_id": r.get("Assay ChEMBL ID", "")
        })
    for _, r in molnet.iterrows():
        reg_records.append({
            "smiles": r["smiles_std"], "pIC50": float(r["pIC50"]),
            "source": "MoleculeNet_BACE", "assay_id": ""
        })
    rr = pd.DataFrame(reg_records)
    reg = rr.groupby("smiles").agg(
        pIC50=("pIC50", "median"),
        pIC50_mean=("pIC50", "mean"),
        pIC50_sd=("pIC50", "std"),
        n_exact_records=("pIC50", "size"),
        sources=("source", lambda x: ";".join(sorted(set(x)))),
        n_assays=("assay_id", lambda x: len(set(v for v in x if str(v).strip())))
    ).reset_index()
    reg["Class"] = (reg["pIC50"] >= 6.0).astype(int)
    reg.to_csv(OUT / "bace_regression_revised.csv", index=False)
    rr.to_csv(OUT / "regression_exact_record_level.csv", index=False)

    rep = reg[reg["n_exact_records"] >= 2].copy()
    variability = {
        "n_regression_molecules": int(len(reg)),
        "n_with_replicates": int(len(rep)),
        "median_within_molecule_sd_pIC50": float(rep["pIC50_sd"].median()) if len(rep) else np.nan,
        "mean_within_molecule_sd_pIC50": float(rep["pIC50_sd"].mean()) if len(rep) else np.nan,
        "classification_n": int(len(cls)),
        "classification_active": int(cls["Class"].sum()),
        "classification_inactive": int((cls["Class"] == 0).sum()),
        "classification_conflicts_excluded": int(len(conflicts)),
    }
    pd.DataFrame([variability]).to_csv(OUT / "dataset_summary_revised.csv", index=False)
    with open(OUT / "dataset_summary_revised.json", "w") as f:
        json.dump(variability, f, indent=2)

    print("\nRevision datasets saved to:", OUT)
    print(pd.DataFrame([variability]).to_string(index=False))
    return cls, reg

if __name__ == "__main__":
    aggregate()
