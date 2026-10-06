"""
Repeated scaffold-disjoint validation for the revised classification dataset.

Addresses reviewer requests for:
- explicit scaffold splitting,
- repeated runs / random seeds,
- mean +/- SD and 95% bootstrap CI,
- majority-class baseline,
- path-based fingerprint baselines,
- Edge-GNN versus PI-EGNN,
- paired significance tests.

Run AFTER 01_rebuild_datasets.py.
"""
from pathlib import Path
import pandas as pd
import numpy as np

from revision_config import CFG
from revision_core import (
    ensure_dir, set_seed, scaffold_group_split, split_audit_table,
    fingerprint_matrix, train_rf_classifier, classification_metrics,
    majority_baseline, train_gnn_once, summarize_repeated,
    paired_wilcoxon_table
)

IN = Path(CFG.outdir) / "01_dataset" / "bace_classification_revised.csv"
OUT = Path(CFG.outdir) / "03_repeated_validation"
ensure_dir(OUT)

def main():
    df = pd.read_csv(IN)
    df = df.dropna(subset=["smiles", "Class"]).reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)

    # Precompute classical fingerprints once.
    fp_kinds = {
        "RF_Morgan": "morgan",
        "RF_RDKPath": "rdkpath",
        "RF_AtomPair": "atompair",
        "RF_TopologicalTorsion": "torsion",
    }
    X = {}
    for model_name, kind in fp_kinds.items():
        X[model_name], _, valid = fingerprint_matrix(
            df["smiles"].tolist(), kind=kind,
            nbits=CFG.fp_bits, radius=CFG.morgan_radius
        )
        if not valid.all():
            raise RuntimeError(f"Invalid molecules remain for {model_name}.")

    all_results = []
    y_all = df["Class"].values

    for seed in CFG.seeds:
        print("\n" + "="*80)
        print("SEED", seed)
        tr, va, te = scaffold_group_split(
            df, seed, test_size=CFG.test_size, val_size=CFG.val_size
        )
        audit = split_audit_table(df, tr, va, te)
        audit["seed"] = seed
        audit.to_csv(OUT / f"split_audit_seed{seed}.csv", index=False)

        split = pd.DataFrame({
            "row_index": np.arange(len(df)), "smiles": df["smiles"],
            "Class": df["Class"], "split": "train"
        })
        split.loc[va, "split"] = "validation"
        split.loc[te, "split"] = "test"
        split.to_csv(OUT / f"split_indices_seed{seed}.csv", index=False)

        # Majority/no-skill baseline.
        m = majority_baseline(y_all[te])
        m["seed"] = seed
        all_results.append(m)

        # Classical fingerprint baselines.
        for model_name in fp_kinds:
            print("Training", model_name)
            clf = train_rf_classifier(X[model_name][tr], y_all[tr], seed)
            prob = clf.predict_proba(X[model_name][te])[:, 1]
            m = classification_metrics(y_all[te], prob)
            m.update({"model": model_name, "seed": seed})
            all_results.append(m)
            pd.DataFrame({
                "row_index": te, "smiles": df.iloc[te]["smiles"].values,
                "y_true": y_all[te], "y_prob": prob
            }).to_csv(OUT / f"pred_{model_name}_seed{seed}.csv", index=False)

        # Edge-GNN: same graph architecture, no path aggregation.
        print("Training Edge_GNN")
        m, _, _ = train_gnn_once(
            df, tr, va, te, CFG, seed,
            path_mode="none", max_path_len=CFG.max_path_len,
            task="classification", save_dir=OUT,
            model_tag="Edge_GNN"
        )
        all_results.append(m)

        # Full PI-EGNN.
        print("Training PI_EGNN")
        m, _, _ = train_gnn_once(
            df, tr, va, te, CFG, seed,
            path_mode="learned", max_path_len=CFG.max_path_len,
            task="classification", save_dir=OUT,
            model_tag="PI_EGNN"
        )
        all_results.append(m)

        pd.DataFrame(all_results).to_csv(OUT / "results_partial.csv", index=False)

    res = pd.DataFrame(all_results)
    res.to_csv(OUT / "repeated_scaffold_results.csv", index=False)
    summarize_repeated(res).to_csv(OUT / "repeated_scaffold_summary.csv", index=False)
    paired_wilcoxon_table(
        res, reference="PI_EGNN",
        metrics=("balanced_accuracy", "f1", "mcc", "roc_auc", "pr_auc")
    ).to_csv(OUT / "paired_wilcoxon_vs_PI_EGNN.csv", index=False)

    print("\nSaved:", OUT)
    print(summarize_repeated(res).to_string(index=False))

if __name__ == "__main__":
    main()
