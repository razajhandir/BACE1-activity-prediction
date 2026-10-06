"""
Y-randomization / response permutation control.

Primary implementation uses the strongest classical Morgan-RF baseline because
it is fast enough for >=20 permutations. An optional PI-EGNN randomized-label
block is available but disabled by default because it is computationally costly.

For each permutation, labels are shuffled before training and the randomized
relationship is evaluated on the corresponding randomized test labels.
Chance-level AUROC/MCC should result if there is no spurious structure-response
relationship.
"""
from pathlib import Path
import copy
import numpy as np
import pandas as pd

from revision_config import CFG
from revision_core import (
    ensure_dir, scaffold_group_split, fingerprint_matrix, train_rf_classifier,
    classification_metrics, train_gnn_once
)

IN = Path(CFG.outdir) / "01_dataset" / "bace_classification_revised.csv"
OUT = Path(CFG.outdir) / "06_y_randomization"
ensure_dir(OUT)

def main():
    df = pd.read_csv(IN).dropna(subset=["smiles", "Class"]).reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)
    X, _, valid = fingerprint_matrix(
        df["smiles"], "morgan", CFG.fp_bits, CFG.morgan_radius
    )
    if not valid.all():
        raise RuntimeError("Invalid molecules remain.")

    rows = []
    base_seed = CFG.seeds[0]
    tr, va, te = scaffold_group_split(df, base_seed, CFG.test_size, CFG.val_size)

    for perm in range(CFG.y_randomization_permutations):
        rng = np.random.default_rng(1000 + perm)
        y_perm = rng.permutation(df["Class"].values)
        model = train_rf_classifier(X[tr], y_perm[tr], 1000 + perm)
        prob = model.predict_proba(X[te])[:, 1]
        m = classification_metrics(y_perm[te], prob)
        m.update({"permutation": perm, "model": "RF_Morgan_y_randomized"})
        rows.append(m)
        print("Permutation", perm, "AUROC", m["roc_auc"], "MCC", m["mcc"])

    pd.DataFrame(rows).to_csv(OUT / "y_randomization_RF_Morgan.csv", index=False)

    # Optional, much slower PI-EGNN y-randomization.
    if CFG.run_gnn_y_randomization:
        gnn_rows = []
        n_gnn = min(10, CFG.y_randomization_permutations)
        for perm in range(n_gnn):
            rng = np.random.default_rng(5000 + perm)
            dperm = df.copy()
            dperm["Class"] = rng.permutation(dperm["Class"].values)
            m, _, _ = train_gnn_once(
                dperm, tr, va, te, CFG, 5000 + perm,
                path_mode="learned", max_path_len=CFG.max_path_len,
                task="classification", save_dir=OUT,
                model_tag=f"PI_EGNN_Yrand_{perm}"
            )
            m["permutation"] = perm
            gnn_rows.append(m)
        pd.DataFrame(gnn_rows).to_csv(OUT / "y_randomization_PI_EGNN.csv", index=False)

if __name__ == "__main__":
    main()
