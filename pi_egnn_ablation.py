"""
PI-EGNN ablation experiments.

Variants:
1) no_path: local edge message passing only
2) uniform_L2: path messages but equal path weights
3) learned_L1
4) learned_L2 (submitted default)
5) learned_L3

The same scaffold-disjoint seed splits are used for every variant.
"""
from pathlib import Path
import pandas as pd

from revision_config import CFG
from revision_core import (
    ensure_dir, scaffold_group_split, train_gnn_once,
    summarize_repeated, paired_wilcoxon_table
)

IN = Path(CFG.outdir) / "01_dataset" / "bace_classification_revised.csv"
OUT = Path(CFG.outdir) / "04_ablation"
ensure_dir(OUT)

def main():
    df = pd.read_csv(IN).dropna(subset=["smiles", "Class"]).reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)

    variants = [
        ("No_path", "none", CFG.max_path_len),
        ("Uniform_path_L2", "uniform", CFG.max_path_len),
    ]
    for L in CFG.ablation_path_lengths:
        variants.append((f"PI_EGNN_L{L}", "learned", int(L)))

    rows = []
    for seed in CFG.seeds:
        tr, va, te = scaffold_group_split(df, seed, CFG.test_size, CFG.val_size)
        for name, mode, L in variants:
            print(f"Seed {seed}: {name}")
            m, _, _ = train_gnn_once(
                df, tr, va, te, CFG, seed,
                path_mode=mode, max_path_len=L,
                task="classification", save_dir=OUT, model_tag=name
            )
            rows.append(m)
            pd.DataFrame(rows).to_csv(OUT / "ablation_partial.csv", index=False)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "ablation_results.csv", index=False)
    summarize_repeated(res).to_csv(OUT / "ablation_summary.csv", index=False)

    reference = f"PI_EGNN_L{CFG.max_path_len}"
    paired_wilcoxon_table(
        res, reference=reference,
        metrics=("balanced_accuracy", "f1", "mcc", "roc_auc", "pr_auc")
    ).to_csv(OUT / "ablation_paired_tests.csv", index=False)

    print(summarize_repeated(res).to_string(index=False))

if __name__ == "__main__":
    main()
