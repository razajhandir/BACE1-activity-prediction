"""
Structural-alert and CNS-oriented physicochemical screening.

This does NOT claim to calculate a validated CNS-MPO score. It applies transparent
RDKit-based heuristic criteria and standard structural-alert catalogs.

Catalogs:
- PAINS
- BRENK
- NIH
- ZINC

Sequential recommendation for revised screening:
1) PI-EGNN probability/rank
2) applicability-domain status
3) alert-free requirement
4) CNS-oriented physicochemical heuristic
5) docking only after the above filters
"""
from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors, FilterCatalog

from revision_config import CFG
from revision_core import ensure_dir

OUT = Path(CFG.outdir) / "07_structural_alerts_cns"
ensure_dir(OUT)

def choose_input():
    choices = [
        Path(CFG.outdir) / "05b_zinc_rescreen" / "zinc_revision_screened_with_ids.csv",
        Path(CFG.original_screened_zinc_csv),
        Path(CFG.original_top20_csv),
    ]
    for p in choices:
        if p.exists():
            df = pd.read_csv(p)
            smi = next((c for c in ["smiles", "SMILES", "Smiles"] if c in df), None)
            if smi:
                return df.rename(columns={smi: "smiles"}), p
    raise FileNotFoundError("Set a ZINC screened/top-candidate CSV path in revision_config.py.")

def catalog(name):
    cats = FilterCatalog.FilterCatalogParams.FilterCatalogs
    p = FilterCatalog.FilterCatalogParams()
    p.AddCatalog(getattr(cats, name))
    return FilterCatalog.FilterCatalog(p)

CATALOGS = {n: catalog(n) for n in ["PAINS", "BRENK", "NIH", "ZINC"]}

def alerts(mol, cat):
    ms = cat.GetMatches(mol)
    return ";".join(sorted(set(m.GetDescription() for m in ms)))

def props(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return {"valid": False}
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rb = Lipinski.NumRotatableBonds(mol)
    qed = QED.qed(mol)
    heavy = mol.GetNumHeavyAtoms()
    a = {name: alerts(mol, cat) for name, cat in CATALOGS.items()}
    alert_free = all(not v for v in a.values())
    cns_like = (
        mw <= CFG.cns_mw_max and
        CFG.cns_logp_min <= logp <= CFG.cns_logp_max and
        tpsa <= CFG.cns_tpsa_max and
        hbd <= CFG.cns_hbd_max and
        rb <= CFG.cns_rb_max and
        qed >= CFG.cns_qed_min
    )
    return {
        "valid": True, "MW": mw, "LogP": logp, "TPSA": tpsa,
        "HBD": hbd, "HBA": hba, "RB": rb, "QED": qed,
        "heavy_atoms": heavy, "PAINS_alerts": a["PAINS"],
        "BRENK_alerts": a["BRENK"], "NIH_alerts": a["NIH"],
        "ZINC_alerts": a["ZINC"], "alert_free": alert_free,
        "cns_heuristic_pass": cns_like,
        "cns_filter_definition": (
            f"MW<={CFG.cns_mw_max}; {CFG.cns_logp_min}<=LogP<={CFG.cns_logp_max}; "
            f"TPSA<={CFG.cns_tpsa_max}; HBD<={CFG.cns_hbd_max}; "
            f"RB<={CFG.cns_rb_max}; QED>={CFG.cns_qed_min}"
        )
    }

def main():
    df, source = choose_input()
    desc = pd.DataFrame([props(s) for s in df["smiles"]])
    out = pd.concat([df.reset_index(drop=True), desc], axis=1)

    ad_file = Path(CFG.outdir) / "05_applicability_domain" / "candidate_applicability_domain.csv"
    if ad_file.exists():
        ad = pd.read_csv(ad_file)
        # Merge by smiles to avoid row-order assumptions.
        keep = [c for c in ["smiles", "max_tanimoto", "ad_threshold", "in_domain",
                            "exact_scaffold_in_training"] if c in ad]
        out = out.merge(ad[keep].drop_duplicates("smiles"), on="smiles", how="left")

    if "in_domain" in out:
        out["recommended_for_docking"] = (
            out["valid"].fillna(False) &
            out["alert_free"].fillna(False) &
            out["cns_heuristic_pass"].fillna(False) &
            out["in_domain"].fillna(False)
        )
    else:
        out["recommended_for_docking"] = (
            out["valid"].fillna(False) &
            out["alert_free"].fillna(False) &
            out["cns_heuristic_pass"].fillna(False)
        )

    # Keep original model ranking; do not invent a new weighted score.
    score_cols = [c for c in ["PI_EGNN_ensemble_mean", "PI_EGNN_score", "pi_egnn_active_probability", "active_probability", "probability"] if c in out]
    if score_cols:
        out = out.sort_values(
            ["recommended_for_docking", score_cols[0]],
            ascending=[False, False]
        )
    else:
        out = out.sort_values("recommended_for_docking", ascending=False)

    out.to_csv(OUT / "zinc_candidates_structural_alert_CNS_AD.csv", index=False)
    out[out["recommended_for_docking"]].to_csv(
        OUT / "zinc_candidates_recommended_for_redocking.csv", index=False
    )

    print("Input:", source)
    print("Total:", len(out))
    print("Alert-free:", int(out["alert_free"].sum()))
    print("CNS heuristic pass:", int(out["cns_heuristic_pass"].sum()))
    print("Recommended for docking:", int(out["recommended_for_docking"].sum()))

if __name__ == "__main__":
    main()
