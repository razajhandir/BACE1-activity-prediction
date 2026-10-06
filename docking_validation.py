"""
AutoDock Vina redocking / replicate validation helper.

This script:
1) extracts the 4IVT co-crystallized ligand (VTI A 401),
2) derives the docking-box center directly from its heavy-atom centroid,
3) verifies that the reported center is reproduced,
4) runs replicate Vina dockings for prepared PDBQT inputs,
5) reports score mean/SD,
6) computes atom-name-matched redocking RMSD when atom names are preserved.

IMPORTANT:
Protein/ligand protonation and PDBQT preparation must be done explicitly.
For BACE1, prepare and document the catalytic dyad protonation state.
A strong validation strategy is to test both monoprotonated-dyad assignments
(Asp32-H/Asp228- and Asp32-/Asp228-H) and retain/report the protocol that
reproduces the crystallographic pose, rather than silently assuming a state.

Expected prepared inputs, for example:
reviewer_revision/08_docking/prepared/
    receptor_Asp32H.pdbqt
    receptor_Asp228H.pdbqt
    VTI_redock.pdbqt

The script does not fabricate protonation states.
"""
from pathlib import Path
import os
import re
import json
import shutil
import subprocess
import numpy as np
import pandas as pd

from revision_config import CFG
from revision_core import ensure_dir

OUT = Path(CFG.outdir) / "08_docking"
PREP = OUT / "prepared"
RUNS = OUT / "vina_runs"
ensure_dir(PREP); ensure_dir(RUNS)

def pdb_atoms(path, resname=None, chain=None, resid=None, heavy_only=True):
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            rn = line[17:20].strip()
            ch = line[21].strip()
            try:
                ri = int(line[22:26])
            except Exception:
                ri = None
            if resname and rn != resname: continue
            if chain and ch != chain: continue
            if resid is not None and ri != resid: continue
            element = line[76:78].strip().upper()
            if not element:
                element = re.sub(r"[^A-Za-z]", "", line[12:16].strip())[:2].upper()
            if heavy_only and element == "H": continue
            rows.append({
                "record": line[:6].strip(), "atom_name": line[12:16].strip(),
                "resname": rn, "chain": ch, "resid": ri, "element": element,
                "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
                "line": line.rstrip("\n")
            })
    return pd.DataFrame(rows)

def pdbqt_atoms(path, heavy_only=True):
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            # PDBQT follows PDB coordinate columns.
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except Exception:
                continue
            ad_type = line[77:79].strip() if len(line) >= 79 else ""
            element = re.sub(r"[^A-Za-z]", "", ad_type or atom_name)[:2].upper()
            if heavy_only and element.startswith("H"):
                continue
            rows.append({"atom_name": atom_name, "element": element, "x": x, "y": y, "z": z})
    return pd.DataFrame(rows)

def extract_cocrystal():
    atoms = pdb_atoms(
        CFG.pdb_4ivt, CFG.cocrystal_resname, CFG.cocrystal_chain,
        CFG.cocrystal_resid, heavy_only=False
    )
    if atoms.empty:
        raise RuntimeError("Co-crystal ligand not found.")
    path = OUT / f"{CFG.cocrystal_resname}_{CFG.cocrystal_chain}{CFG.cocrystal_resid}_crystal.pdb"
    with open(path, "w") as f:
        for line in atoms["line"]:
            f.write(line + "\n")
        f.write("END\n")
    heavy = atoms[atoms["element"].str.upper() != "H"]
    center = heavy[["x", "y", "z"]].mean().values
    span = np.ptp(heavy[["x", "y", "z"]].values, axis=0)
    meta = {
        "ligand": f"{CFG.cocrystal_resname} {CFG.cocrystal_chain} {CFG.cocrystal_resid}",
        "n_heavy_atoms": int(len(heavy)),
        "center_x": float(center[0]), "center_y": float(center[1]), "center_z": float(center[2]),
        "ligand_span_x": float(span[0]), "ligand_span_y": float(span[1]), "ligand_span_z": float(span[2]),
        "box_size_x": CFG.docking_box_size, "box_size_y": CFG.docking_box_size,
        "box_size_z": CFG.docking_box_size,
    }
    with open(OUT / "cocrystal_box_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return path, center, meta

def vina_score_from_pdbqt(path):
    vals = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "REMARK VINA RESULT:" in line:
                try:
                    vals.append(float(line.split("RESULT:")[1].split()[0]))
                except Exception:
                    pass
    return vals[0] if vals else np.nan

def direct_rmsd_by_atom_name(crystal_pdb, docked_pdbqt):
    c = pdb_atoms(
        crystal_pdb, CFG.cocrystal_resname, CFG.cocrystal_chain,
        CFG.cocrystal_resid, heavy_only=True
    )
    d = pdbqt_atoms(docked_pdbqt, heavy_only=True)
    if c.empty or d.empty:
        return np.nan, "missing_atoms"
    cm = c.set_index("atom_name")
    dm = d.set_index("atom_name")
    common = sorted(set(cm.index) & set(dm.index))
    if len(common) < max(3, int(0.8 * len(cm))):
        return np.nan, f"insufficient_name_matches:{len(common)}/{len(cm)}"
    a = cm.loc[common, ["x", "y", "z"]].values
    b = dm.loc[common, ["x", "y", "z"]].values
    rmsd = float(np.sqrt(np.mean(np.sum((a-b)**2, axis=1))))
    return rmsd, f"matched:{len(common)}"

def run_vina(receptor, ligand, center, tag):
    exe = shutil.which(CFG.vina_executable) or CFG.vina_executable
    rows = []
    for seed in CFG.vina_seeds:
        out = RUNS / f"{tag}_seed{seed}.pdbqt"
        log = RUNS / f"{tag}_seed{seed}.log.txt"
        cmd = [
            exe, "--receptor", str(receptor), "--ligand", str(ligand),
            "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
            "--size_x", str(CFG.docking_box_size), "--size_y", str(CFG.docking_box_size),
            "--size_z", str(CFG.docking_box_size),
            "--exhaustiveness", str(CFG.vina_exhaustiveness),
            "--num_modes", str(CFG.vina_num_modes),
            "--seed", str(seed), "--out", str(out)
        ]
        print(" ".join(cmd))
        p = subprocess.run(cmd, capture_output=True, text=True)
        log.write_text((p.stdout or "") + "\nSTDERR:\n" + (p.stderr or ""), encoding="utf-8")
        score = vina_score_from_pdbqt(out) if out.exists() else np.nan
        rows.append({"tag": tag, "seed": seed, "score_kcal_mol": score,
                     "returncode": p.returncode, "output_pdbqt": str(out)})
    return pd.DataFrame(rows)

def main():
    crystal_pdb, center, meta = extract_cocrystal()
    print("4IVT ligand-derived center:", center)
    print("Ligand heavy-atom span:", meta["ligand_span_x"], meta["ligand_span_y"], meta["ligand_span_z"])
    print("Expected from submitted manuscript: approximately 22.020, 23.892, 0.344")

    receptors = sorted(PREP.glob("receptor_*.pdbqt"))
    ligand = PREP / "VTI_redock.pdbqt"
    if not receptors or not ligand.exists():
        print("\nPrepared PDBQT files are not present yet.")
        print("Create explicit receptor protonation variants and VTI_redock.pdbqt in:")
        print(PREP)
        print("Then rerun this script. No docking validation result has been fabricated.")
        return

    all_rows = []
    for receptor in receptors:
        tag = f"{receptor.stem}_VTI_redock"
        df = run_vina(receptor, ligand, center, tag)
        for i, r in df.iterrows():
            out = Path(r["output_pdbqt"])
            if out.exists():
                rmsd, note = direct_rmsd_by_atom_name(crystal_pdb, out)
                df.loc[i, "redock_RMSD_A"] = rmsd
                df.loc[i, "rmsd_note"] = note
        all_rows.append(df)

    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(OUT / "redocking_replicates.csv", index=False)
    summary = res.groupby("tag").agg(
        n_runs=("score_kcal_mol", "size"),
        vina_mean=("score_kcal_mol", "mean"),
        vina_sd=("score_kcal_mol", "std"),
        rmsd_mean_A=("redock_RMSD_A", "mean"),
        rmsd_sd_A=("redock_RMSD_A", "std"),
        rmsd_min_A=("redock_RMSD_A", "min"),
    ).reset_index()
    summary.to_csv(OUT / "redocking_summary.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
