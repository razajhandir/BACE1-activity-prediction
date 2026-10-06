"""
PLIP interaction profiling wrapper.

Use after generating a docked protein-ligand complex PDB.
Requires PLIP CLI (plipcmd) on PATH.

The script:
- runs PLIP,
- parses report.xml if generated,
- exports interaction types/residues,
- separately checks distances from the ligand to BACE1 ASP32 and ASP228.

This replaces simple atom-pair contact counting with a dedicated geometric
interaction profiler, as requested by the reviewers.
"""
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

from revision_config import CFG
from revision_core import ensure_dir

OUT = Path(CFG.outdir) / "09_plip"
ensure_dir(OUT)

# EDIT this to the final validated complex.
COMPLEX_PDB = Path(CFG.outdir) / "08_docking" / "final_complex.pdb"

def parse_pdb_atoms(path):
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                rows.append({
                    "record": line[:6].strip(),
                    "atom": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip(),
                    "resid": int(line[22:26]),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                })
            except Exception:
                pass
    return pd.DataFrame(rows)

def flatten_interactions(xml_path):
    root = ET.parse(xml_path).getroot()
    rows = []
    interesting = {
        "hydrophobic_interaction", "hydrogen_bond", "water_bridge",
        "salt_bridge", "pi_stack", "pi_cation_interaction",
        "halogen_bond", "metal_complex"
    }
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag not in interesting:
            continue
        row = {"interaction_type": tag}
        for child in elem:
            ctag = child.tag.split("}")[-1]
            if child.text and child.text.strip():
                row[ctag] = child.text.strip()
        rows.append(row)
    return pd.DataFrame(rows)

def catalytic_distance_table(pdb_path):
    atoms = parse_pdb_atoms(pdb_path)
    lig = atoms[atoms["record"] == "HETATM"].copy()
    prot = atoms[atoms["record"] == "ATOM"].copy()
    if lig.empty:
        lig = atoms[atoms["record"] == "HETATM"].copy()

    rows = []
    for resid in [32, 228]:
        aa = prot[prot["resid"] == resid]
        if aa.empty or lig.empty:
            rows.append({"residue": f"ASP{resid}", "min_ligand_distance_A": np.nan})
            continue
        A = aa[["x","y","z"]].values[:, None, :]
        B = lig[["x","y","z"]].values[None, :, :]
        d = np.sqrt(((A-B)**2).sum(axis=2))
        rows.append({"residue": f"ASP{resid}", "min_ligand_distance_A": float(d.min())})
    return pd.DataFrame(rows)

def main():
    if not COMPLEX_PDB.exists():
        print("Set COMPLEX_PDB in 09_plip_interactions.py to the final validated complex.")
        return

    exe = shutil.which("plipcmd")
    if not exe:
        raise RuntimeError("PLIP CLI not found. Install PLIP and ensure plipcmd is on PATH.")

    run_dir = OUT / "plip_output"
    ensure_dir(run_dir)
    cmd = [exe, "-f", str(COMPLEX_PDB), "-x", "-t", "-o", str(run_dir)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    (OUT / "plip_stdout_stderr.txt").write_text(
        (p.stdout or "") + "\nSTDERR:\n" + (p.stderr or ""), encoding="utf-8"
    )

    xmls = list(run_dir.rglob("report.xml"))
    if xmls:
        flat = flatten_interactions(xmls[0])
        flat.to_csv(OUT / "plip_interactions.csv", index=False)
        if not flat.empty:
            flat.groupby("interaction_type").size().rename("count").reset_index().to_csv(
                OUT / "plip_interaction_counts.csv", index=False
            )

    catalytic_distance_table(COMPLEX_PDB).to_csv(
        OUT / "catalytic_Asp32_Asp228_distances.csv", index=False
    )
    print("PLIP outputs saved to:", OUT)

if __name__ == "__main__":
    main()
