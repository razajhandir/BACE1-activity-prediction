"""
Creates a reproducibility manifest for repository deposition.

Includes:
- SHA256 hashes,
- Python/platform/library versions,
- revision configuration,
- list of generated split files, predictions, summaries and docking inputs.

Run only after the main revision analyses are complete.
"""
from pathlib import Path
import hashlib
import json
import platform
import sys
import subprocess
import pandas as pd

from revision_config import CFG
from revision_core import ensure_dir

OUT = Path(CFG.outdir) / "10_reproducibility"
ensure_dir(OUT)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def package_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None

def main():
    root = Path(CFG.outdir)
    rows = []
    for p in root.rglob("*"):
        if p.is_file() and OUT not in p.parents:
            try:
                rows.append({
                    "relative_path": str(p.relative_to(root)),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256(p)
                })
            except Exception:
                pass
    pd.DataFrame(rows).to_csv(OUT / "file_manifest_sha256.csv", index=False)

    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scikit-learn": package_version("scikit-learn"),
        "scipy": package_version("scipy"),
        "rdkit": package_version("rdkit"),
        "torch": package_version("torch"),
        "torch-geometric": package_version("torch-geometric"),
        "xgboost": package_version("xgboost"),
        "catboost": package_version("catboost"),
    }
    with open(OUT / "software_versions.json", "w") as f:
        json.dump(versions, f, indent=2)

    with open(OUT / "revision_config_snapshot.json", "w") as f:
        json.dump(vars(CFG), f, indent=2, default=str)

    print("Reproducibility manifest written to:", OUT)

if __name__ == "__main__":
    main()
