# BACE1-activity-prediction
Path-aware graph neural network (PI-EGNN) for BACE1 activity prediction, scaffold-disjoint evaluation, ZINC virtual screening, applicability-domain analysis, validated docking, and protein–ligand interaction profiling.
# PI-EGNN for BACE1 Activity Prediction and Computational Compound Prioritization

This repository contains the code, processed data, evaluation outputs, and supporting files associated with the study:

**“Path-Aware Graph Neural Networks for BACE1 Activity Prediction and Computational Compound Prioritization”**

The study introduces **PI-EGNN**, a path-aware graph neural network for BACE1 activity classification, and evaluates it using repeated scaffold-disjoint validation, molecular-fingerprint baselines, path-component ablation, external ZINC screening, applicability-domain analysis, molecular docking, and protein–ligand interaction profiling.

---

## Overview

PI-EGNN extends conventional molecular graph message passing by incorporating learned aggregation over short molecular paths. The framework was evaluated on a curated BACE1 dataset constructed from MoleculeNet and ChEMBL.

The computational workflow includes:

- BACE1 dataset curation and molecular standardization
- Consistent binary activity labeling at pIC50 = 6
- Repeated Bemis–Murcko scaffold-disjoint evaluation
- Random-forest fingerprint baselines
- Edge-only GNN comparison
- PI-EGNN path-length and weighting ablation
- Response-randomization control
- External ZINC virtual screening
- ZINC provenance verification
- Scaffold-aligned applicability-domain analysis
- PAINS/NIH structural-alert filtering
- CNS-oriented physicochemical assessment
- BACE1 docking-protocol validation by crystallographic-ligand redocking
- Repeated AutoDock Vina docking
- Ligand-efficiency analysis
- PLIP-based protein–ligand interaction profiling

---

## Dataset

The final BACE1 classification dataset contains:

| Property | Value |
|---|---:|
| Total unique molecules | 5,989 |
| Active molecules | 3,797 |
| Inactive molecules | 2,192 |
| Molecules with exact pIC50 information | 5,750 |
| Classification conflicts excluded | 73 |

The principal PI-EGNN configuration uses:
Graph layers:        3
Hidden dimension:    128
Dropout:             0.35
Maximum path length: 2
Optimizer:           AdamW
Learning rate:       5e-4
Weight decay:        1e-5
Batch size:          32
Maximum epochs:      100
Gradient clipping:   5.0

The following configurations were evaluated using identical scaffold partitions:

No-path GNN
PI-EGNN L=1
PI-EGNN L=2
PI-EGNN L=3
Uniform-path L=2

Applicability Domain
The applicability domain was defined using:
Fingerprint: Morgan
Radius:      2
Bits:        2048
Similarity:  Tanimoto

Structural Alerts and Candidate Selection
Candidate prioritization incorporated:
- PI-EGNN ranking
- ZINC provenance verification
- PAINS A/B/C exclusion
- NIH structural-alert exclusion
- scaffold-aligned AD support
- Bemis–Murcko scaffold diversity
- physicochemical profiling

Molecular Docking
Docking was performed using BACE1 structure:
PDB ID: 4IVT
Co-crystallized ligand: VTI

The docking protocol tested both monoprotonated catalytic-dyad configurations:
Asp32H / Asp228-
Asp32- / Asp228H

Five independent AutoDock Vina runs were performed using seeds:
11, 22, 33, 44, 55

Docking settings:
AutoDock Vina: 1.2.7
Exhaustiveness: 32
Number of modes: 9
Energy range: 3 kcal/mol

Grid center:
x = 22.376
y = 22.871
z = 0.873 Å

Grid size:
21.964 × 20.000 × 20.000 Å³

Redocking Validation
The crystallographic VTI ligand was redocked into both catalytic-dyad receptor states.
Median symmetry-corrected heavy-atom RMSD:
Asp32- / Asp228H: 1.839 Å
Asp32H / Asp228-: 1.849 Å

Both states satisfied the predefined acceptance criterion:
Median RMSD <= 2.0 Å

The Asp32- / Asp228H state was retained for production docking because it produced the marginally lower median RMSD.
Docking Results
Mean AutoDock Vina scores across five runs:
Compound	Mean Vina score (kcal/mol)
Atabecestat	-9.346
Verubecestat	-9.235
Lanabecestat	-9.178
LY2811376	-7.931
ZINC2522617	-6.324
ZINC33434936	-6.316
ZINC1319172015	-5.812


The reference inhibitors produced more favorable absolute Vina scores than the prioritized ZINC compounds.
Docking scores are treated as empirical structural-ranking outputs and not as experimental binding free energies.
Protein–Ligand Interaction Profiling
Protein–ligand interaction analysis was performed using:
PLIP:       3.0.1
OpenBabel:  3.2.1

The analysis included:
VTI crystallographic ligand
Atabecestat
ZINC1319172015
ZINC2522617
ZINC33434936

Detected interaction classes included:
- hydrogen bonds
- hydrophobic contacts
- salt bridges
- pi-stacking
- pi-cation interactions
- halogen bonds
Particular attention was given to catalytic residues:
Asp32
Asp228

Interaction patterns derived from static docking poses should not be interpreted as evidence of binding stability or experimental inhibition.
Software Environment
The computational workflow used:
Python:              3.10.21
RDKit:               2026.03.1
scikit-learn:        1.7.2
PyTorch:             2.5.1+cu121
PyTorch Geometric:   2.8.0.post1
CUDA:                12.1
PDBFixer:            1.12.0
OpenMM:              8.6
Meeko:               0.8.0
AutoDock Vina:       1.2.7
PLIP:                3.0.1
OpenBabel:           3.2.1

Repository Structure
A recommended organization is:
BACE1-PI-EGNN/
│
├── README.md
├── requirements.txt
│
├── code/
│   ├── revision_config.py
│   ├── revision_core.py
│   ├── rebuild_datasets.py
│   ├── 
│   ├── y_randomization.py
│   ├── FINAL_AD_ALIGNED_SELECTION.py
│   ├── FINAL_DOCKING_VALIDATION.py
│   ├── FINAL_PLIP_INTERACTIONS.py
│   └── FULL_ZINC_APPLICABILITY_DOMAIN.py
│
├── data/
│   ├── curated_bace1/
│   ├── split_indices/
│   └── zinc_screening/
│
├── results/
│   ├── classification/
│   ├── ablation/
│   ├── randomization/
│   ├── applicability_domain/
│   ├── docking/
│   └── plip/
│
├── figures/
│
└── supplementary/

