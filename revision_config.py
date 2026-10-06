from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class RevisionConfig:
    # -----------------------------
    # EDIT THESE PATHS FOR YOUR PC
    # -----------------------------
    project_dir: str = r"D:/articles/BACE1"
    bace_csv: str = r"D:/articles/BACE1/data/bace.csv"
    chembl_raw_csv: str = r"D:/articles/BACE1/data/DOWNLOAD-GvYnSOB18_NqmDckwnmipqCmU_5W_EAcESb8kMG-00E_eq_.csv"
    zinc_uri: str = r"D:/articles/BACE1/ZINC-downloader-2D-smi.uri"
    pdb_4ivt: str = r"D:/articles/BACE1/4IVT.pdb"

    # Optional inputs produced by the original pipeline
    original_merged_csv: str = r"D:/articles/BACE1/data/bace_moleculenet_chembl_merged.csv"
    original_top20_csv: str = r"D:/articles/BACE1/zinc20_top20_for_docking.csv"
    original_screened_zinc_csv: str = r"D:/articles/BACE1/outputs_all_spyder/zinc_screened_all.csv"

    # Revision output root
    outdir: str = r"D:/articles/BACE1/reviewer_revision"

    # -----------------------------
    # Dataset curation
    # -----------------------------
    target_chembl_id: str = "CHEMBL4822"
    target_organism: str = "Homo sapiens"
    target_type: str = "SINGLE PROTEIN"
    activity_threshold_nm: float = 1000.0  # pIC50 >= 6
    assay_scope: str = "single_protein"    # "single_protein" or "all_binding"
    keep_stereochemistry: bool = True
    canonicalize_tautomers: bool = False
    uncharge_for_identity: bool = True
    exclude_nonempty_validity_comment: bool = True

    # -----------------------------
    # Repeated scaffold validation
    # -----------------------------
    seeds: tuple = (11, 22, 33, 44, 55)
    test_size: float = 0.15
    val_size: float = 0.15
    fp_bits: int = 2048
    morgan_radius: int = 2

    # Submitted PI-EGNN settings recovered from the Spyder script
    epochs: int = 100
    batch_size: int = 32
    hidden_dim: int = 128
    layers: int = 3
    dropout: float = 0.35
    lr: float = 5e-4
    weight_decay: float = 1e-5
    max_path_len: int = 2
    early_stopping_patience: int = 20

    # Ablation
    ablation_path_lengths: tuple = (1, 2, 3)

    # y-randomization
    y_randomization_permutations: int = 20
    run_gnn_y_randomization: bool = False

    # Applicability domain
    ad_percentile: float = 5.0
    ad_calibration_max_n: int = 12000
    nearest_neighbors_k: int = 5

    # CNS-oriented heuristic (NOT claimed as CNS-MPO)
    cns_mw_max: float = 450.0
    cns_logp_min: float = 1.0
    cns_logp_max: float = 4.0
    cns_tpsa_max: float = 90.0
    cns_hbd_max: int = 2
    cns_rb_max: int = 8
    cns_qed_min: float = 0.50

    # Docking validation
    vina_executable: str = "vina"
    vina_exhaustiveness: int = 16
    vina_num_modes: int = 9
    vina_seeds: tuple = (101, 202, 303, 404, 505)
    docking_box_size: float = 22.0
    cocrystal_resname: str = "VTI"
    cocrystal_chain: str = "A"
    cocrystal_resid: int = 401

    def path(self, *parts):
        return str(Path(self.outdir).joinpath(*parts))

CFG = RevisionConfig()
