
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
PROCESSED = ROOT / "processed"
CONFIG_DIR = ROOT / "config"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

RAW_B3DB = DATA_DIR / "B3DB_classification.tsv"
RAW_CHEMBL = DATA_DIR / "chembl_37_chemreps.txt" / "chembl_37_chemreps.txt"

CLEANED_B3DB = PROCESSED / "cleaned_b3db.csv"
DESCRIPTORS = PROCESSED / "descriptors.csv"
LABELS = PROCESSED / "labels.csv"
CONFORMERS = PROCESSED / "conformers.pkl"
MOL_GRAPHS = PROCESSED / "molecular_graphs.pkl"
LINE_GRAPHS = PROCESSED / "line_graphs.pkl"
DIST_GRAPHS = PROCESSED / "distance_graphs.pkl"
GEOM_FEATURES = PROCESSED / "geometry_features.pkl"
SPLITS_DIR = PROCESSED / "splits"

CHEMBL_CLEAN = PROCESSED / "chembl_clean.csv"

PRETRAINED_CKPT = CKPT_DIR / "pretrained" / "gin_pretrained.pt"
FINETUNED_CKPT = CKPT_DIR / "finetuned" / "best_model.pt"


def require(*paths, hint="python main.py preprocess"):
    missing = [
        str(path) for path in paths
        if not Path(path).exists() or Path(path).stat().st_size == 0
    ]

    if missing:
        files = "\n  ".join(missing)
        raise FileNotFoundError(
            f"Missing or empty files:\n  {files}\nRun `{hint}` first."
        )


def remove_stale(*paths, splits=False):
    for path in paths:
        Path(path).unlink(missing_ok=True)

    if splits:
        for split in ("scaffold", "random"):
            for part in ("train", "val", "test"):
                file_path = SPLITS_DIR / split / f"{part}.csv"
                file_path.unlink(missing_ok=True)


def load_config(name):
    config_path = CONFIG_DIR / f"{name}.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
