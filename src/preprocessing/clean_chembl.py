
import random

import pandas as pd
from rdkit import Chem

from utils.paths import RAW_CHEMBL, CHEMBL_CLEAN, PROCESSED, load_config, require


def clean_chembl():
    require(
        RAW_CHEMBL,
        hint="put chembl_37_chemreps.txt into data/chembl_37_chemreps.txt/",
    )

    config = load_config("pretrain")
    target = config["num_molecules"]
    random.seed(config.get("seed", 42))
    PROCESSED.mkdir(parents=True, exist_ok=True)

    total_rows = 2_900_000
    sample_rate = min(1.0, 1.3 * target / total_rows)

    df = pd.read_csv(
        RAW_CHEMBL,
        sep="\t",
        usecols=["canonical_smiles"],
        dtype=str,
        skiprows=lambda row: row > 0 and random.random() > sample_rate,
    )

    smiles_list = []

    for smiles in df["canonical_smiles"].dropna():
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        if mol.GetNumBonds() == 0:
            continue

        if not 5 <= mol.GetNumHeavyAtoms() <= 60:
            continue

        if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            continue

        smiles_list.append(smiles)

        if len(smiles_list) == target:
            break

    pd.DataFrame({"smiles": smiles_list}).to_csv(CHEMBL_CLEAN, index=False)
    print("ChEMBL molecules kept:", len(smiles_list))
    print("Saved:", CHEMBL_CLEAN)
