
import pickle
from multiprocessing import Pool

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from utils.paths import (
    CLEANED_B3DB, CONFORMERS, LABELS, DESCRIPTORS,
    MOL_GRAPHS, LINE_GRAPHS, DIST_GRAPHS,
    require, remove_stale,
)


def embed(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42

        if AllChem.EmbedMolecule(mol, params) != 0:
            params.useRandomCoords = True

            if AllChem.EmbedMolecule(mol, params) != 0:
                return None

        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=50)
        except Exception:
            pass

        return mol

    except Exception:
        return None


def generate_conformers(num_workers=1, max_molecules=None):
    require(CLEANED_B3DB, hint="python main.py preprocess --stage clean")

    df = pd.read_csv(CLEANED_B3DB)

    if max_molecules is not None:
        df = df.iloc[:max_molecules].copy()

    smiles_list = df["smiles"].tolist()
    print("Generating conformers:", len(smiles_list))

    if num_workers > 1:
        with Pool(num_workers) as pool:
            molecules = list(pool.imap(embed, smiles_list, chunksize=16))
    else:
        molecules = []

        for i, smiles in enumerate(smiles_list):
            molecules.append(embed(smiles))

            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(smiles_list)}")

    valid_indices = [
        i for i, mol in enumerate(molecules) if mol is not None
    ]

    df = df.iloc[valid_indices].reset_index(drop=True)
    conformers = [
        {"smiles": smiles_list[i], "mol": molecules[i]}
        for i in valid_indices
    ]

    remove_stale(
        CONFORMERS, LABELS, DESCRIPTORS, MOL_GRAPHS,
        LINE_GRAPHS, DIST_GRAPHS, splits=True,
    )

    df.to_csv(CLEANED_B3DB, index=False)
    df[["label"]].to_csv(LABELS, index=False)

    with open(CONFORMERS, "wb") as file:
        pickle.dump(conformers, file)

    print("Successful conformers:", len(valid_indices))
    print("Failed conformers:", len(molecules) - len(valid_indices))
    print("Saved:", CONFORMERS)
