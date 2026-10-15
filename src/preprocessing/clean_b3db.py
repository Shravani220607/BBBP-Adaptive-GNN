
import pandas as pd
from rdkit import Chem

from utils.paths import (
    RAW_B3DB, CLEANED_B3DB, PROCESSED, CONFORMERS, LABELS,
    DESCRIPTORS, MOL_GRAPHS, LINE_GRAPHS, DIST_GRAPHS,
    require, remove_stale,
)


def largest_fragment(mol):
    fragments = Chem.GetMolFrags(mol, asMols=True)
    return max(fragments, key=lambda item: item.GetNumHeavyAtoms())


def clean_b3db():
    require(RAW_B3DB, hint="put B3DB_classification.tsv into data/")
    PROCESSED.mkdir(parents=True, exist_ok=True)

    remove_stale(
        CONFORMERS, LABELS, DESCRIPTORS, MOL_GRAPHS,
        LINE_GRAPHS, DIST_GRAPHS, splits=True,
    )

    df = pd.read_csv(RAW_B3DB, sep="\t")
    df = df[["SMILES", "BBB+/BBB-"]].dropna()
    df["label"] = df["BBB+/BBB-"].map({"BBB+": 1, "BBB-": 0})
    df = df.dropna(subset=["label"])

    rows = []

    for smiles, label in zip(df["SMILES"], df["label"]):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        mol = largest_fragment(mol)

        if mol.GetNumBonds() == 0:
            continue

        rows.append((Chem.MolToSmiles(mol), int(label)))

    result = pd.DataFrame(rows, columns=["smiles", "label"])

    conflicting = result.groupby("smiles")["label"].nunique()
    conflicting = conflicting[conflicting > 1].index
    result = result[~result["smiles"].isin(conflicting)]
    result = result.drop_duplicates("smiles").reset_index(drop=True)

    result.to_csv(CLEANED_B3DB, index=False)

    print("Clean samples:", len(result))
    print("Label counts:", result["label"].value_counts().to_dict())
    print("Saved:", CLEANED_B3DB)

    return result
