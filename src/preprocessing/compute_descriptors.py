
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from utils.paths import CLEANED_B3DB, DESCRIPTORS, LABELS, CONFORMERS, require


DESCRIPTOR_COLUMNS = [
    "molecular_weight",
    "logp",
    "tpsa",
    "h_donors",
    "h_acceptors",
    "rotatable_bonds",
    "ring_count",
]


def compute_descriptors():
    require(
        CLEANED_B3DB, LABELS, CONFORMERS,
        hint="python main.py preprocess --stage conformers",
    )

    df = pd.read_csv(CLEANED_B3DB)
    rows = []

    for smiles in df["smiles"]:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Invalid SMILES in cleaned data: {smiles}")

        rows.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
        ])

    descriptors = pd.DataFrame(rows, columns=DESCRIPTOR_COLUMNS)
    assert len(descriptors) == len(df)
    assert not descriptors.isna().any().any()

    descriptors.to_csv(DESCRIPTORS, index=False)
    print("Descriptors saved:", descriptors.shape)
