from rdkit import Chem
import torch
from torch_geometric.data import Data


def bond_features(bond):
    return [
        bond.GetBondTypeAsDouble(),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]


def mol_to_line_graph(mol):
    bonds = list(mol.GetBonds())

    if bonds:
        x = torch.tensor(
            [bond_features(bond) for bond in bonds],
            dtype=torch.float,
        )
    else:
        x = torch.empty((0, 3), dtype=torch.float)

    edges = []

    for atom in mol.GetAtoms():
        bond_ids = [bond.GetIdx() for bond in atom.GetBonds()]

        for i in bond_ids:
            for j in bond_ids:
                if i != j:
                    edges.append([i, j])

    if edges:
        edge_index = torch.tensor(
            edges, dtype=torch.long
        ).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def smiles_to_line_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return mol_to_line_graph(mol)