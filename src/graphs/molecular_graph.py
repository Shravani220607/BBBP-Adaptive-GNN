from rdkit import Chem
import torch
from torch_geometric.data import Data


def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
    ]


def mol_to_graph(mol):
    if mol is None:
        raise ValueError("Cannot create a graph from an invalid molecule.")

    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float,
    )

    edges = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])

    if edges:
        edge_index = torch.tensor(
            edges, dtype=torch.long
        ).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return mol_to_graph(mol)