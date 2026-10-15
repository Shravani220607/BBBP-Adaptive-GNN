import numpy as np
import torch
from torch_geometric.data import Data


DEFAULT_CUTOFFS = [2.0, 3.0, 4.0, 5.0]


def mol_to_multiscale_distance_graphs(mol, cutoffs=DEFAULT_CUTOFFS):
    pos = mol.GetConformer().GetPositions()
    num_atoms = len(pos)

    x = torch.tensor(
        [[atom.GetAtomicNum()] for atom in mol.GetAtoms()],
        dtype=torch.float,
    )

    diff = pos[:, None, :] - pos[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    not_self = ~np.eye(num_atoms, dtype=bool)

    graphs = []

    for cutoff in cutoffs:
        src, dst = np.nonzero(
            (distances <= float(cutoff)) & not_self
        )

        if len(src):
            edge_index = torch.tensor(
                np.stack([src, dst]), dtype=torch.long
            )
            edge_attr = torch.tensor(
                distances[src, dst], dtype=torch.float
            ).view(-1, 1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph.cutoff = float(cutoff)
        graphs.append(graph)

    return graphs


def mol_to_distance_graph(mol, max_distance=5.0):
    return mol_to_multiscale_distance_graphs(
        mol, cutoffs=[max_distance]
    )[0]


def mol_to_candidate_spatial_graph(mol, max_distance=5.0):
    pos = mol.GetConformer().GetPositions()
    num_atoms = len(pos)

    x = torch.tensor(
        [[atom.GetAtomicNum()] for atom in mol.GetAtoms()],
        dtype=torch.float,
    )

    diff = pos[:, None, :] - pos[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)

    bonded = set()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bonded.add((i, j))
        bonded.add((j, i))

    src, dst = np.nonzero(
        (distances <= float(max_distance))
        & ~np.eye(num_atoms, dtype=bool)
    )

    pairs = set(zip(src.tolist(), dst.tolist()))
    pairs.update(bonded)
    pairs = sorted(pairs)

    if pairs:
        src = [i for i, _ in pairs]
        dst = [j for _, j in pairs]

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(
            distances[src, dst], dtype=torch.float
        ).view(-1, 1)
        is_bonded = torch.tensor(
            [[float((i, j) in bonded)] for i, j in pairs],
            dtype=torch.float,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        is_bonded = torch.empty((0, 1), dtype=torch.float)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        is_bonded=is_bonded,
    )
    graph.pos = torch.tensor(pos, dtype=torch.float)
    graph.num_candidate_edges = edge_index.size(1)
    graph.num_bonded_edges = int(is_bonded.sum().item())

    return graph