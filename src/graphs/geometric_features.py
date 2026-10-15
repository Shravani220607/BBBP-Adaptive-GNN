import numpy as np
from rdkit.Chem import rdMolTransforms


DIST_DIM = 8
ANGLE_DIM = 8
TORSION_DIM = 8
TOTAL_GEOM_DIM = 24


def extract_geometric_features(mol):
    zeros = np.zeros(TOTAL_GEOM_DIM, dtype=np.float32)

    def empty_result():
        return {
            "distances": zeros[:8].copy(),
            "angles": zeros[8:16].copy(),
            "torsions": zeros[16:24].copy(),
            "combined": zeros.copy(),
        }

    if mol is None or mol.GetNumConformers() == 0:
        return empty_result()

    try:
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        pos = conf.GetPositions()

        bond_lengths = [
            rdMolTransforms.GetBondLength(
                conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            )
            for bond in mol.GetBonds()
        ]

        if bond_lengths:
            mean_bond = np.mean(bond_lengths)
            std_bond = np.std(bond_lengths)
            min_bond = np.min(bond_lengths)
            max_bond = np.max(bond_lengths)
        else:
            mean_bond = std_bond = min_bond = max_bond = 0.0

        if num_atoms > 1:
            diff = pos[:, None, :] - pos[None, :, :]
            distances = np.linalg.norm(diff, axis=-1)
            pair_distances = distances[
                np.triu_indices(num_atoms, k=1)
            ]

            mean_spatial = np.mean(pair_distances)
            std_spatial = np.std(pair_distances)
            max_spatial = np.max(pair_distances)

            center = pos.mean(axis=0)
            radius = np.sqrt(np.mean(np.sum((pos - center) ** 2, axis=1)))
        else:
            mean_spatial = std_spatial = max_spatial = radius = 0.0

        distance_features = np.array([
            mean_bond, std_bond, min_bond, max_bond,
            mean_spatial, std_spatial, max_spatial, radius,
        ], dtype=np.float32)

        angles = []

        for atom in mol.GetAtoms():
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            center = atom.GetIdx()

            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    angles.append(
                        rdMolTransforms.GetAngleDeg(
                            conf, neighbors[i], center, neighbors[j]
                        )
                    )

        if angles:
            angles = np.asarray(angles, dtype=np.float32)
            angle_features = np.array([
                angles.mean(),
                angles.std(),
                angles.min(),
                angles.max(),
                np.mean((angles >= 104) & (angles <= 115)),
                np.mean((angles >= 116) & (angles <= 125)),
                np.mean(angles >= 165),
                len(angles) / max(num_atoms, 1),
            ], dtype=np.float32)
        else:
            angle_features = np.zeros(ANGLE_DIM, dtype=np.float32)

        torsions = []

        for bond in mol.GetBonds():
            j = bond.GetBeginAtomIdx()
            k = bond.GetEndAtomIdx()

            for atom_i in mol.GetAtomWithIdx(j).GetNeighbors():
                i = atom_i.GetIdx()
                if i == k:
                    continue

                for atom_l in mol.GetAtomWithIdx(k).GetNeighbors():
                    l = atom_l.GetIdx()
                    if l == j or l == i:
                        continue

                    torsions.append(
                        rdMolTransforms.GetDihedralDeg(conf, i, j, k, l)
                    )

        if torsions:
            torsions = np.asarray(torsions, dtype=np.float32)
            radians = np.radians(torsions)

            torsion_features = np.array([
                torsions.mean(),
                torsions.std(),
                np.abs(torsions).mean(),
                np.mean(np.abs(torsions) < 45),
                np.mean(np.abs(torsions) > 135),
                np.cos(radians).mean(),
                np.sin(radians).mean(),
                len(torsions) / max(num_atoms, 1),
            ], dtype=np.float32)
        else:
            torsion_features = np.zeros(TORSION_DIM, dtype=np.float32)

        features = np.concatenate([
            distance_features, angle_features, torsion_features
        ])
        features = np.nan_to_num(
            features, nan=0.0, posinf=0.0, neginf=0.0
        )

        return {
            "distances": features[:8],
            "angles": features[8:16],
            "torsions": features[16:24],
            "combined": features,
        }

    except Exception:
        return empty_result()