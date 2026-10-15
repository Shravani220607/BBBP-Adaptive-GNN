
import math
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from torch_geometric.data import Batch

from graphs.molecular_graph import smiles_to_graph
from graphs.line_graph import smiles_to_line_graph
from graphs.adaptive_distance_graph import (
    mol_to_multiscale_distance_graphs,
)
from graphs.geometric_features import extract_geometric_features
from models.bbbp_model import BBBModel
from utils.paths import ROOT


def build_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES string.")

    if mol.GetNumBonds() == 0:
        raise ValueError("The input molecule must contain at least one bond.")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42

    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError("Could not generate a 3D conformer.")

    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=100)
    except Exception:
        pass

    return mol


def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)

    values = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
    ]

    return np.asarray(values, dtype=np.float32).reshape(1, -1)


def normalize(values, mean, std):
    if mean is None or std is None:
        return values

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std[std < 1e-6] = 1.0

    return (values - mean) / std


def predict(smiles, model_type="adaptive", threshold=0.5, checkpoint_path=None):
    if model_type not in ("adaptive", "baseline"):
        raise ValueError("Model must be 'adaptive' or 'baseline'.")

    if checkpoint_path is None:
        checkpoint_path = (
            ROOT / "checkpoints" /
            ("adaptive_best.pt" if model_type == "adaptive"
             else "baseline_best.pt")
        )
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    adaptive = model_type == "adaptive"
    budget = checkpoint.get("edge_budget_ratio", 0.5)
    preserve_bonds = checkpoint.get("preserve_bonds", True)
    ablation_mode = checkpoint.get(
        "ablation_mode", "adaptive" if adaptive else "full"
    )

    model = BBBModel(
        ablation_mode=ablation_mode,
        use_adaptive_spatial=adaptive,
        edge_budget_ratio=budget,
        preserve_bonds=preserve_bonds,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    mol = build_molecule(smiles)

    graph = Batch.from_data_list([smiles_to_graph(smiles)]).to(device)
    line_graph = Batch.from_data_list(
        [smiles_to_line_graph(smiles)]
    ).to(device)

    distance_list = mol_to_multiscale_distance_graphs(mol)
    candidate_edges = int(distance_list[-1].edge_index.shape[1])
    distance_graphs = [
        Batch.from_data_list([item]).to(device)
        for item in distance_list
    ]

    descriptors = normalize(
        get_descriptors(smiles),
        checkpoint.get("desc_mean"),
        checkpoint.get("desc_std"),
    )
    descriptors = torch.tensor(descriptors, dtype=torch.float32, device=device)

    geometry = extract_geometric_features(mol)
    if isinstance(geometry, dict):
        geometry = geometry["combined"]

    geometry = np.asarray(geometry, dtype=np.float32).reshape(1, -1)
    geometry = normalize(
        geometry,
        checkpoint.get("geom_mean"),
        checkpoint.get("geom_std"),
    )
    geometry = torch.tensor(geometry, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(
            graph_data=graph,
            line_data=line_graph,
            distance_data=distance_graphs,
            descriptors=descriptors,
            geometry_data=geometry,
        )
        p_positive = torch.sigmoid(logits.reshape(-1))[0].item()

    p_negative = 1.0 - p_positive
    predicted_class = "BBB+" if p_positive >= threshold else "BBB-"

    print("\n========== BBBP PREDICTION ==========")
    print(f"SMILES             : {smiles}")
    print(f"Model              : {model_type}")
    print(f"Checkpoint         : {checkpoint_path.name}")
    print(f"Predicted class    : {predicted_class}")
    print(f"BBB+ probability   : {p_positive:.4f}")
    print(f"BBB- probability   : {p_negative:.4f}")
    print(f"Decision threshold : {threshold:.2f}")
    print(f"Candidate edges    : {candidate_edges}")

    if adaptive:
        print(f"Edge budget ratio  : {budget:.2f}")
        print(f"Preserve bonds     : {preserve_bonds}")

        for name, module in model.named_modules():
            for attribute in ("last_edge_stats", "edge_stats", "last_stats"):
                stats = getattr(module, attribute, None)
                if isinstance(stats, dict):
                    print(f"Edge stats ({name}):")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    break

    print("=====================================")

    return {
        "smiles": smiles,
        "model": model_type,
        "predicted_class": predicted_class,
        "bbb_plus_probability": p_positive,
        "bbb_minus_probability": p_negative,
        "threshold": threshold,
        "candidate_edges": candidate_edges,
    }
