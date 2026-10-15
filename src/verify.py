"""Consistency checks and tensor shape verification.
Run with:
    python main.py verify
    python main.py verify --what shapes
    python main.py verify --what data
"""

import pickle
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from utils.paths import (
    CLEANED_B3DB,
    DESCRIPTORS,
    LABELS,
    CONFORMERS,
    MOL_GRAPHS,
    LINE_GRAPHS,
    DIST_GRAPHS,
    GEOM_FEATURES,
    SPLITS_DIR,
    require,
)


def verify_data():
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from preprocessing.compute_descriptors import DESCRIPTOR_COLUMNS

    require(
        CLEANED_B3DB,
        DESCRIPTORS,
        LABELS,
        CONFORMERS,
        MOL_GRAPHS,
        LINE_GRAPHS,
        DIST_GRAPHS,
    )

    df = pd.read_csv(CLEANED_B3DB)
    desc = pd.read_csv(DESCRIPTORS)
    labels = pd.read_csv(LABELS)

    with open(CONFORMERS, "rb") as f:
        conformers = pickle.load(f)
    with open(MOL_GRAPHS, "rb") as f:
        mol_graphs = pickle.load(f)
    with open(LINE_GRAPHS, "rb") as f:
        line_graphs = pickle.load(f)
    with open(DIST_GRAPHS, "rb") as f:
        dist_graphs = pickle.load(f)

    geom_features = None
    if GEOM_FEATURES.exists():
        with open(GEOM_FEATURES, "rb") as f:
            geom_features = pickle.load(f)

    n = len(df)
    sizes = {
        "cleaned_b3db.csv": n,
        "descriptors.csv": len(desc),
        "labels.csv": len(labels),
        "conformers.pkl": len(conformers),
        "molecular_graphs.pkl": len(mol_graphs),
        "line_graphs.pkl": len(line_graphs),
        "distance_graphs.pkl": len(dist_graphs),
    }
    if geom_features is not None:
        sizes["geometry_features.pkl"] = len(geom_features)

    print("Row counts:", sizes)
    assert len(set(sizes.values())) == 1, "processed files have different lengths"
    assert list(desc.columns) == DESCRIPTOR_COLUMNS, "unexpected descriptor columns"
    assert not desc.isna().any().any(), "NaN in descriptors"
    assert (labels["label"].to_numpy() == df["label"].to_numpy()).all(), "labels.csv != cleaned_b3db.csv"

    for i, smi in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        conf_mol = conformers[i]["mol"]
        heavy_z = [a.GetAtomicNum() for a in mol.GetAtoms()]

        assert conformers[i]["smiles"] == smi, f"row {i}: conformer smiles mismatch"
        assert mol_graphs[i].num_nodes == mol.GetNumAtoms(), f"row {i}: 2D graph size"
        assert mol_graphs[i].x.shape[1] == 4, f"row {i}: 2D graph feature dim"
        assert line_graphs[i].num_nodes == mol.GetNumBonds(), f"row {i}: line graph size"
        assert line_graphs[i].x.shape[1] == 3, f"row {i}: line graph feature dim"

        # Multi-scale distance graphs check
        d_item = dist_graphs[i]
        if isinstance(d_item, (list, tuple)):
            assert len(d_item) == 4, f"row {i}: expected 4 multi-scale graphs (2 A, 3 A, 4 A, 5 A)"
            for scale_idx, g_scale in enumerate(d_item):
                assert g_scale.num_nodes == conf_mol.GetNumAtoms(), f"row {i} scale {scale_idx}: atom count"
                assert g_scale.x.shape[1] == 1, f"row {i} scale {scale_idx}: feature dim"
        else:
            assert d_item.num_nodes == conf_mol.GetNumAtoms(), f"row {i}: 3D graph size"

        # Geometric features check
        if geom_features is not None:
            g_feat = geom_features[i]
            comb = g_feat["combined"] if isinstance(g_feat, dict) else g_feat
            assert len(comb) == 24, f"row {i}: expected 24 geometric features"

        assert abs(Descriptors.MolWt(mol) - desc["molecular_weight"][i]) < 1e-3, f"row {i}: descriptors"

    for split in ["scaffold", "random"]:
        idx = {}
        for part in ["train", "val", "test"]:
            s = pd.read_csv(SPLITS_DIR / split / f"{part}.csv")
            rows = s["idx"].to_numpy()
            assert (df["smiles"].to_numpy()[rows] == s["smiles"].to_numpy()).all(), f"{split}/{part} smiles"
            assert (df["label"].to_numpy()[rows] == s["label"].to_numpy()).all(), f"{split}/{part} labels"
            idx[part] = set(rows.tolist())
        assert not (
            idx["train"] & idx["val"] or idx["train"] & idx["test"] or idx["val"] & idx["test"]
        ), f"{split}: overlapping splits"
        assert set().union(*idx.values()) == set(range(n)), f"{split}: splits do not cover all rows"
        print(f"Split {split}: OK ({len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])})")

    print("verify_data: ALL DATA INTEGRITY CHECKS PASSED")


def _random_graph(num_nodes, in_dim, with_edge_attr=False, max_d=5.0):
    src = torch.randint(0, num_nodes, (3 * num_nodes,))
    dst = torch.randint(0, num_nodes, (3 * num_nodes,))
    g = Data(x=torch.rand(num_nodes, in_dim), edge_index=torch.stack([src, dst]))
    if with_edge_attr:
        g.edge_attr = torch.rand(g.edge_index.size(1), 1) * max_d
    return g


def shape_trace(batch_size=4):
    """Pushes a multi-view batch through the full upgraded model and prints/asserts all tensor shapes."""
    from models.bbbp_model import BBBModel

    model = BBBModel().eval()

    # 1. 2D Molecular Graph batch
    graph = Batch.from_data_list([_random_graph(12 + i, 4) for i in range(batch_size)])

    # 2. Line Graph batch
    line = Batch.from_data_list([_random_graph(13 + i, 3) for i in range(batch_size)])

    # 3. Multi-scale 3D Distance Graphs: 4 batches (2 A, 3 A, 4 A, 5 A)
    cutoffs = [2.0, 3.0, 4.0, 5.0]
    dist_multiscale = [
        Batch.from_data_list([_random_graph(20 + i, 1, with_edge_attr=True, max_d=c) for i in range(batch_size)])
        for c in cutoffs
    ]

    # 4. Physicochemical descriptors [B, 7]
    desc = torch.rand(batch_size, 7)

    # 5. Geometric features [B, 24] (distances, angles, torsions)
    geom = torch.rand(batch_size, 24)

    def show(name, t, expected=None):
        print(f"{name:<28}{tuple(t.shape)}")
        if expected is not None:
            assert tuple(t.shape) == expected, f"{name}: {tuple(t.shape)} != {expected}"

    B = batch_size
    with torch.no_grad():
        print("\n--- Input Tensors ---")
        show("graph.x (2D nodes)", graph.x)
        show("line.x (bond nodes)", line.x)
        for idx, (c, g) in enumerate(zip(cutoffs, dist_multiscale)):
            show(f"dist_{c}A.edge_attr", g.edge_attr)
        show("descriptors", desc, (B, 7))
        show("geometry (dist/ang/dih)", geom, (B, 24))

        print("\n--- Sub-Module Encoders ---")
        h2d = model.graph_encoder(graph)
        show("h2d (GIN 2D)", h2d, (B, 128))

        hline = model.line_encoder(line)
        show("hline (Line GIN)", hline, (B, 128))

        # Multi-scale distance graphs + Cutoff Attention
        scale_embs = [model.distance_encoder(g) for g in dist_multiscale]
        for idx, (c, emb) in enumerate(zip(cutoffs, scale_embs)):
            show(f"h3d_scale_{c}A", emb, (B, 128))

        h3d = model.cutoff_attention(scale_embs)
        show("h3d (Cutoff-Attended)", h3d, (B, 128))
        cutoff_weights = model.cutoff_attention.get_last_weights()
        assert cutoff_weights.shape == (B, 4), f"cutoff weights shape {cutoff_weights.shape} != {(B, 4)}"
        print(f"cutoff_attention_weights: shape={cutoff_weights.shape}")

        hdesc = model.descriptor_encoder(desc)
        show("hdesc (Descriptors)", hdesc, (B, 128))

        hgeom = model.geometry_encoder(geom)
        show("hgeom (Geometry)", hgeom, (B, 128))

        print("\n--- Multi-View Attention Fusion & Classification ---")
        fused = model.fusion(h2d=h2d, hline=hline, h3d=h3d, hdesc=hdesc, hgeom=hgeom)
        show("fused (5-view attention)", fused, (B, 128))
        view_weights = model.fusion.get_last_weights()
        assert view_weights.shape == (B, 5), f"view weights shape {view_weights.shape} != {(B, 5)}"
        print(f"view_attention_weights:   shape={view_weights.shape}")

        adapted = model.adapter(fused)
        show("adapted (Bottleneck PEFT)", adapted, (B, 128))

        logits = model.classifier(adapted)
        show("logits (Prediction)", logits, (B, 1))

        # Full forward call verification
        full = model(
            graph_data=graph,
            line_data=line,
            distance_data=dist_multiscale,
            descriptors=desc,
            geometry_data=geom,
        )
        assert torch.allclose(full, logits, atol=1e-5), "forward() differs from trace"

        # Adaptive sparse spatial model verification
        print("\n--- Proposed Adaptive Sparse Spatial-Edge Model ---")
        adapt_model = BBBModel(use_adaptive_spatial=True, edge_budget_ratio=0.5).eval()
        adapt_out = adapt_model(
            graph_data=graph,
            line_data=line,
            distance_data=dist_multiscale,
            descriptors=desc,
            geometry_data=geom,
        )
        show("adapt_logits (Prediction)", adapt_out, (B, 1))
        edge_stats = adapt_model.last_edge_stats
        print(f"adaptive_edge_stats: candidates={edge_stats.get('candidates')}, retained={edge_stats.get('retained')}, bonded={edge_stats.get('bonded')}, sparsity={edge_stats.get('sparsity'):.2%}")

    print("\nshape_trace: ALL MULTI-VIEW SHAPES, BASELINE, AND ADAPTIVE CHECKS PASSED!\n")
