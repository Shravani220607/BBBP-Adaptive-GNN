"""BBBModel: Comprehensive Multi-View Graph Neural Network for Blood-Brain Barrier Permeability.
Integrates 5 views:
1. 2D Molecular Graph (GIN encoder)
2. Line Graph (Line graph encoder for bond angles/connectivity)
3. Multi-scale 3D Distance Graphs (2 A, 3 A, 4 A, 5 A with learned CutoffAttention)
4. Physicochemical Descriptors (7 RDKit descriptors via MLP encoder)
5. Geometric Features (Pairwise distances, bond angles, torsions via GeometryEncoder)

Includes:
- Multi-scale distance graphs (2 A, 3 A, 4 A, 5 A)
- Learned Cutoff Attention mechanism
- Explicit 3D GeometryEncoder (distances, angles, torsions)
- Dynamic Attention Fusion with attention cache
- Bottleneck Adapter (PEFT)
- Classifier head
- Configurable flags for ablation studies
"""

import torch
import torch.nn as nn

from models.encoders.gin_encoder import GINEncoder
from models.encoders.line_encoder import LineEncoder
from models.encoders.distance_encoder import DistanceEncoder
from models.encoders.descriptor_encoder import DescriptorEncoder
from models.encoders.geometry_encoder import GeometryEncoder
from models.adaptive_cutoff.cutoff_network import CutoffAttention, SpatialEdgeSelector
from models.fusion.attention_fusion import AttentionFusion
from models.adapters.peft_adapter import Adapter
from models.classifier.bbb_classifier import BBBClassifier


class BBBModel(nn.Module):
    def __init__(
        self,
        ablation_mode="full",
        use_2d=True,
        use_line=True,
        use_3d=True,
        use_desc=True,
        use_geom=True,
        use_cutoff_attn=True,
        use_adaptive_spatial=False,
        edge_budget_ratio=0.5,
        preserve_bonds=True,
    ):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.use_adaptive_spatial = use_adaptive_spatial or ablation_mode.startswith("adaptive")
        self.edge_budget_ratio = edge_budget_ratio
        self.preserve_bonds = preserve_bonds

        # Parse ablation mode presets if provided
        if ablation_mode == "2d_only":
            use_2d = True
            use_line = False
            use_3d = False
            use_desc = False
            use_geom = False
            use_cutoff_attn = False
        elif ablation_mode == "2d_3d":
            use_2d = True
            use_3d = True
            use_line = False
            use_desc = False
            use_geom = False
        elif ablation_mode == "2d_3d_line":
            use_2d = True
            use_3d = True
            use_line = True
            use_desc = False
            use_geom = False
        elif ablation_mode == "no_cutoff_attn":
            use_cutoff_attn = False
        elif ablation_mode == "no_geometry":
            use_geom = False
        elif ablation_mode == "full":
            use_2d = True
            use_line = True
            use_3d = True
            use_desc = True
            use_geom = True
            use_cutoff_attn = True

        self.use_2d = use_2d
        self.use_line = use_line
        self.use_3d = use_3d
        self.use_desc = use_desc
        self.use_geom = use_geom
        self.use_cutoff_attn = use_cutoff_attn

        # 1. 2D Molecular Graph Encoder (atom features: [atomic_num, degree, formal_charge, aromatic] -> 4)
        if self.use_2d:
            self.graph_encoder = GINEncoder(input_dim=4)

        # 2. Line Graph Encoder (bond features: [bond_type, conjugated, in_ring] -> 3)
        if self.use_line:
            self.line_encoder = LineEncoder(input_dim=3)

        # 3. 3D Distance Graph Encoder & Adaptive Spatial Edge Selector / Cutoff Attention
        if self.use_3d:
            self.distance_encoder = DistanceEncoder(input_dim=1)
            if self.use_adaptive_spatial:
                self.edge_selector = SpatialEdgeSelector(
                    node_in_dim=1,
                    budget_ratio=self.edge_budget_ratio,
                    preserve_bonds=self.preserve_bonds,
                )
            else:
                self.cutoff_attention = CutoffAttention(
                    embedding_dim=128,
                    num_cutoffs=4,
                    use_attention=self.use_cutoff_attn,
                )

        # 4. Physicochemical Descriptors Encoder (7 RDKit descriptors)
        if self.use_desc:
            self.descriptor_encoder = DescriptorEncoder(input_dim=7)

        # 5. Geometric Features Encoder (distances, angles, torsions)
        if self.use_geom:
            self.geometry_encoder = GeometryEncoder(
                dist_dim=8, angle_dim=8, torsion_dim=8, hidden_dim=32, output_dim=128
            )

        # Multi-view attention fusion & adapter & classifier
        self.fusion = AttentionFusion(embedding_dim=128)
        self.adapter = Adapter()
        self.classifier = BBBClassifier()
        self.last_edge_stats = {}

    def get_regularization_loss(self):
        """Auxiliary edge sparsity penalty loss during training."""
        if self.use_adaptive_spatial and hasattr(self, "edge_selector"):
            return self.edge_selector.last_budget_loss
        return torch.tensor(0.0)

    def forward(
        self,
        graph_data=None,
        line_data=None,
        distance_data=None,
        descriptors=None,
        geometry_data=None,
    ):
        """Forward pass supporting baseline multi-scale cutoff graphs or adaptive spatial selection."""
        # View 1: 2D Graph
        h2d = self.graph_encoder(graph_data) if (self.use_2d and graph_data is not None) else None

        # View 2: Line Graph
        hline = self.line_encoder(line_data) if (self.use_line and line_data is not None) else None

        # View 3: 3D Spatial Geometry
        h3d = None
        if self.use_3d and distance_data is not None:
            if self.use_adaptive_spatial:
                cand_graph = distance_data[-1] if isinstance(distance_data, (list, tuple)) else distance_data
                if hasattr(cand_graph, "is_bonded") and cand_graph.is_bonded is not None:
                    is_bonded = cand_graph.is_bonded
                elif graph_data is not None and hasattr(graph_data, "edge_index"):
                    max_n = cand_graph.num_nodes + 1
                    b_u, b_v = graph_data.edge_index[0], graph_data.edge_index[1]
                    bonded_keys = set((b_u.long() * max_n + b_v.long()).cpu().tolist())
                    c_u, c_v = cand_graph.edge_index[0], cand_graph.edge_index[1]
                    c_keys = (c_u.long() * max_n + c_v.long()).cpu().tolist()
                    is_bonded = torch.tensor(
                        [[1.0] if k in bonded_keys else [0.0] for k in c_keys],
                        dtype=torch.float,
                        device=cand_graph.edge_index.device,
                    )
                else:
                    is_bonded = torch.zeros((cand_graph.edge_index.size(1), 1), dtype=torch.float, device=cand_graph.edge_index.device)

                sel_idx, sel_attr, sel_w = self.edge_selector(
                    cand_graph.x,
                    cand_graph.edge_index,
                    cand_graph.edge_attr,
                    is_bonded,
                    batch=cand_graph.batch,
                )
                self.last_edge_stats = getattr(self.edge_selector, "last_edge_stats", {})
                h3d = self.distance_encoder(cand_graph, edge_index=sel_idx, edge_attr=sel_attr, edge_weight=sel_w)
            else:
                if isinstance(distance_data, (list, tuple)):
                    scale_embs = [self.distance_encoder(g) for g in distance_data]
                    h3d = self.cutoff_attention(scale_embs)
                else:
                    h3d = self.distance_encoder(distance_data)

        # View 4: Descriptors
        hdesc = (
            self.descriptor_encoder(descriptors)
            if (self.use_desc and descriptors is not None)
            else None
        )

        # View 5: Geometric Features (distances, angles, torsions)
        hgeom = (
            self.geometry_encoder(geometry_data)
            if (self.use_geom and geometry_data is not None)
            else None
        )

        # Fuse active views with learned attention weights
        fused = self.fusion(
            h2d=h2d,
            hline=hline,
            h3d=h3d,
            hdesc=hdesc,
            hgeom=hgeom,
        )

        adapted = self.adapter(fused)
        return self.classifier(adapted)  # [B, 1] logits
