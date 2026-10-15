"""CutoffAttention: Learns attention weights over multi-scale distance graph embeddings
(2 A, 3 A, 4 A, 5 A) to produce a unified, weighted 3D geometric embedding.
Stores attention weights for visualization and explainability.
"""

import torch
import torch.nn as nn
import numpy as np

CUTOFF_LABELS = ["2 Å", "3 Å", "4 Å", "5 Å"]


class CutoffAttention(nn.Module):
    """Attention layer over multi-scale distance graph representations.

    Input:
        embeddings: Tensor of shape [B, 4, embedding_dim] or list of 4 tensors [B, embedding_dim]
    Output:
        weighted_embedding: Tensor of shape [B, embedding_dim]
    """

    def __init__(self, embedding_dim=128, num_cutoffs=4, use_attention=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_cutoffs = num_cutoffs
        self.use_attention = use_attention

        # Simple, expressive attention layer
        self.attn_net = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Store last attention weights for inspection, visualization, and Streamlit demo
        self.last_weights = None  # [B, 4]

    def forward(self, embeddings):
        """Args:

        embeddings: [B, 4, D] or list of 4 tensors each [B, D].
        """
        if isinstance(embeddings, (list, tuple)):
            # Stack along scale dimension
            embeddings = torch.stack(embeddings, dim=1)  # [B, 4, D]

        if not self.use_attention:
            # Ablation fallback: uniform average pooling without learned attention
            b, s, _ = embeddings.shape
            self.last_weights = torch.full((b, s), 1.0 / s, device="cpu")
            return embeddings.mean(dim=1)

        # Compute unnormalized attention scores: [B, 4, 1]
        scores = self.attn_net(embeddings)

        # Normalize with softmax across cutoffs
        weights = torch.softmax(scores, dim=1)  # [B, 4, 1]

        # Cache weights for visualization
        self.last_weights = weights.squeeze(-1).detach().cpu()  # [B, 4]

        # Compute weighted sum
        weighted_embedding = torch.sum(weights * embeddings, dim=1)  # [B, D]
        return weighted_embedding

    def get_last_weights(self):
        """Returns the most recent attention weights as numpy array or dict."""
        if self.last_weights is None:
            return None
        return self.last_weights.numpy()


class GaussianRBF(nn.Module):
    def __init__(self, num_centers=16, max_dist=8.0):
        super().__init__()
        centers = torch.linspace(0.0, max_dist, num_centers)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / float(centers[1] - centers[0]) ** 2

    def forward(self, dist):
        return torch.exp(-self.gamma * (dist - self.centers) ** 2)


class SpatialEdgeSelector(nn.Module):
    """Adaptive, budget-constrained sparse spatial-edge selector with bond preservation."""

    def __init__(
        self,
        node_in_dim=1,
        node_emb_dim=32,
        rbf_centers=16,
        hidden_dim=64,
        budget_ratio=0.5,
        preserve_bonds=True,
    ):
        super().__init__()
        self.budget_ratio = budget_ratio
        self.preserve_bonds = preserve_bonds
        self.node_proj = nn.Linear(node_in_dim, node_emb_dim)
        self.rbf = GaussianRBF(num_centers=rbf_centers)
        in_dim = 2 * node_emb_dim + rbf_centers + 1
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.last_budget_loss = torch.tensor(0.0)
        self.last_edge_stats = {}

    def forward(self, x, edge_index, edge_dist, is_bonded, batch=None):
        import time
        t0 = time.perf_counter()
        E = edge_index.size(1)
        if E == 0:
            self.last_edge_stats = {"candidates": 0, "retained": 0, "sparsity": 0.0, "time_ms": 0.0}
            return edge_index, edge_dist, torch.ones((0, 1), device=edge_index.device)

        h_nodes = self.node_proj(x)
        src, dst = edge_index[0], edge_index[1]
        dist_rbf = self.rbf(edge_dist.view(-1, 1))
        is_bonded_feat = is_bonded.view(-1, 1)

        edge_feat = torch.cat([h_nodes[src], h_nodes[dst], dist_rbf, is_bonded_feat], dim=-1)
        scores = self.scorer(edge_feat).squeeze(-1)

        is_bonded_mask = (is_bonded.view(-1) > 0.5)
        num_bonded = int(is_bonded_mask.sum().item())
        target_budget = max(num_bonded if self.preserve_bonds else 1, int(np.ceil(self.budget_ratio * E)))

        if self.preserve_bonds:
            non_bonded_idx = torch.nonzero(~is_bonded_mask).view(-1)
            bonded_idx = torch.nonzero(is_bonded_mask).view(-1)
            k_spatial = max(0, target_budget - num_bonded)

            if len(non_bonded_idx) > 0 and k_spatial < len(non_bonded_idx):
                nb_scores = scores[non_bonded_idx]
                topk_rel = torch.topk(nb_scores, k_spatial, sorted=False).indices
                selected_nb_idx = non_bonded_idx[topk_rel]
            else:
                selected_nb_idx = non_bonded_idx

            selected_indices = torch.cat([bonded_idx, selected_nb_idx]) if len(selected_nb_idx) > 0 else bonded_idx
        else:
            k = min(E, target_budget)
            selected_indices = torch.topk(scores, k, sorted=False).indices

        # Straight-Through estimator during training for differentiable selection
        if self.training:
            probs = torch.sigmoid(scores)
            hard_mask = torch.zeros(E, device=scores.device)
            hard_mask[selected_indices] = 1.0
            st_weights = hard_mask + probs - probs.detach()
            if self.preserve_bonds and num_bonded > 0:
                st_weights[bonded_idx] = 1.0

            # Sparsity penalty loss
            target_p = float(self.budget_ratio)
            self.last_budget_loss = 0.1 * torch.relu(probs.mean() - target_p) ** 2
            out_weights = st_weights.view(-1, 1)
            out_edge_index = edge_index
            out_edge_attr = edge_dist
        else:
            self.last_budget_loss = torch.tensor(0.0, device=scores.device)
            selected_indices = selected_indices.sort().values
            out_edge_index = edge_index[:, selected_indices]
            out_edge_attr = edge_dist[selected_indices]
            out_weights = torch.ones((out_edge_index.size(1), 1), device=edge_index.device)

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        retained = out_edge_index.size(1)
        self.last_edge_stats = {
            "candidates": E,
            "retained": retained,
            "bonded": num_bonded,
            "sparsity": float(1.0 - (retained / max(1, E))),
            "time_ms": t_elapsed,
        }
        return out_edge_index, out_edge_attr, out_weights

