import torch
import torch.nn as nn

from torch_geometric.nn import GINConv, GINEConv, global_mean_pool


class GaussianRBF(nn.Module):
    """Expands a distance (in Angstrom) into a vector of Gaussian bumps."""

    def __init__(self, num_centers=16, max_dist=8.0):
        super().__init__()
        centers = torch.linspace(0.0, max_dist, num_centers)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / float(centers[1] - centers[0]) ** 2
        self.num_centers = num_centers

    def forward(self, dist):
        # dist: [E, 1] -> [E, num_centers]
        return torch.exp(-self.gamma * (dist - self.centers) ** 2)


class GINEncoder(nn.Module):
    """GIN encoder: graph -> 128-d embedding.

    use_distance=False : plain GINConv (molecular graph, line graph)
    use_distance=True  : GINEConv that also uses edge_attr (= distance in Angstrom),
                         needed for the 3D distance graph. Plain GINConv ignores
                         edge_attr, so the distances would never be used.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=128,
        num_layers=3,
        use_distance=False,
    ):
        super().__init__()

        self.use_distance = use_distance
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if use_distance:
            self.rbf = GaussianRBF()
            self.edge_proj = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if use_distance:
                self.convs.append(GINEConv(mlp))
                self.edge_proj.append(nn.Linear(self.rbf.num_centers, hidden_dim))
            else:
                self.convs.append(GINConv(mlp))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.projection = nn.Linear(hidden_dim, output_dim)

    def node_embeddings(self, x, edge_index, edge_attr=None, edge_weight=None):
        """Per-node embeddings [N, hidden_dim] (also used for pretraining)."""
        h = self.input_proj(x)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if self.use_distance:
                e = self.edge_proj[i](self.rbf(edge_attr))
                if edge_weight is not None:
                    e = e * edge_weight.view(-1, 1)
                h = conv(h, edge_index, e)
            else:
                h = conv(h, edge_index)
            h = torch.relu(norm(h))

        return h

    def forward(self, data, edge_index=None, edge_attr=None, edge_weight=None):
        ei = edge_index if edge_index is not None else data.edge_index
        ea = edge_attr if edge_attr is not None else (data.edge_attr if self.use_distance else None)
        h = self.node_embeddings(data.x, ei, ea, edge_weight=edge_weight)

        # passing num_graphs keeps the output aligned with the labels
        pooled = global_mean_pool(h, data.batch, data.num_graphs)

        return self.projection(pooled)
