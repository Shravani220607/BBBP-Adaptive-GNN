import torch
import torch.nn as nn


class GeometryEncoder(nn.Module):
    def __init__(
        self,
        dist_dim=8,
        angle_dim=8,
        torsion_dim=8,
        hidden_dim=32,
        output_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.dist_dim = dist_dim
        self.angle_dim = angle_dim
        self.torsion_dim = torsion_dim

        self.dist_mlp = nn.Sequential(
            nn.Linear(dist_dim, hidden_dim),
            nn.ReLU(),
        )
        self.angle_mlp = nn.Sequential(
            nn.Linear(angle_dim, hidden_dim),
            nn.ReLU(),
        )
        self.torsion_mlp = nn.Sequential(
            nn.Linear(torsion_dim, hidden_dim),
            nn.ReLU(),
        )

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, dist_feat, angle_feat=None, torsion_feat=None):
        if angle_feat is None and torsion_feat is None:
            angle_start = self.dist_dim
            torsion_start = self.dist_dim + self.angle_dim

            angle_feat = dist_feat[:, angle_start:torsion_start]
            torsion_feat = dist_feat[:, torsion_start:]
            dist_feat = dist_feat[:, :self.dist_dim]

        h_dist = self.dist_mlp(dist_feat)
        h_angle = self.angle_mlp(angle_feat)
        h_torsion = self.torsion_mlp(torsion_feat)

        combined = torch.cat([h_dist, h_angle, h_torsion], dim=-1)
        return self.projector(combined)