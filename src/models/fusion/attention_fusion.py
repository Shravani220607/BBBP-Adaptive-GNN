import torch
import torch.nn as nn


DEFAULT_VIEW_NAMES = [
    "2D Graph",
    "Line Graph",
    "3D MultiScale",
    "Descriptors",
    "Geometry",
]


class AttentionFusion(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.last_weights = None
        self.last_view_names = None

    def forward(
        self,
        h2d=None,
        hline=None,
        h3d=None,
        hdesc=None,
        hgeom=None,
        views_list=None,
        view_names=None,
    ):
        if views_list is None:
            candidates = [
                ("2D Graph", h2d),
                ("Line Graph", hline),
                ("3D MultiScale", h3d),
                ("Descriptors", hdesc),
                ("Geometry", hgeom),
            ]
            views = [value for _, value in candidates if value is not None]
            names = [name for name, value in candidates if value is not None]
        else:
            views = views_list
            names = view_names or [
                f"View_{i}" for i in range(len(views))
            ]

        if not views:
            raise ValueError("No views were provided to AttentionFusion.")

        if len(views) == 1:
            self.last_weights = torch.ones(
                (views[0].shape[0], 1), device="cpu"
            )
            self.last_view_names = names
            return views[0]

        embeddings = torch.stack(views, dim=1)
        scores = self.attention(embeddings)
        weights = torch.softmax(scores, dim=1)

        self.last_weights = weights.squeeze(-1).detach().cpu()
        self.last_view_names = names

        return (weights * embeddings).sum(dim=1)

    def get_last_weights(self):
        if self.last_weights is None:
            return None
        return self.last_weights.numpy()