from models.encoders.gin_encoder import GINEncoder


class DistanceEncoder(GINEncoder):
    def __init__(self, input_dim=1, **kwargs):
        super().__init__(
            input_dim=input_dim,
            use_distance=True,
            **kwargs,
        )