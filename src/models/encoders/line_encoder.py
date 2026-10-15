from models.encoders.gin_encoder import GINEncoder


class LineEncoder(GINEncoder):
    def __init__(self, input_dim=3, **kwargs):
        super().__init__(
            input_dim=input_dim,
            use_distance=False,
            **kwargs,
        )