from torch import nn

import src.config as cfg
from src.models.primaries.simple import SimplePrimary
from src.models.auxiliaries.vec2tensor import Vec2Tensor


class ReconClassifier(nn.Module):
    def __init__(self, *args, dim_c=512, dim_s=512, **kwargs):
        super().__init__(*args, **kwargs)

        # Simple Primary
        self.primary = SimplePrimary(dim_c=512, dim_s=512)

        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=dim_c, out_features=cfg.NUM_CLASSES),
            nn.Sigmoid(),
        )

        # Image reconstructor
        self.reconstructor = Vec2Tensor(
            input_features=dim_s, output_shape=(
                cfg.NUM_CHANNELS, cfg.HEIGHT, cfg.WIDTH
            )
        )

    def forward(self, x):
        c, s = self.primary(x)

        y_pred = self.classifier(c)
        x_rec = self.reconstructor(s)

        return y_pred, x_rec
