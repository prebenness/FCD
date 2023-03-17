from torch import nn
from src.models.primaries.input_embedders.resnet import ResNetEmbedder
from src.models.primaries.separation_mechanisms.feedforward import \
    FeedForwardSeparator

import src.config as cfg


class SimplePrimary(nn.Module):
    def __init__(self, *args, dim_c=512, dim_s=512, **kwargs):
        super().__init__(*args, **kwargs)

        # ResNet18 embedder
        self.embedder = ResNetEmbedder(
            resnet_type=18, num_channels=cfg.NUM_CHANNELS
        )
        # Simple feedforward separator
        self.separator = FeedForwardSeparator(
            in_features=512, out_features=[dim_c, dim_s]
        )

    def forward(self, x):
        r = self.embedder(x)
        c, s = self.separator(r)

        return c, s
