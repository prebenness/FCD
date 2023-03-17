'''
Simple separation mechanism consisting of a feed forward NN with two outputs
'''

from torch import nn


class FeedForwardSeparator(nn.Module):
    '''
    Out features are given per signal
    '''

    def __init__(
        self, in_features, out_features, *args, dim_hidden=512, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Out features expected per signal stream
        if isinstance(out_features, list):
            out_features_c, out_features_s = out_features
        else:
            # If one number, set same for both
            out_features_c, out_features_s = out_features, out_features

        # x -> r
        self.fch1 = nn.Sequential(
            nn.Linear(
                in_features=in_features, out_features=dim_hidden
            ),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # r -> c
        self.fcc = nn.Sequential(
            nn.Linear(
                in_features=dim_hidden, out_features=out_features_c
            ),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # r -> s
        self.fcs = nn.Sequential(
            nn.Linear(
                in_features=dim_hidden, out_features=out_features_s
            ),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, r):
        h1 = self.fch1(r)
        c = self.fcc(h1)
        s = self.fcs(h1)

        return c, s
