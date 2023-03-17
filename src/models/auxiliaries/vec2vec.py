from torch import nn


class Vec2Vec(nn.Module):
    '''
    Vector to vector reconstruction module
    '''

    def __init__(
        self, in_features, out_features, *args, hidden_dim=512, num_hidden=4,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden

        # Input
        self.fc_in = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # Hidden layers
        self.fc_hidden = self._make_hidden_layers()

        # Output
        self.fc_out = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=out_features),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        x = self.fc_in(x)
        x = self.fc_hidden(x)
        x = self.fc_out(x)
        return x

    def _make_hidden_layers(self):
        layers = []
        for _ in self.num_hidden:
            layers += [
                nn.Linear(
                    in_features=self.hidden_dim, out_features=self.hidden_dim
                ),
                nn.LeakyReLU(negative_slope=0.1)
            ]

        return nn.Sequential(*layers)
