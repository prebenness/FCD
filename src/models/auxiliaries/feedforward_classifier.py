from torch import nn


class FeedforwardClassifier(nn.Module):
    def __init__(
        self, input_features, num_classes, *args, num_hidden=2, dim_hidden=512, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Input layer
        self.fc_in = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=dim_hidden),
            nn.LeakyReLU(negative_slope=0.1),
        )

        # Hidden layers
        hidden_layers = []
        for _ in range(num_hidden):
            hidden_layers += [
                nn.Linear(in_features=dim_hidden, out_features=dim_hidden),
                nn.LeakyReLU(negative_slope=0.1),
            ]
        self.fc_hidden = nn.Sequential(*hidden_layers)

        # Output layer
        self.fc_out = nn.Sequential(
            nn.Linear(in_features=dim_hidden, out_features=num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc_in(x)
        x = self.fc_hidden(x)
        x = self.fc_out(x)
        return x
