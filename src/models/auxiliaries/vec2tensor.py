from torch import nn

from src.utils.calculate.conv_shape import calc_deconv_arch
import src.config as cfg


class Vec2Tensor(nn.Module):
    def __init__(self, input_features, output_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_features = input_features
        self.output_shape = output_shape

        self.deconvs = self._make_deconv_blocks()

    def _make_deconv_blocks(self):
        # TODO: implement general architecture for vec -> img recon
        # Initially only supports MNIST
        assert cfg.DATASET == 'mnist'

        arch = calc_deconv_arch(
            in_features=self.input_features, out_shape=self.output_shape
        )

        deconvs = []
        for (ic, oc, k, s, p, op) in arch:
            deconvs.append(
                DeConv2dBlock(
                    in_channels=ic, out_channels=oc, kernel_size=k, stride=s,
                    padding=p, output_padding=op
                )
            )

        m = nn.Sequential(*deconvs)

        return m

    def forward(self, x):
        x = x.reshape((-1, 32, 4, 4))
        return self.deconvs(x)


class DeConv2dBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, *args,
        padding=0, output_padding=0,  instance_norm=True, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if instance_norm:
            # affine=True for learnable affine parameters
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.norm = nn.Identity()

        # Initialize activation
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            output_padding=output_padding
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
