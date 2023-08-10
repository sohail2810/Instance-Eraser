import torch.nn as nn
import functools

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf = 1
        nf_prev = 1
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_prev, ndf * nf,
                          kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf),
                nn.LeakyReLU(0.2, True)
            ]

        nf_prev = nf
        nf = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_prev, ndf * nf,
                      kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)