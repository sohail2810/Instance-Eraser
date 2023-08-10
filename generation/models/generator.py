import torch.nn as nn
import functools

class Inconv(nn.Module):
    def __init__(self, input_channels, output_channels, norm_layer, use_bias):
        super(Inconv, self).__init__()
        self.inconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=0,
                      bias=use_bias),
            norm_layer(output_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.inconv(x)
        return x
      
class Outconv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.outconv(x)
        return x

class Up(nn.Module):
    def __init__(self, input_channels, output_channels, norm_layer, use_bias):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=use_bias),
            norm_layer(output_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
class Down(nn.Module):
    def __init__(self, input_channels, output_channels, norm_layer, use_bias):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3,
                      stride=2, padding=1, bias=use_bias),
            norm_layer(output_channels),
            nn.ReLU(True)
        )
    def forward(self, x):
        x = self.down(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_bias)

    def build_conv_block(self, dim, norm_layer, use_bias):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)
    
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.inc = Inconv(input_nc, ngf, norm_layer, use_bias)
        self.down1 = Down(ngf, ngf * 2, norm_layer, use_bias)
        self.down2 = Down(ngf * 2, ngf * 4, norm_layer, use_bias)

        model = []
        for _ in range(n_blocks):
            model += [ResBlock(ngf * 4, norm_layer=norm_layer, use_bias=use_bias)]
        self.resblocks = nn.Sequential(*model)

        self.up1 = Up(ngf * 4, ngf * 2, norm_layer, use_bias)
        self.up2 = Up(ngf * 2, ngf, norm_layer, use_bias)

        self.outc = Outconv(ngf, output_nc)

    def forward(self, input):
        out = {}
        out['in'] = self.inc(input)
        out['d1'] = self.down1(out['in'])
        out['d2'] = self.down2(out['d1'])
        out['bottle'] = self.resblocks(out['d2'])
        out['u1'] = self.up1(out['bottle'])
        out['u2'] = self.up2(out['u1'])
        return self.outc(out['u2'])