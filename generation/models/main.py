from torch.nn import init
from torch.optim import lr_scheduler

from .generator import ResnetGenerator
from .discriminator import NLayerDiscriminator

def init_weights(net):
    gain=0.02
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def init_net(net, device):
    net.to(device)
    init_weights(net)
    return net

def get_scheduler(optimizer, opt):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + opt.start_epoch - opt.iters) / float(opt.iters_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

def Generator(input_channels, output_channels, ngf, device):
    net = ResnetGenerator(input_channels, output_channels, ngf)
    return init_net(net, device)

def Discriminator(input_channels, ndf, device):
    net = NLayerDiscriminator(input_channels, ndf)
    return init_net(net, device)
