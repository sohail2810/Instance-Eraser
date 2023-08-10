from __future__ import print_function
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from .models.main import Generator, Discriminator, get_scheduler
from .models.loss import GANLoss

from .data import get_training_set, get_test_set

cudnn.benchmark = True
torch.cuda.manual_seed(123)

class Args:
    batch_size = 4
    test_batch_size = 8
    input_channels = 3
    output_channels = 3
    ngf = 64
    ndf = 64
    start_epoch = 1
    iters = 100
    iters_decay = 100
    lr = 0.0002
    lr_policy = 'lambda'
    lr_decay_iters = 50
    beta1 = 0.5
    cuda = False
    threads = 1
    seed = 123
    lamb = 10

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":
    opt = Args()
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    print('===> Loading datasets')
    train_set = get_training_set()
    test_set = get_test_set()
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, collate_fn=collate_fn, batch_size=opt.batch_size, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, collate_fn=collate_fn, batch_size=opt.test_batch_size, shuffle=False)
    print('===> Building models')
    net_g = Generator(opt.input_channels, opt.output_channels, opt.ngf, device)
    net_d = Discriminator(opt.input_channels + opt.output_channels, opt.ndf, device)

    gan_loss = GANLoss().to(device)
    L1_loss = nn.L1Loss().to(device)
    mse_loss = nn.MSELoss().to(device)

    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, opt)
    net_d_scheduler = get_scheduler(optimizer_d, opt)

    for epoch in range(opt.start_epoch, opt.iters + opt.iters_decay + 1):
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a, real_b = batch[0].to(device), batch[1].to(device)
            fake_b = net_g(real_a)

            optimizer_d.zero_grad()
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab.detach())
            loss_d_fake = gan_loss(pred_fake, False)
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = net_d.forward(real_ab)
            loss_d_real = gan_loss(pred_real, True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab)
            loss_g_gan = gan_loss(pred_fake, True)
            loss_g_l1 = L1_loss(fake_b, real_b) * opt.lamb
            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            optimizer_g.step()

            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))
            if iteration % 1000 == 0:
                net_g_model_out_path = "checkpoints/netG_model_epoch_{}.pth".format(epoch)
                net_d_model_out_path = "checkpoints/netD_model_epoch_{}.pth".format(epoch)
                torch.save(net_g, net_g_model_out_path)
                torch.save(net_d, net_d_model_out_path)

        net_g_scheduler.step()
        net_d_scheduler.step()

        avg_psnr = 0
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = net_g(input)
            mse = mse_loss(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
