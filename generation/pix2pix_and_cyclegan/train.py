"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
import os
import matplotlib.pyplot as plt

class BackgroundDataset(Dataset):

    def __init__(self, 
                 input_dir = "/home/rrachala/private/instance_eraser/train_inputs",
                 label_dir = "/datasets/COCO-2017/train_2017/",
                 transform = None
                ):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data = []
        self.input_dir = input_dir
        self.label_dir = label_dir
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.jpg')):
                self.data.append(filename)
        if transform is None:
            transform_list = [
                            transforms.ToTensor(),
                            transforms.Resize(size=(256, 256)),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_file = self.data[idx]
        img_name = os.path.join(self.input_dir, img_file)
        gen_image = plt.imread(img_name)
        gen_image = gen_image/255
        
        img_name = os.path.join(self.label_dir, img_file)
        image = plt.imread(img_name)
        image = image/255

        gen_img = self.transform(gen_image.astype(np.float32))
        image = self.transform(image.astype(np.float32))
        
        data = {
            "A": gen_img,
            "B": image,
            "A_paths": ""
            
        }
        return data
    
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_training_set(input_dir = "/home/rrachala/private/instance_eraser/inputs",
                 label_dir = "/datasets/COCO-2017/anno2017/instances_train2017.json",
                 transform = None
                 ):

    return BackgroundDataset(input_dir, label_dir, transform)


def get_test_set(input_dir = "/home/rrachala/private/instance_eraser/inputs",
                 label_dir = "/datasets/COCO-2017/val2017",
                 transform = None
                 ):

    return BackgroundDataset(input_dir, label_dir, transform)

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
#     dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    pretrained_path = "/home/rrachala/private/pytorch-CycleGAN-and-pix2pix/checkpoints/facades_label2photo_pretrained/latest_net_G.pth"
    
    print('===> Loading datasets')
    # train_set = get_training_set()
    test_set = get_test_set()
    # training_data_loader = DataLoader(dataset=train_set, num_workers=1, collate_fn=collate_fn, batch_size=64, shuffle=True)
    dataset = DataLoader(dataset=test_set, num_workers=1, collate_fn=collate_fn, batch_size=8, shuffle=False)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    state_dict = torch.load(pretrained_path, map_location=str(model.device))
    model.netG.module.load_state_dict(state_dict)
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
