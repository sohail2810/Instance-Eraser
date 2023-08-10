"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
import os
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')
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
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
#     dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    test_set = get_test_set()
    # training_data_loader = DataLoader(dataset=train_set, num_workers=1, collate_fn=collate_fn, batch_size=64, shuffle=True)
    dataset = DataLoader(dataset=test_set, num_workers=1, collate_fn=collate_fn, batch_size=1, shuffle=False)
    print(len(dataset))
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = f"img_{i}"#model.get_image_paths()     # get image paths
#         if i % 5 == 0:  # save images to an HTML file
        print('processing (%d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML
