import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
import os
import matplotlib.pyplot as plt


class BackgroundDataset(Dataset):

    def __init__(self, 
                 input_dir = "/training_inputs",
                 label_dir = "/datasets/COCO-2017/train2017/",
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
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            img_file = self.data[idx]
            img_name = os.path.join(self.input_dir, img_file)
            gen_image = plt.imread(img_name)
            gen_image = gen_image[:,:,:3]/255

            img_name = os.path.join(self.label_dir, img_file)
            img_name = img_name.replace(".png", ".jpg")
            image = plt.imread(img_name)
            image = image/255

            gen_img = self.transform(gen_image.astype(np.float32))
            image = self.transform(image.astype(np.float32))

            return gen_img, image
        except Exception as err:
            return None