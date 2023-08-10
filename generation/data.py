from os.path import join

from dataset import BackgroundDataset


def get_training_set(input_dir = "training_inputs",
                 label_dir = "/datasets/COCO-2017/val2017/",
                 transform = None
                 ):

    return BackgroundDataset(input_dir, label_dir, transform)


def get_test_set(input_dir = "testing_inputs",
                 label_dir = "/datasets/COCO-2017/val2017/",
                 transform = None
                 ):

    return BackgroundDataset(input_dir, label_dir, transform)
