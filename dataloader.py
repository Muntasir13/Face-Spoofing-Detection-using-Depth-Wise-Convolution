import os
import cv2
import json
import torch
import torchvision
from config import *
from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms



class MyImageFolder(datasets.ImageFolder):
    '''
        This class is used to not add the image to the batch if an image is not supported
    '''
    __init__ = datasets.ImageFolder.__init__
    def __getitem__(self, index):
        try:
            return super(MyImageFolder, self).__getitem__(index)
        except Exception as e:
            pass


class DatasetLoader:
    '''
        The Dataset Loader class.
        arguments:
            folder_path: the path to the dataset folder (type: string)
            batch_size: the size of batch (type: int)
            shuffle: should the dataset loading be shuffled (type: bool)
            drop_last: should the last items be dropped if the batch size does not equal to batch_size (type: bool)
    '''
    def __init__(self, folder_path,  batch_size, shuffle, drop_last):
        self.location = folder_path
        self.batch_size = batch_size
        self.shuffle=shuffle
        self.drop_last=drop_last

    def my_collate(self, batch):
        batch = list(filter(lambda x : x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def load_dataset(self):
        transform = transforms.Compose(
            [
                transforms.Resize((image_width, image_height)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
        dataset = MyImageFolder(root=self.location, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last, collate_fn=self.my_collate)