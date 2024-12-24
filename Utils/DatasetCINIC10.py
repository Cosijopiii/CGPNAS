import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from pytorch_cinic.dataset import CINIC10

from Utils.BaseDataset import BaseDataset


class datasetCINIC10(BaseDataset):
    """
    CINIC-10 dataset class.
    """
    def __init__(self, validation=False, download=True, seed=0, pin_memory=True, num_workers=0, cutout=16,train_size=80,batch_size_train=128,batch_size_test=128):
        normalization = transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        super().__init__(10, 3, 0.2, batch_size_train, batch_size_test, normalization, transform_train, transform_test,
                         './data/cinic-10', validation, download, seed, pin_memory, num_workers, cutout, train_size)

    def get_trainset(self):
        """
        Returns the CINIC-10 training dataset.
        """
        return CINIC10(root=self.root, partition="train", download=self.download, transform=self.transform_train)

    def get_validationset(self):
        """
        Returns the CINIC-10 validation dataset.
        """
        return CINIC10(root=self.root, partition="train", download=self.download, transform=self.transform_test)

    def get_testset(self):
        """
        Returns the CINIC-10 test dataset.
        """
        return CINIC10(root=self.root, partition="test", download=self.download, transform=self.transform_test)