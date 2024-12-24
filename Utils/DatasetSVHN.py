import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from Utils.BaseDataset import BaseDataset


class datasetSVHN(BaseDataset):
    """
    SVHN dataset class.
    """
    def __init__(self, validation=False, download=True, seed=0, pin_memory=True, num_workers=0, cutout=16, train_size=80,batch_size_train=128,batch_size_test=128):
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalization,
        ])
        super().__init__(10, 3, 0.2, batch_size_train, batch_size_test, normalization, transform_train, transform_test,
                         './data/svhn', validation, download, seed, pin_memory, num_workers, cutout,train_size)

    def get_trainset(self):
        """
        Returns the SVHN training dataset.
        """
        return torchvision.datasets.SVHN(root=self.root, split="train", download=self.download, transform=self.transform_train)

    def get_validationset(self):
        """
        Returns the SVHN validation dataset.
        """
        return torchvision.datasets.SVHN(root=self.root, split="train", download=self.download, transform=self.transform_test)

    def get_testset(self):
        """
        Returns the SVHN test dataset.
        """
        return torchvision.datasets.SVHN(root=self.root, split="test", download=self.download, transform=self.transform_test)
