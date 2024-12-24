import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from Utils.BaseDataset import BaseDataset


class datasetCIFAR100(BaseDataset):
    """
    CIFAR-100 dataset class.
    """

    def __init__(self, validation=False, download=True, seed=0, pin_memory=True, num_workers=0, cutout=16, train_size=80,batch_size_train=128,batch_size_test=128):
        normalization = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
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
        super().__init__(100, 3, 0.2, batch_size_train, batch_size_test, normalization, transform_train, transform_test, './data/cifar-100',
                         validation, download, seed, pin_memory, num_workers, cutout, train_size)


    def get_trainset(self):
        """
        Returns the CIFAR-100 training dataset.
        """
        return torchvision.datasets.CIFAR100(root=self.root, train=True, download=self.download,
                                             transform=self.transform_train)

    def get_validationset(self):
        """
        Returns the CIFAR-100 validation dataset.
        """
        return torchvision.datasets.CIFAR100(root=self.root, train=True, download=self.download,
                                             transform=self.transform_test)

    def get_testset(self):
        """
        Returns the CIFAR-100 test dataset.
        """
        return torchvision.datasets.CIFAR100(root=self.root, train=False, download=self.download,
                                             transform=self.transform_test)

    def do_validation(self):
        """
        Prepares the training and validation datasets and their respective DataLoaders.
        """
        trainset = self.get_trainset()
        validationSet = self.get_validationset()
        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))
        train_size = (100 - self.train_size) / 100

        split_train = int(np.floor(train_size * num_train))
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split_train:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.trainloader = DataLoader(trainset, batch_size=self.batch_size_train, sampler=train_sampler,
                                      num_workers=self.num_workers, pin_memory=self.pin_memory)

        self.validationloader = DataLoader(validationSet, batch_size=self.batch_size_test, sampler=valid_sampler,
                                           num_workers=self.num_workers, pin_memory=self.pin_memory)

