import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from Utils.Cutout import Cutout


class BaseDataset:
    """
    Base class for datasets.

    Attributes:
        n_class (int): Number of classes in the dataset.
        channel (int): Number of channels in the images.
        valid_size (float): Proportion of the training set to use for validation.
        batch_size_train (int): Batch size for training.
        batch_size_test (int): Batch size for testing.
        normalization (transforms.Normalize): Normalization transform.
        transform_train (transforms.Compose): Transformations for training data.
        transform_test (transforms.Compose): Transformations for test data.
        root (str): Root directory for the dataset.
        download (bool): Whether to download the dataset.
        seed (int): Random seed for reproducibility.
        pin_memory (bool): Whether to use pinned memory.
        num_workers (int): Number of worker threads for data loading.
        cutout (int): Size of the cutout augmentation.
        trainloader (DataLoader): DataLoader for training data.
        testloader (DataLoader): DataLoader for test data.
        validationloader (DataLoader): DataLoader for validation data.
        train_size (int): Number of training dataset size % (default: 80).
    """
    def __init__(self, n_class, channel, valid_size, batch_size_train, batch_size_test, normalization, transform_train,
                 transform_test, root, validation=False, download=True, seed=0, pin_memory=True, num_workers=0, cutout=16, train_size=80):

        self.n_class = n_class
        self.channel = channel
        self.train_size=train_size
        self.valid_size = valid_size
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.normalization = normalization
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.root = root
        self.download = download
        self.seed = seed
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.cutout = cutout
        self.trainloader = None
        self.testloader = None
        self.validationloader = None
        if validation:
            self.do_validation()
        else:
            self.do()

    def do(self):
        """
        Prepares the training and test datasets and their respective DataLoaders.
        """
        if self.cutout > 1:
            self.transform_train.transforms.append(Cutout(self.cutout))
        trainset = self.get_trainset()
        testset = self.get_testset()
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size_train, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.testloader = DataLoader(testset, batch_size=self.batch_size_test, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def do_validation(self):
        """
        Prepares the training and validation datasets and their respective DataLoaders.
        """
        trainset = self.get_trainset()
        validationSet = self.get_validationset()
        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size_train, sampler=train_sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.validationloader = DataLoader(validationSet, batch_size=self.batch_size_test, sampler=valid_sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def get_trainset(self):
        """
        Returns the training dataset. Must be implemented by subclasses.

        Args:
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        """
        raise NotImplementedError

    def get_validationset(self):
        """
        Returns the validation dataset. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_testset(self):
        """
        Returns the test dataset. Must be implemented by subclasses.
        """
        raise NotImplementedError

