
import torchvision.transforms as transforms

from Utils.BaseDataset import BaseDataset
from Utils.RotatedCIFAR import CIFAR_Rotation_Dataset


class datasetCIFAR100Pretext(BaseDataset):
    """
    CIFAR-100 Pretext dataset class.
    """
    def __init__(self, validation=False, download=True, seed=0, pin_memory=True, num_workers=0, cutout=16, train_size=80,batch_size_train=128,batch_size_test=128):
        normalization = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalization,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalization,
        ])
        super().__init__(4, 3, 0.2, batch_size_train, batch_size_test, normalization, transform_train, transform_test,
                         './data/cifar-100', validation, download, seed, pin_memory, num_workers, cutout, train_size)

    def get_trainset(self):
        """
        Returns the CIFAR-100 Pretext training dataset.
        """
        return CIFAR_Rotation_Dataset(root=self.root, train=True, download=self.download, transform=self.transform_train)

    def get_validationset(self):
        """
        Returns the CIFAR-100 Pretext validation dataset.
        """
        return CIFAR_Rotation_Dataset(root=self.root, train=True, download=self.download, transform=self.transform_test)
    def get_testset(self):
        """
        Returns the CIFAR-100 Pretext test dataset.
        """
        return CIFAR_Rotation_Dataset(root=self.root, train=False, download=self.download, transform=self.transform_test)