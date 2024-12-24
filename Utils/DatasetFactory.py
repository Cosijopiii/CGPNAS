from Utils.DatasetCIFAR10 import datasetCIFAR10
from Utils.DatasetCIFAR100 import datasetCIFAR100
from Utils.DatasetCIFAR100Pretext import datasetCIFAR100Pretext
from Utils.DatasetCINIC10 import datasetCINIC10
from Utils.DatasetSVHN import datasetSVHN

class DatasetFactory:
    """
    Factory class for creating dataset instances based on the dataset name.

    Attributes:
        dataset_names (list): List of supported dataset names.
        root (str): Root directory for datasets.
    """
    dataset_names = ["CIFAR-10", "CIFAR-100", "SVHN", "CIFAR-100P", "CINIC"]
    root = './data'

    @staticmethod
    def getDataset(name, validation=False, download=True, seed=0, pin_memory=True, num_workers=0, cutout=16, train_size=80,batch_size_train=128,batch_size_test=128):
        """
        Returns the specified dataset.

        Args:
            name (str): Name of the dataset.
            validation (bool): Whether to include a validation set.
            download (bool): Whether to download the dataset.
            seed (int): Random seed for reproducibility.
            pin_memory (bool): Whether to use pinned memory.
            num_workers (int): Number of worker threads for data loading.
            cutout (int): Size of the cutout augmentation.
            train_size (int): Size parameter for certain datasets.
            batch_size_train (int): Batch size for training.
            batch_size_test (int): Batch size for testing.
        Returns:
            dataset: The requested dataset instance.
        """
        if name == DatasetFactory.dataset_names[0]:
            return datasetCIFAR10(validation, download, seed, pin_memory, num_workers, cutout, train_size, batch_size_train,batch_size_test)
        elif name == DatasetFactory.dataset_names[1]:
            return datasetCIFAR100(validation, download, seed, pin_memory, num_workers, cutout, train_size, batch_size_train,batch_size_test)
        elif name == DatasetFactory.dataset_names[2]:
            return datasetSVHN(validation, download, seed, pin_memory, num_workers, cutout, train_size, batch_size_train,batch_size_test)
        elif name == DatasetFactory.dataset_names[3]:
            return datasetCIFAR100Pretext(validation, download, seed, pin_memory, num_workers, cutout, train_size, batch_size_train,batch_size_test)
        elif name == DatasetFactory.dataset_names[4]:
            return datasetCINIC10(validation, download, seed, pin_memory, num_workers, cutout, train_size, batch_size_train,batch_size_test)
        else:
            raise ValueError(f"Dataset {name} is not recognized.")