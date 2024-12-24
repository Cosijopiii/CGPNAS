import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CIFAR_Rotation_Dataset(Dataset):
    """
    CIFAR-100 Rotation Dataset.

    This dataset applies rotations to the CIFAR-100 images and assigns rotation labels.
    The rotations are 0, 90, 180, and 270 degrees.

    Attributes:
        cifar100 (torchvision.datasets.CIFAR100): The CIFAR-100 dataset.
        rotation_labels (torch.Tensor): Tensor containing rotation labels [0, 1, 2, 3].
        indices_per_rotation (list): List of lists containing indices for each rotation level.
        indices (list): List of indices for accessing the dataset in a rotated manner.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """
        Initializes the CIFAR_Rotation_Dataset.

        Args:
            root (str): Root directory of the dataset.
            train (bool): If True, creates dataset from training set, otherwise from test set.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            download (bool): If True, downloads the dataset from the internet and puts it in root directory.
        """
        self.cifar100 = torchvision.datasets.CIFAR100(root=root, train=train, transform=transform, download=download)
        self.rotation_labels = torch.Tensor([0, 1, 2, 3])  # Map to rotation labels

        # Divide training images into 4 rotation levels
        self.indices_per_rotation = [[], [], [], []]  # Rotation 0, 90, 180, 270 respectively
        for i, (img, target) in enumerate(self.cifar100):
            rotation_label = target % 4  # Compute rotation label
            self.indices_per_rotation[rotation_label].append(i)  # Assign index to corresponding rotation level

        # Concatenate indices in sequential order
        self.indices = []
        for i in range(len(self.cifar100)):
            rotation_idx = i % 4
            sample_idx = i // 4
            real_index = self.indices_per_rotation[rotation_idx][sample_idx]
            self.indices.append(real_index)

    def __getitem__(self, index):
        """
        Returns the image and the corresponding rotation label.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            tuple: (image, rotation_label) where rotation_label is the label indicating the rotation applied to the image.
        """
        # Get the image and the corresponding target
        img, target = self.cifar100[self.indices[index]]

        # Compute the rotation label based on the target
        rotation_label = target % 4

        # Apply the rotation transformation to the image based on the rotation label
        if rotation_label == 1:
            img = transforms.functional.rotate(img, 90)
        elif rotation_label == 2:
            img = transforms.functional.rotate(img, 180)
        elif rotation_label == 3:
            img = transforms.functional.rotate(img, 270)

        # Return the image with the new target (rotation label)
        return img, rotation_label

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.indices)