
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init


from Evolution.CgpTonn_nW import CgpNetW
from Utils.DatasetFactory import DatasetFactory


def weights_init_kaiming(m: nn.Module):
    """Initialize weights using Kaiming Normal initialization."""
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm2d") != -1:
        init.uniform_(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net: nn.Module, init_type: str = "kaiming"):
    """Apply weight initialization to the network."""
    if init_type == "kaiming":
        net.apply(weights_init_kaiming)
    else:
        raise ValueError(f"Unsupported initialization method: {init_type}")


class CNNTrainW:
    """
    CNN Training Wrapper.
    Manages dataset loading, model training, and validation.
    """

    def __init__(self, model, dataset_name: str, validation: bool = True, verbose: bool = True,
                 img_size: int = 32, batch_size: int = 128, aux_head: bool = False, size: int = 80):
        """
        Initialize training parameters and dataset loaders.

        Args:
            model: CGP-defined model structure.
            dataset_name (str): Name of the dataset.
            validation (bool): Whether to use validation mode.
            verbose (bool): Flag to enable verbose logging.
            img_size (int): Input image size.
            batch_size (int): Batch size for training and testing.
            aux_head (bool): Use auxiliary head in the model.
            size (int): Size parameter for dataset loading.
        """
        self.model = model
        self.dataset_name = dataset_name
        self.validation = validation
        self.verbose = verbose
        self.img_size = img_size
        self.batch_size = batch_size
        self.aux_head = aux_head
        self.size = size

        self.dataset = DatasetFactory.getDataset(
            self.dataset_name,
            validation=self.validation,
            download=True,
            pin_memory=True,
            num_workers=0,
            cutout=16,
            train_size=self.size,
            batch_size_train=self.batch_size,
            batch_size_test=self.batch_size,
        )
        self.train_loader = self.dataset.trainloader
        self.test_loader = self.dataset.validationloader if validation else self.dataset.testloader
        self.n_class = self.dataset.n_class
        self.channel = self.dataset.channel

    def __call__(self, gpu_id: int, epoch_num: int = 10, mode: bool = True) -> tuple:
        """
        Train the model and return validation accuracy and the trained model.

        Args:
            gpu_id (int): ID of the GPU to use (-1 for CPU).
            epoch_num (int): Number of epochs to train.
            mode (bool): Training mode (evolutionary or complete).

        Returns:
            tuple: (Validation accuracy, Trained model)
        """
        device = torch.device("mps" if torch.backends.mps.is_available() else (f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"))
        if torch.cuda.is_available() and gpu_id >= 0:
            torch.cuda.set_device(gpu_id)

        try:

            model_instance = CgpNetW(self.model, self.channel, self.n_class, self.img_size, self.aux_head)
            init_weights(model_instance, "kaiming")
            model_instance = model_instance.to(device)
        except Exception as e:
            raise RuntimeError(f"Error initializing model: {e}")

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model_instance.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_num)

        for epoch in range(1, epoch_num + 1):
            self._train_epoch(model_instance, optimizer, criterion, device)
            scheduler.step()

            if self.verbose:
                print(f"Epoch {epoch}/{epoch_num} completed.")

        valid_acc = self._validate(model_instance, criterion, device) if self.validation else 0.0
        return valid_acc, model_instance

    def _train_epoch(self, model: nn.Module, optimizer, criterion, device: torch.device):
        """Train the model for one epoch."""
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, aux_outputs = model(inputs)
            loss = criterion(outputs, labels)

            if self.aux_head:
                loss += 0.4 * criterion(aux_outputs, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            total_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        accuracy = correct / total
        if self.verbose:
            print(f"Training Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    def _validate(self, model: nn.Module, criterion, device: torch.device) -> float:
        """Validate the model on the test set."""
        model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                total_loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

        accuracy = correct / total
        if self.verbose:
            print(f"Validation Accuracy: {accuracy:.4f}")
        return accuracy


