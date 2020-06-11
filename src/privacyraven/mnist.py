import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from torchvision.datasets import MNIST

from privacyraven.utils import convert_to_inference, set_hparams, train_and_test


def train_mnist_victim(
    transform=None,
    batch_size=100,
    num_workers=4,
    rand_split_val=None,
    gpus=1,
    max_epochs=8,
    learning_rate=1e-3,
):
    """Trains a 3-layer fully connected neural network on MNIST data

    Parameters:
        transform (Torchvision.transforms): transformation to be applied to MNIST data
        batch_size (int): size of batches to be trained and tested upon
        num_workers (int): number of workers assigned to computations
        rand_split_val (array): description of how val and train data are split
        gpus (int): num of gpus available to train upon
        max_epochs (int): the maximum # of epochs to run
        learning_rate (float): the learning rate for the optimizer

    Returns:
        Trained model ready for inference"""
    # Define hyperparameters implied by the use of MNIST

    input_size = 784
    targets = 10
    if transform is None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    if rand_split_val is None:
        rand_split_val = [55000, 5000]

    # Establish hyperparameters and DataLoaders
    hparams = set_hparams(
        transform,
        batch_size,
        num_workers,
        rand_split_val,
        gpus,
        max_epochs,
        learning_rate,
        input_size,
        targets,
    )

    train_dataloader, val_dataloader, test_dataloader = get_mnist_loaders(hparams)
    # Train, test, and convert the model to inference
    mnist_model = train_and_test(
        LightningClassifier, train_dataloader, val_dataloader, test_dataloader, hparams
    )
    mnist_model = convert_to_inference(mnist_model)
    return mnist_model


def get_mnist_loaders(hparams):
    """Downloads MNIST and creates PyTorch DataLoaders"""
    mnist_train = MNIST(
        os.getcwd(), train=True, download=True, transform=hparams["transform"]
    )
    mnist_test = MNIST(
        os.getcwd(), train=False, download=True, transform=hparams["transform"]
    )
    mnist_train, mnist_val = random_split(mnist_train, hparams["rand_split_val"])

    train_dataloader = DataLoader(
        mnist_train,
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
    )

    val_dataloader = DataLoader(
        mnist_val, batch_size=hparams["batch_size"], num_workers=hparams["num_workers"]
    )

    test_dataloader = DataLoader(
        mnist_test, batch_size=hparams["batch_size"], num_workers=hparams["num_workers"]
    )
    return train_dataloader, val_dataloader, test_dataloader


class LightningClassifier(pl.LightningModule):
    def __init__(self, hparams):
        """Defines a three layer fully connected neural network"""
        super(LightningClassifier, self).__init__()
        self.hparams = hparams
        self.layer_1 = torch.nn.Linear(self.hparams["input_size"], 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, self.hparams["targets"])

    def forward(self, x):
        """Establishes the neural network's forward pass

        Parameters:
            x (Torch tensor): MNIST input image

         Returns:
            output probability vector for MNIST classes"""
        batch_size, channels, width, height = x.size()

        # Input Layer: (batch_size, 1, 28, 28) -> (batch_size, 1*28*28)
        x = x.view(batch_size, -1)

        # Layer 1: (batch_size, 1*28*28) -> (batch_size, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # Layer 2: (batch_size, 128) -> (batch_size, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # Layer 3: (batch_size, 256) -> (batch_size, 10)
        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        """Calculates loss- the difference between model predictions and true labels

        Parameters:
            logits (Torch tensor): model output predictions
            labels (Torch tensor): true values for predictions

        Returns:
            Cross entropy loss"""
        return F.cross_entropy(logits, labels)

    def training_step(self, train_batch, batch_idx):
        """Pushes training data batch through model and calculates loss in loop

        Parameters:
            train_batch (Torch tensor): batch of training data from training dataloader
            batch_idx (int): index of batch in contention

        Returns:
            Formatted string with cross entropy loss and training logs"""
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, val_batch, batch_idx):
        """Pushes validation data batch through model and calculates loss in loop

        Parameters:
            val_batch (Tensor): batch of validation data from validation dataloader
            batch_idx (int): index of batch in contention

        Returns:
            Formatted string with resultant cross entropy loss"""
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        targets_hat = torch.argmax(logits, dim=1)
        n_correct_pred = torch.sum(y == targets_hat).item()
        return {"val_loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """Returns validation step results at the end of the epoch

        Parameters:
            outputs (array): result of validation step for each batch

        Returns:
            Formatted string with resultant metrics"""
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        """Sets up the optimization scheme"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def test_step(self, batch, batch_idx):
        """Pushes test data into the model and returns relevant metrics

        Parameters:
            batch (Torch tensor): batch of test data from test dataloader
            batch_idx (int): index of batch in contention

        Returns:
            Formatted string with relevant metrics"""
        x, y = batch
        y_hat = self(x)
        targets_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == targets_hat).item()
        return {
            "test_loss": F.cross_entropy(y_hat, y),
            "n_correct_pred": n_correct_pred,
            "n_pred": len(x),
        }

    def test_epoch_end(self, outputs):
        """Returns test step results at the end of the epoch

        Parameters:
            outputs (array): result of test step for each batch

        Returns:
            Formatted string with resultant metrics"""
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss}
        return {"avg_test_loss": avg_loss, "log": tensorboard_logs}
