"""
These models will be depreciated soon. Use at your own risk.
"""
import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, models, transforms
from tqdm import tqdm


class ThreeLayerClassifier(pl.LightningModule):
    def __init__(self, hparams):
        """Defines a three layer fully connected neural network"""
        super(ThreeLayerClassifier, self).__init__()
        self.hparams = hparams
        self.layer_1 = torch.nn.Linear(self.hparams["input_size"], 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, self.hparams["targets"])

    def forward(self, x):
        """Establishes the neural network's forward pass

        Parameters:
            x: A Torch tensor of the input data

        Returns:
            output probability vector for classes"""
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
            logits: A Torch tensor of model output predictions
            labels: A Torch tensor of true values for predictions

        Returns:
            Cross entropy loss"""
        return F.cross_entropy(logits, labels)

    def training_step(self, train_batch, batch_idx):
        """Pushes training data batch through model and calculates loss in loop

        Parameters:
            train_batch: A Torch tensor of a batch of training data from training dataloader
            batch_idx: An integer of the index of batch in contention

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
            val_batch: A Torch tensor batch of validation data from validation dataloader
            batch_idx: An integer of the index of batch in contention

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
            outputs: An array with the result of validation step for each batch
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
            batch: A Torch tensor of a batch of test data
            batch_idx: An integer of the index of batch in contention
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
            outputs: An array with the result of test step for each batch
        Returns:
            Formatted string with resultant metrics"""
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss}
        return {"avg_test_loss": avg_loss, "log": tensorboard_logs}


class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self, hparams):
        """Create a classifier with a pretrained MobileNet backbone"""
        super(ImagenetTransferLearning, self).__init__()
        self.hparams = hparams
        self.feature_extractor = models.mobilenet_v2(pretrained=True)
        self.feature_extractor.eval()

        # Establish classifier
        # self.layer_1 = torch.nn.Linear(hparams["input_size"], 128)
        self.layer_1 = torch.nn.Linear(1000, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, hparams["targets"])

    def forward(self, x):
        """Establishes the neural network's forward pass

        Parameters:
            x: A Torch tensor of the input image

        Returns:
            Output probability vector for classes
        """
        x = self.feature_extractor(x)
        batch_size, hidden = x.size()

        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)

        x = torch.log_softmax(x, dim=1)
        return x

    def nll_loss(self, logits, labels):
        """Calculates loss

        Parameters:
            logits: A Torch tensor of the model output predictions
            labels: A Torch tensor of the true values for predictions

        Returns:
            Loss
        """
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        """Pushes training data batch through model and calculates loss in loop

        Parameters:
            train_batch: A Torch tensor with the batch of training data
            batch_idx: An integer of the index of batch in contention

        Returns:
            Formatted string with cross entropy loss and training logs
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = self.nll_loss(logits, y)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, val_batch, batch_idx):
        """Pushes validation data batch through model and calculates loss in loop

        Parameters:
            val_batch: A Torch tensor of a batch of validation data
            batch_idx: An integer of the index of batch in contention

        Returns:
            Formatted string with resultant cross entropy loss
        """
        x, y = val_batch
        logits = self.forward(x)
        loss = self.nll_loss(logits, y)
        targets_hat = torch.argmax(logits, dim=1)
        n_correct_pred = torch.sum(y == targets_hat).item()
        return {"val_loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """Returns validation step results at the end of the epoch

        Parameters:
            outputs: An array of the result of validation step for each batch

        Returns:
            Formatted string with resultant metrics
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        """Sets up the optimization scheme"""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams["learning_rate"]
        )
        return optimizer

    def test_step(self, batch, batch_idx):
        """Pushes test data into the model and returns relevant metrics

        Parameters:
            batch: A Torch tensor of a batch of test data from test dataloader
            batch_idx: An integer of the index of batch in contention

        Returns:
            Formatted string with relevant metrics"""
        x, y = batch
        y_hat = self(x)
        targets_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == targets_hat).item()
        return {
            "test_loss": F.nll_loss(y_hat, y),
            "n_correct_pred": n_correct_pred,
            "n_pred": len(x),
        }

    def test_epoch_end(self, outputs):
        """Returns test step results at the end of the epoch

        Parameters:
            outputs: An array with the results of test step for each batch

        Returns:
            Formatted string with resultant metrics"""
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss}
        return {"avg_test_loss": avg_loss, "log": tensorboard_logs}
