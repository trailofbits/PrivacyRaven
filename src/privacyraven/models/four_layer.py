import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics

class FourLayerClassifier(pl.LightningModule):
    """This module describes a neural network with four fully connected layers
    containing 420 hidden units and a dropout layer (of 0.4). Cross entropy
    loss is utilized, the Adam optimizer is used, and accuracy is reported."""

    def __init__(self, hparams):
        """Defines overall computations"""
        super().__init__()

        self.hparams = hparams
        self.save_hyperparameters()

        # A dictionary informs of the model of the input size, number of
        # target classes, and learning rate.
        self.fc1 = nn.Linear(self.hparams["input_size"], 420)
        self.fc2 = nn.Linear(420, 420)
        self.fc3 = nn.Linear(420, 420)
        self.fc4 = nn.Linear(420, self.hparams["targets"])
        self.dropout = nn.Dropout(0.4)

        # Instantiate accuracy metrics for each phase
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        # self.train_acc = pl.metrics.Accuracy()
        # self.valid_acc = pl.metrics.Accuracy()
        # self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        """Executes the forward pass and inference phase"""
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

    def training_step(self, batch, batch_idx):
        """Runs the training loop"""
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        self.train_acc(torch.nn.functional.softmax(y_hat, dim=1), y)
        self.log("train_accuracy", self.train_acc) #, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Runs the validation loop"""
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("valid_loss", loss) 
        self.valid_acc(torch.nn.functional.softmax(y_hat, dim=1), y)
        # self.valid_acc(y_hat, y)
        self.log("valid_accuracy", self.valid_acc) #, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Tests the network"""
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        self.test_acc(torch.nn.functional.softmax(y_hat, dim=1), y)
        # self.test_acc(y_hat, y)
        self.log("test_accuracy", self.test_acc) #, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        """Executes optimization for training and validation"""
        return torch.optim.Adam(self.parameters(), self.hparams["learning_rate"])


