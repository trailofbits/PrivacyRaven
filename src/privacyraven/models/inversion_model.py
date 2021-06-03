import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as nnf
from torch import topk, add, log as vlog, tensor, sort
from tqdm import tqdm
from torch.cuda import device_count

class InversionModel(pl.LightningModule):
    def __init__(self, hparams, inversion_params, classifier):
        super().__init__()

        self.classifier = classifier
        self.hparams = hparams
        #self.save_hyperparameters()
        self.nz = inversion_params["nz"]
        self.ngf = inversion_params["ngf"]
        self.c = inversion_params["affine_shift"]
        self.t = inversion_params["truncate"]
        self.mse_loss = 0
        self.classifier.eval()
        self.train()
        #self.device = "cuda:0" if device_count() else None

        self.decoder = nn.Sequential(
            # input is Z
            nn.ConvTranspose2d(10, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def training_step(self, batch, batch_idx):
        #print(len(batch))
        images, _ = batch

        for data in images:
            augmented = torch.empty(1, 1, 28, 28)
            augmented[0] = data
            #print("Augmented size:", augmented.size())
            Fwx = self.classifier(augmented)
            #print("Fwx vector: ", Fwx)
            reconstructed = self(Fwx[0])
            augmented = nnf.pad(input=augmented, pad=(2, 2, 2, 2))
            loss = nnf.mse_loss(reconstructed, augmented)
            self.log("train_loss: ", loss)

        return loss

    def test_step(self, batch, batch_idx):
        images, _ = batch

        for data in images:
            augmented = torch.empty(1, 1, 28, 28)
            augmented[0] = data
            Fwx = self.classifier(augmented)
            #print("Fwx vector: ", Fwx, len(Fwx))
            reconstructed = self(Fwx[0])
            augmented = nnf.pad(input=augmented, pad=(2, 2, 2, 2))
            #print(reconstructed.size(), augmented.size())
            loss = nnf.mse_loss(reconstructed, augmented)
            self.log("test_loss: ", loss)

        return loss
        
    def forward(self, Fwx):
        Fwx = add(Fwx, self.c)
        #print("Fwx: ", Fwx)
        Fwx = torch.zeros(len(Fwx)).scatter_(0, sort(Fwx.topk(self.t).indices).values, Fwx)
        Fwx = torch.reshape(Fwx, (10, 1))
        #print("Forward Fwx:", Fwx, Fwx.size())
        Fwx = Fwx.view(-1, self.nz, 1, 1)
        Fwx = self.decoder(Fwx)
        #print("Old size: ", Fwx.size())
        #Fwx = nnf.pad(input=Fwx, pad=(2, 2, 2, 2))
        #print("New size: ", Fwx.size())
        Fwx = Fwx.view(-1, 1, 32, 32)

        return Fwx
        

    def configure_optimizers(self):
        """Executes optimization for training and validation"""
        return torch.optim.Adam(self.parameters(), 1e-3)




