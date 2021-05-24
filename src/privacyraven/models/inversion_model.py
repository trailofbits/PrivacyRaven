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
            nn.ConvTranspose2d(
                10,
                512,
                stride=(1, 1), 
                padding=(1, 1),
                kernel_size=(4, 4),
            ),
            nn.BatchNorm2d(self.ngf),
            nn.Tanh(),

            nn.ConvTranspose2d(
                512,
                256,
                stride=(1, 1), 
                padding=(1, 1),
                kernel_size=(4, 4),
            ),

            nn.BatchNorm2d(self.ngf),
            nn.Tanh(),
            
            nn.ConvTranspose2d(
                256,
                128,
                stride=(2, 2), 
                padding=(1, 1),
                kernel_size=(4, 4),
            ),
            nn.Tanh(),
            nn.ConvTranspose2d(
                128,
                1,
                stride=(2, 2), 
                padding=(1, 1),
                kernel_size=(4, 4),
            ),

            nn.Sigmoid()

        )

    def training_step(self, batch, batch_idx):
        #print(len(batch))
        data, _ = batch
        Fwx = self.classifier(data)
        print("Fwx vector: ", Fwx, len(Fwx))
        recontructed = self(Fwx)
        loss = nnf.mse_loss(recontructed, data)

        return loss

    def test_step(self, batch, batch_idx):
        data, _ = batch
        Fwx = self.classifier(data)
        recontructed = self(Fwx)
        loss = nnf.mse_loss(recontructed, data)
        
    def forward(self, Fwx):
        Fwx = add(vlog(nnf.softmax(Fwx, dim=1)), self.c)
        #print("Fwx: ", Fwx)
        Fwx = torch.zeros(len(Fwx)).scatter_(0, sort(Fwx.topk(self.t).indices).values, Fwx)
        Fwx = Fwx.view(-1, self.nz, 1, 1)
        Fwx = self.decoder(Fwx).view(-1, 1, 28, 28)

        return Fwx

    def configure_optimizers(self):
        """Executes optimization for training and validation"""
        return torch.optim.Adam(self.parameters(), 1e-3)




