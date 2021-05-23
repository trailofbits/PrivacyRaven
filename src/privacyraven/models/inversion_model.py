import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as nnf
from tqdm import tqdm

class InversionModel(pl.LightningModule):
    def __init__(self, hparams, inversion_params, classifier):
        super().__init__()

        self.classifier = classifier
        self.hparams = hparams
        #self.save_hyperparameters()
        self.nz = inversion_params["nz"]
        self.ngf = inversion_params["ngf"]
        self.mse_loss = 0

        self.classifier.eval()
        self.train()
        
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


        for (data, target) in enumerate(tqdm(batch)):

            # (image tensor, label tensor)

            # Fwx is the prediction vector outputted by our model Fw
            # reconstructed is the reconstructed image outputted by the inversion model
            # data is the ground truth image

            Fwx = self.classifier(data)
            recontructed = self(Fwx)
            loss = nnf.mse_loss(recontructed, data)
            
        return loss

    def test_step(self, batch, batch_idx):
        for (data, target) in enumerate(tqdm(batch)):

            # (image tensor, label tensor)

            # Fwx is the prediction vector outputted by our model Fw
            # reconstructed is the reconstructed image outputted by the inversion model
            # data is the ground truth image

            Fwx = self.classifier(data)
            recontructed = self(Fwx)
            loss = nnf.mse_loss(recontructed, data)
            
    def forward(self, x):
        x = add(vlog(nnf.softmax(x, dim=1)), c)
        x = torch.zeros(len(x), device=device).scatter_(0, sort(x.topk(t).indices).values, Fwx)
        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x).view(-1, 1, 28, 28)

        return x

    def configure_optimizers(self):
        """Executes optimization for training and validation"""
        return torch.optim.Adam(self.parameters(), 1e-3)




