import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as nnf
from tqdm import tqdm

class InversionModel(pl.LightningModule):
	def __init__(self, hparams, nz, ngf, classifier):
		super().__init__()

		self.save_hyperparameters()
		self.nz = nz
		self.ngf = ngf
		self.mse_loss = 0
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(
				nz,
				stride=(1, 1), 
				padding=(1, 1),
				kernel_size=(4, 4)
			)
			nn.BatchNorm2d(ngf)
			nn.Tanh()

			nn.ConvTranspose2d(
				nz,
				stride=(1, 1), 
				padding=(1, 1),
				kernel_size=(4, 4)
			)

			nn.BatchNorm2d(ngf)
			nn.Tanh()
			
			nn.ConvTranspose2d(
				nz,
				stride=(2, 2), 
				padding=(1, 1),
				kernel_size=(4, 4)
			)
			nn.BatchNorm2d(ngf)

			nn.Sigmoid()

		)

	def training_step(self, batch, batch_idx):

		self.classifier.eval()
	    self.train()

	    for k, (data, target) in tqdm(range(batch)):

	        # (image tensor, label tensor)

	        # Fwx is the prediction vector outputted by our model Fw
	        # reconstructed is the reconstructed image outputted by the inversion model
	        # data is the ground truth image

	        Fwx = forward_model(data)
	        recontructed = inversion(Fwx)
	        loss = nnf.mse_loss(recontructed, data)

	def test_step(self, batch, batch_idx):
	    for k, (data, target) in tqdm(range(batch)):

	        # (image tensor, label tensor)

	        # Fwx is the prediction vector outputted by our model Fw
	        # reconstructed is the reconstructed image outputted by the inversion model
	        # data is the ground truth image

	        Fwx = forward_model(data)
	        recontructed = inversion(Fwx)
	        loss = nnf.mse_loss(recontructed, data)
	        
	def forward(self, x):
		x = add(vlog(nnf.softmax(x, dim=1)), c)
		x = torch.zeros(len(x), device=device).scatter_(0, sort(x.topk(t).indices).values, Fwx)
		x = x.view(-1, self.nz, 1, 1)
		x = self.decoder(x).view(-1, 1, 28, 28)

		return x

    def configure_optimizers(self):
        """Executes optimization for training and validation"""
        return torch.optim.Adam(self.parameters(), self.hparams["learning_rate"])




