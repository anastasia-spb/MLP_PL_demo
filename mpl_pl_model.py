import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from mpl import MPLNet


class MPLNetWrapper(pl.LightningModule):
    def __init__(self, in_dim: int, out_dim: int, lr: float, weight_decay: float, batch_size: int):
        super(MPLNetWrapper, self).__init__()
        self.save_hyperparameters()

        self.model = MPLNet(input_dim=self.hparams.in_dim, output_dim=self.hparams.out_dim)
        self.loss = nn.CrossEntropyLoss()

    def _loss(self, logits, gt):
        return self.loss(logits, gt)


    def forward(self, batch):
        output = self.model(batch)
        return output
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x_batch, _ = batch
        output = self(x_batch)
        probabilities = F.softmax(output, dim=-1)
        return probabilities

    def training_step(self, batch, batch_idx):
        x_batch, target = batch
        output = self(x_batch)

        loss_val = self._loss(output, target)
        self.log("train_loss", loss_val, sync_dist=True, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size)

        return loss_val

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_pred = self(x_batch)

        loss_val = self._loss(y_pred, y_batch)
        self.log("val_loss", loss_val, sync_dist=True, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size)

        probabilities = F.softmax(y_pred, dim=-1)

        return {'val_loss': loss_val, 'probabilities': probabilities}

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        output = self(x_batch)
        loss_val = self._loss(output, y_batch)
        probabilities = F.softmax(output, dim=-1)

        return {'val_loss': loss_val, 'probabilities': probabilities}

    def test_epoch_end(self, outputs):
        # Log results at the end of epoch
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode='min',
                                                                  factor=0.2,
                                                                  patience=10,
                                                                  min_lr=1e-6,
                                                                  verbose=False)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }

