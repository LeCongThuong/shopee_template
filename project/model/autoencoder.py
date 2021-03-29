import hydra
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim=64, optim_encoder=None, optim_decoder=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        encoder_optim = hydra.utils.instantiate(self.hparams.optim_encoder, params=self.encoder.parameters())
        decoder_optim = hydra.utils.instantiate(self.hparams.optim_decoder, params=self.decoder.parameters())
        return [encoder_optim, decoder_optim], []

    def on_train_start(self):
        # Proper logging of hyperparams and metrics in TB
        self.logger.log_hyperparams(self.hparams, {"loss/val": 0, "accuracy/val": 0, "accuracy/test": 0})

