import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Stack(nn.Module):
    def __init__(self, channels, width):
        super(Stack, self).__init__()
        self.channels = channels
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.width, self.channels)


class VAE(pl.LightningModule):
    def __init__(self, seq_len, n_features=1, enc_out_dim=512, latent_dim=256, lr=1e-4, alpha=1024):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim

        self.save_hyperparameters()
        self.lr = lr
        self.alpha = alpha

        # encoder, decoder
        self.encoder = nn.Sequential(
            nn.Linear(seq_len * n_features, 392), nn.BatchNorm1d(392), nn.LeakyReLU(0.1),
            nn.Linear(392, 196), nn.BatchNorm1d(196), nn.LeakyReLU(0.1),
            nn.Linear(196, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, enc_out_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, 196), nn.BatchNorm1d(196), nn.LeakyReLU(0.1),
            nn.Linear(196, 392), nn.BatchNorm1d(392), nn.LeakyReLU(0.1),
            nn.Linear(392, seq_len)
        )
        self.tanh = nn.Tanh()

        # distribution parameters
        self.hidden2mu = nn.Linear(enc_out_dim, latent_dim)
        self.hidden2log_var = nn.Linear(enc_out_dim, latent_dim)

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var

    def decode(self, x):
        x = self.decoder(x)
        return x

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn_like(sigma)
        return mu + sigma * z

    def training_step(self, batch, batch_idx):
        x = batch
        mu, log_var, x_out = self.forward(x)
        kl_loss = (-0.5 * (1 + log_var - mu ** 2 -
                           torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        # print(kl_loss.item(),recon_loss.item())
        loss = recon_loss*self.alpha + kl_loss

        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        mu, log_var, x_out = self.forward(x)

        kl_loss = (-0.5 * (1 + log_var - mu ** 2 -
                           torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        # print(kl_loss.item(),recon_loss.item())
        loss = recon_loss * self.alpha + kl_loss
        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        # print(x.mean(),x_out.mean())
        return x_out, loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, )
        return {
            "optimizer": optimizer, "lr_scheduler": lr_scheduler,
            "monitor": "val_loss"
        }

    def forward(self, x):
        enc_in = x.view(x.size(0), -1)
        mu, log_var = self.encode(enc_in)
        hidden = self.reparametrize(mu, log_var)
        dec_out = self.decoder(hidden)
        x_hat = self.tanh(dec_out.view(dec_out.size(0), self.seq_len, self.n_features))
        return mu, log_var, x_hat
