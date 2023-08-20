import torch
import torch.nn as nn

from .enc_dec import Encoder, Decoder


class VAE(nn.Module):
    """Variational Autoencoder"""

    def __init__(self, channels: int, z_dim: int):
        super().__init__()
        self.encoder = Encoder(channels, z_dim)
        self.decoder = Decoder(channels, z_dim)
        self.z_dim = z_dim

    def forward(self, x: torch.Tensor):
        z, mean, logvar = self.encoder(x)
        lamb = self.decoder(z)  # x_hat can be sampled from Bernoulli(lamb)
        return lamb, mean, logvar

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def loss(
        self,
        x: torch.Tensor,
        lamb: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ):
        """calculate loss

        Args:
            x (torch.Tensor): (B, channels, 64, 64)
            lamb (torch.Tensor): (B, channels, 64, 64)
            mean (torch.Tensor): (B, z_dim)
            logvar (torch.Tensor): (B, z_dim)

        Returns:
            torch.Tensor: total loss
            torch.Tensor: reconstruction loss
            torch.Tensor: kl loss
        """
        # reconstruction loss (ref: disentanglement_lib by Google)
        clipped_x = torch.clamp(x, 1e-6, 1 - 1e-6)
        recon_loss = (
            nn.functional.binary_cross_entropy(lamb, x, reduction="sum")
            - nn.functional.binary_cross_entropy(clipped_x, clipped_x, reduction="sum")
        ) / x.shape[0]
        # recon_loss = ((x - lamb) ** 2).sum() / x.shape[0]

        # KL divergence loss
        kl_loss = (
            0.5 * torch.sum(-1 - logvar + mean**2 + torch.exp(logvar)) / x.shape[0]
        )
        return recon_loss + kl_loss, recon_loss, kl_loss  # loss = - ELBO


class BetaVAE(nn.Module):
    """Beta VAE"""

    def __init__(self, channels: int, z_dim: int, beta: float):
        super().__init__()
        self.encoder = Encoder(channels, z_dim)
        self.decoder = Decoder(channels, z_dim)
        self.z_dim = z_dim
        self.beta = beta

    def forward(self, x: torch.Tensor):
        z, mean, logvar = self.encoder(x)
        lamb = self.decoder(z)  # x_hat can be sampled from Bernoulli(lamb)
        return lamb, mean, logvar

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def loss(
        self,
        x: torch.Tensor,
        lamb: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ):
        """calculate loss

        Args:
            x (torch.Tensor): (B, channels, 64, 64)
            lamb (torch.Tensor): (B, channels, 64, 64)
            mean (torch.Tensor): (B, z_dim)
            logvar (torch.Tensor): (B, z_dim)

        Returns:
            torch.Tensor: total loss
            torch.Tensor: reconstruction loss
            torch.Tensor: kl loss
        """
        # reconstruction loss (ref: disentanglement_lib by Google)
        clipped_x = torch.clamp(x, 1e-6, 1 - 1e-6)
        recon_loss = (
            nn.functional.binary_cross_entropy(lamb, x, reduction="sum")
            - nn.functional.binary_cross_entropy(clipped_x, clipped_x, reduction="sum")
        ) / x.shape[0]
        # recon_loss = ((x - lamb) ** 2).sum() / x.shape[0]

        # KL divergence loss
        kl_loss = (
            0.5 * torch.sum(-1 - logvar + mean**2 + torch.exp(logvar)) / x.shape[0]
        )
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss  # loss = - ELBO


class AnnealedVAE(nn.Module):
    """Annealed VAE"""

    def __init__(
        self,
        channels: int,
        z_dim: int,
        c_start: float,
        c_end: float,
        gamma: float,
        epochs: int,
    ):
        super().__init__()
        self.encoder = Encoder(channels, z_dim)
        self.decoder = Decoder(channels, z_dim)
        self.z_dim = z_dim
        self.c_start = c_start
        self.c_end = c_end
        self.gamma = gamma
        self.epochs = epochs
        self.c = c_start

    def forward(self, x: torch.Tensor):
        z, mean, logvar = self.encoder(x)
        lamb = self.decoder(z)  # x_hat can be sampled from Bernoulli(lamb)
        return lamb, mean, logvar

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def next_epoch(self):
        # update self.c
        self.c += (self.c_end - self.c_start) / (self.epochs - 1)

    def loss(
        self,
        x: torch.Tensor,
        lamb: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ):
        """calculate loss

        Args:
            x (torch.Tensor): (B, channels, 64, 64)
            lamb (torch.Tensor): (B, channels, 64, 64)
            mean (torch.Tensor): (B, z_dim)
            logvar (torch.Tensor): (B, z_dim)

        Returns:
            torch.Tensor: total loss
            torch.Tensor: reconstruction loss
            torch.Tensor: kl loss
        """
        # reconstruction loss (ref: disentanglement_lib by Google)
        clipped_x = torch.clamp(x, 1e-6, 1 - 1e-6)
        recon_loss = (
            nn.functional.binary_cross_entropy(lamb, x, reduction="sum")
            - nn.functional.binary_cross_entropy(clipped_x, clipped_x, reduction="sum")
        ) / x.shape[0]
        # recon_loss = ((x - lamb) ** 2).sum() / x.shape[0]

        # KL divergence loss
        kl_loss = (
            0.5 * torch.sum(-1 - logvar + mean**2 + torch.exp(logvar)) / x.shape[0]
        )
        kl_loss_term = (
            torch.sum(
                torch.abs(
                    0.5 * torch.sum(-1 - logvar + mean**2 + torch.exp(logvar), dim=1)
                    - self.c
                )
            )
            / x.shape[0]
        )
        return (
            recon_loss + self.gamma * kl_loss_term,
            recon_loss,
            kl_loss,
        )  # loss = - ELBO
