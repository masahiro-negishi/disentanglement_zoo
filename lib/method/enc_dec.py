import torch
import torch.nn as nn


class Encoder(nn.Module):
    """convolutional encoder"""

    def __init__(self, in_channels: int, z_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # (B, in_channels, 64, 64) -> (B, 32, 32, 32
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),  # (B, 32, 32, 32) -> (B, 32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
            ),  # (B, 32, 16, 16) -> (B, 64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
            ),  # (B, 64, 8, 8) -> (B, 64, 4, 4)
            nn.ReLU(),
            nn.Flatten(),  # (B, 64, 4, 4) -> (B, 1024)
            nn.Linear(1024, 256),  # (B, 1024) -> (B, 256)
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, z_dim)  # (B, 256) -> (B, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)  # (B, 256) -> (B, z_dim)

    def forward(self, x):
        med = self.model(x)
        mean = self.fc_mean(med)
        logvar = self.fc_logvar(med)  # log(σ^2)
        eps = torch.randn_like(logvar)
        z = mean + eps * torch.exp(logvar / 2)  # reparameterization trick
        return z, mean, logvar


class Decoder(nn.Module):
    """convolutional decoder"""

    def __init__(self, out_channels: int, z_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),  # (B, z_dim) -> (B, 256)
            nn.ReLU(),
            nn.Linear(256, 1024),  # (B, 256) -> (B, 1024)
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),  # (B, 1024) -> (B, 64, 4, 4)
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
            ),  # (B, 64, 4, 4) -> (B, 64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
            ),  # (B, 64, 8, 8) -> (B, 32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),  # (B, 32, 16, 16) -> (B, 32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # (B, 32, 32, 32) -> (B, out_channels, 64, 64)
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.model(z)
