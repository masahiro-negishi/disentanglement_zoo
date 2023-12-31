import pytest
import torch

from .vae import VAE, BetaVAE, AnnealedVAE


@pytest.mark.parametrize(
    ("channels", "z_dim", "batch_size", "device"),
    [(3, 10, 10, "cpu"), (1, 40, 64, "cuda")],
)
def test_VAE(channels: int, z_dim: int, batch_size: int, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Skip GPU test because cuda is not available")
    vae = VAE(channels=channels, z_dim=z_dim)
    vae.to(device)
    vae.train()
    x = torch.rand(batch_size, channels, 64, 64).to(device)
    lamb, mean, logvar = vae(x)
    assert lamb.shape == x.shape
    assert mean.shape == (batch_size, z_dim)
    assert logvar.shape == (batch_size, z_dim)
    loss, recons_loss, kl_loss = vae.loss(x=x, lamb=lamb, mean=mean, logvar=logvar)
    assert loss.shape == ()
    assert type(loss) == torch.Tensor
    assert recons_loss.shape == ()
    assert type(recons_loss) == torch.Tensor
    assert kl_loss.shape == ()
    assert type(kl_loss) == torch.Tensor
    loss.backward()


@pytest.mark.parametrize(
    ("channels", "z_dim", "batch_size", "device", "beta"),
    [
        (1, 5, 5, "cpu", 100),
        (3, 15, 32, "cuda", 15.5),
    ],
)
def test_BetaVAE(channels: int, z_dim: int, batch_size: int, device: str, beta: float):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Skip GPU test because cuda is not available")
    beta_vae = BetaVAE(channels=channels, z_dim=z_dim, beta=beta)
    beta_vae.to(device)
    beta_vae.train()
    x = torch.rand(batch_size, channels, 64, 64).to(device)
    lamb, mean, logvar = beta_vae(x)
    assert lamb.shape == x.shape
    assert mean.shape == (batch_size, z_dim)
    assert logvar.shape == (batch_size, z_dim)
    loss, recons_loss, kl_loss = beta_vae.loss(x=x, lamb=lamb, mean=mean, logvar=logvar)
    assert loss.shape == ()
    assert type(loss) == torch.Tensor
    assert recons_loss.shape == ()
    assert type(recons_loss) == torch.Tensor
    assert kl_loss.shape == ()
    assert type(kl_loss) == torch.Tensor
    loss.backward()


@pytest.mark.parametrize(
    (
        "channels",
        "z_dim",
        "batch_size",
        "device",
        "c_start",
        "c_end",
        "gamma",
        "epochs",
    ),
    [
        (1, 5, 5, "cpu", 0, 10, 1000, 3),
        (3, 15, 32, "cuda", 1, 10.5, 100, 4),
    ],
)
def test_AnnealedVAE(
    channels: int,
    z_dim: int,
    batch_size: int,
    device: str,
    c_start: float,
    c_end: float,
    gamma: float,
    epochs: int,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Skip GPU test because cuda is not available")
    annealed_vae = AnnealedVAE(
        channels=channels,
        z_dim=z_dim,
        c_start=c_start,
        c_end=c_end,
        gamma=gamma,
        epochs=epochs,
    )
    annealed_vae.to(device)
    annealed_vae.train()
    x = torch.rand(batch_size, channels, 64, 64).to(device)
    for i in range(epochs):
        lamb, mean, logvar = annealed_vae(x)
        assert lamb.shape == x.shape
        assert mean.shape == (batch_size, z_dim)
        assert logvar.shape == (batch_size, z_dim)
        loss, recons_loss, kl_loss = annealed_vae.loss(
            x=x, lamb=lamb, mean=mean, logvar=logvar
        )
        assert loss.shape == ()
        assert type(loss) == torch.Tensor
        assert recons_loss.shape == ()
        assert type(recons_loss) == torch.Tensor
        assert kl_loss.shape == ()
        assert type(kl_loss) == torch.Tensor
        loss.backward()
        annealed_vae.next_epoch()
