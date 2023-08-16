import os

import pytest
import torch

from ..data.dataset import prepare_dataloader
from ..method.vae import VAE
from .reconstruction import visualize_reconstruction


@pytest.mark.parametrize(
    ("dataset", "num", "save_path", "device"),
    [
        ("shapes3d", 5, "test.png", "cpu"),
        ("shapes3d", 3, "test.png", "cuda"),
    ],
)
def test_visualize_reconstruction(dataset: str, num: int, save_path: str, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Skip GPU test because cuda is not available")
    trainloader = prepare_dataloader(dataset=dataset, train_size=10, batch_size=2)
    model = VAE(channels=trainloader.dataset.observation_shape[0], z_dim=10).to(device)
    visualize_reconstruction(model, trainloader, num, save_path, device)
    assert os.path.exists(save_path)
    os.remove(save_path)
