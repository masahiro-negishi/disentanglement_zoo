import os

import pytest
import torch

from ..data.dataset import prepare_dataloader
from ..method.vae import VAE
from .reconstruction import visualize_reconstruction


@pytest.mark.parametrize(
    ("dataset", "num", "save_dir", "device"),
    [
        (
            "shapes3d",
            5,
            os.path.join(os.path.dirname(__file__), "..", "..", "result", "test"),
            "cpu",
        ),
        (
            "shapes3d",
            3,
            os.path.join(os.path.dirname(__file__), "..", "..", "result", "test"),
            "cuda",
        ),
    ],
)
def test_visualize_reconstruction(dataset: str, num: int, save_dir: str, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Skip GPU test because cuda is not available")
    trainloader, evalloader = prepare_dataloader(
        dataset=dataset,
        train_size=10,
        eval_size=6,
        batch_size=2,
        seed=0,
        only_initial_shuffle_train=True,
    )
    model = VAE(channels=trainloader.dataset.observation_shape[0], z_dim=10).to(device)
    os.makedirs(os.path.join(save_dir, "eval"))
    visualize_reconstruction(
        model,
        trainloader,
        num,
        os.path.join(save_dir, "eval", "recons_train.png"),
        device,
    )
    visualize_reconstruction(
        model,
        evalloader,
        num,
        os.path.join(save_dir, "eval", "recons_eval.png"),
        device,
    )
    assert os.path.exists(os.path.join(save_dir, "eval", "recons_train.png"))
    assert os.path.exists(os.path.join(save_dir, "eval", "recons_eval.png"))
    os.remove(os.path.join(save_dir, "eval", "recons_train.png"))
    os.remove(os.path.join(save_dir, "eval", "recons_eval.png"))
    os.rmdir(os.path.join(save_dir, "eval"))
    os.rmdir(save_dir)
