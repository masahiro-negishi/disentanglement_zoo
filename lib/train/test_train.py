import os

import pytest
import torch

from .loss_curve import plot_loss_curve
from .train import train


@pytest.mark.parametrize(
    (
        "dataset",
        "train_size",
        "batch_size",
        "seed",
        "z_dim",
        "device",
        "lr",
        "epochs",
        "train_log",
        "save",
        "save_dir",
    ),
    [
        (
            "shapes3d",
            100,
            32,
            0,
            10,
            "cpu",
            1e-3,
            1,
            -1,
            False,
            ".",
        ),
        (
            "shapes3d",
            200,
            10,
            1,
            20,
            "cuda",
            1e-3,
            2,
            2,
            True,
            os.path.join(os.path.dirname(__file__), "..", "result", "test"),
        ),
    ],
)
def test_train(
    dataset,
    train_size,
    batch_size,
    seed,
    z_dim,
    device,
    lr,
    epochs,
    train_log,
    save,
    save_dir,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Skip GPU test because cuda is not available")
    train(
        dataset,
        train_size,
        batch_size,
        seed,
        z_dim,
        device,
        lr,
        epochs,
        train_log,
        save,
        save_dir,
    )
    if save:
        assert os.path.exists(os.path.join(save_dir, "train", "model.pt"))
        assert os.path.exists(os.path.join(save_dir, "train", "train_loss.png"))
        assert os.path.exists(os.path.join(save_dir, "train", "settings.json"))
        os.remove(os.path.join(save_dir, "train", "model.pt"))
        os.remove(os.path.join(save_dir, "train", "train_loss.png"))
        os.remove(os.path.join(save_dir, "train", "settings.json"))
        os.rmdir(os.path.join(save_dir, "train"))
        os.rmdir(save_dir)


@pytest.mark.parametrize(
    ("loss_history", "save_path", "title"),
    [
        ([100, 50, 25, 12.5, 6.25], "train_loss.png", "loss"),
    ],
)
def test_plot_loss_curve(loss_history: list, save_path: str, title: str):
    plot_loss_curve(loss_history, save_path, title)
    assert os.path.exists(save_path)
    os.remove(save_path)
