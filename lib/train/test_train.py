import os

import pytest
import torch

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
        "save_model",
        "save_path",
    ),
    [
        ("shapes3d", 100, 32, 0, 10, "cpu", 1e-3, 1, -1, False, "model.pt"),
        ("shapes3d", 200, 10, 1, 20, "cuda", 1e-3, 2, 2, True, "model.pt"),
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
    save_model,
    save_path,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Skip GPU test because cuda is not available")
    train_loss_history = train(
        dataset,
        train_size,
        batch_size,
        seed,
        z_dim,
        device,
        lr,
        epochs,
        train_log,
        save_model,
        save_path,
    )
    assert len(train_loss_history) == epochs
    assert type(train_loss_history[0]) == float
    if save_model:
        assert os.path.isfile("model.pt")
        os.remove("model.pt")
