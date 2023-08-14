import os

import pytest

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
        save_model,
        save_path,
    )
    if save_model:
        assert os.path.isfile("model.pt")
        os.remove("model.pt")
