import os

import pytest
import torch

from .loss_curve import plot_loss_curve
from .train import train


@pytest.mark.parametrize(
    (
        "dataset",
        "train_size",
        "eval_size",
        "batch_size",
        "model_name",
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
            50,
            32,
            "VAE",
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
            30,
            10,
            "BetaVAE",
            1,
            20,
            "cuda",
            1e-3,
            2,
            2,
            True,
            os.path.join(os.path.dirname(__file__), "..", "..", "result", "test"),
        ),
    ],
)
def test_train(
    dataset,
    train_size,
    eval_size,
    batch_size,
    model_name,
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
    if model_name == "VAE":
        train(
            dataset,
            train_size,
            eval_size,
            batch_size,
            model_name,
            seed,
            z_dim,
            device,
            lr,
            epochs,
            train_log,
            save,
            save_dir,
        )
    elif model_name == "BetaVAE":
        train(
            dataset,
            train_size,
            eval_size,
            batch_size,
            model_name,
            seed,
            z_dim,
            device,
            lr,
            epochs,
            train_log,
            save,
            save_dir,
            beta=5.0,
        )
    else:
        raise ValueError(f"{model_name} is not supported")
    if save:
        assert os.path.exists(os.path.join(save_dir, "train", "model.pt"))
        assert os.path.exists(os.path.join(save_dir, "train", "train_loss.png"))
        assert os.path.exists(os.path.join(save_dir, "train", "eval_loss.png"))
        assert os.path.exists(os.path.join(save_dir, "train", "settings.json"))
        assert os.path.exists(os.path.join(save_dir, "train", "loss.json"))
        os.remove(os.path.join(save_dir, "train", "model.pt"))
        os.remove(os.path.join(save_dir, "train", "train_loss.png"))
        os.remove(os.path.join(save_dir, "train", "eval_loss.png"))
        os.remove(os.path.join(save_dir, "train", "settings.json"))
        os.remove(os.path.join(save_dir, "train", "loss.json"))
        os.rmdir(os.path.join(save_dir, "train"))
        os.rmdir(save_dir)


def test_plot_loss_curve():
    loss_history = list(range(10))
    recons_loss_history = list(range(10))
    kl_loss_history = list(range(10))
    plot_loss_curve(
        loss_history,
        recons_loss_history,
        kl_loss_history,
        "train_loss.png",
        "Train loss",
    )
    assert os.path.exists("train_loss.png")
    os.remove("train_loss.png")
