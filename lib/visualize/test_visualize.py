import os

import pytest
import torch

from .loss_curve import plot_loss_curve


@pytest.mark.parametrize(
    ("loss_history", "save_path", "title"),
    [
        ([100, 50, 25, 12.5, 6.25], "test.png", "test"),
    ],
)
def test_plot_loss_curve(loss_history: list, save_path: str, title: str):
    plot_loss_curve(loss_history, save_path, title)
    assert os.path.exists(save_path)
    os.remove(save_path)
