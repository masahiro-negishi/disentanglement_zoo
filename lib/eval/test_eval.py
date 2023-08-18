import os

import pytest
import torch

from ..data.dataset import prepare_dataloader
from ..method.vae import VAE
from .latent_distribution import visualize_latent_distribution
from .reconstruction import visualize_reconstruction


@pytest.fixture(scope="module", params=[("shapes3d", "cpu"), ("shapes3d", "cuda")])
def model_and_data(request):
    if request.param[1] == "cuda" and not torch.cuda.is_available():
        pytest.skip("Skip GPU test because cuda is not available")
    trainloader, evalloader = prepare_dataloader(
        dataset=request.param[0],
        train_size=10,
        eval_size=6,
        batch_size=2,
        seed=0,
        only_initial_shuffle_train=True,
    )
    model = VAE(channels=trainloader.dataset.observation_shape[0], z_dim=10).to(
        request.param[1]
    )
    return model, trainloader, evalloader, request.param[1]


@pytest.fixture(scope="function")
def directory_operation():
    save_dir = os.path.join(os.path.dirname(__file__), "..", "..", "result", "test")
    os.makedirs(os.path.join(save_dir, "eval"))
    yield save_dir
    for file in os.listdir(os.path.join(save_dir, "eval")):
        os.remove(os.path.join(save_dir, "eval", file))
    os.rmdir(os.path.join(save_dir, "eval"))
    os.rmdir(save_dir)


@pytest.mark.parametrize(
    ("num"),
    [
        (5),
        (3),
    ],
)
def test_visualize_reconstruction(num: int, model_and_data, directory_operation):
    model, trainloader, evalloader, device = model_and_data
    save_dir = directory_operation
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


def test_visualize_latent_distribution(model_and_data, directory_operation):
    model, trainloader, evalloader, device = model_and_data
    save_dir = directory_operation
    visualize_latent_distribution(
        model,
        trainloader,
        os.path.join(save_dir, "eval", "latent_dist_train.png"),
        device,
    )
    visualize_latent_distribution(
        model,
        evalloader,
        os.path.join(save_dir, "eval", "latent_dist_eval.png"),
        device,
    )
    assert os.path.exists(os.path.join(save_dir, "eval", "latent_dist_train.png"))
    assert os.path.exists(os.path.join(save_dir, "eval", "latent_dist_eval.png"))
