import json
import os
import random

import numpy as np
import torch

from ..data.dataset import prepare_dataloader
from ..method.vae import VAE, BetaVAE, AnnealedVAE
from .latent_distribution import visualize_latent_distribution
from .reconstruction import visualize_reconstruction
from .change_one_variable import visualize_change_one_variable


def eval(
    save_dir: str,
    recons: bool,
    num_recons: int,
    latent_dist: str,
    change_one_variable: bool,
    num_change_one_variable: int,
    device: str,
):
    """evaluate trained model

    Args:
        save_dir (str): path to directory where model is saved and evaluation results will be saved
        recons (bool): whether to do reconstruction
        num_recons (int): # of samples to do reconstruction
        latent_dist (bool): whether to visualize latent distribution
        change_one_variable (bool): whether to visualize the results of changing one variable
        num_change_one_variable (int): # of input images for change_one_variable
        device (str): device to use
    """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"{save_dir} does not exist")

    # load settings
    with open(os.path.join(save_dir, "train", "settings.json"), "r") as f:
        settings = json.load(f)

    # recover model and dataloader
    torch.manual_seed(settings["seed"])
    np.random.seed(settings["seed"])
    random.seed(settings["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if settings["model_name"] == "VAE":
        model = VAE(channels=3, z_dim=settings["z_dim"])
    elif settings["model_name"] == "BetaVAE":
        model = BetaVAE(channels=3, z_dim=settings["z_dim"], beta=settings["beta"])
    elif settings["model_name"] == "AnnealedVAE":
        model = AnnealedVAE(
            channels=3,
            z_dim=settings["z_dim"],
            c_start=settings["c_start"],
            c_end=settings["c_end"],
            gamma=settings["gamma"],
            epochs=settings["epochs"],
        )
    else:
        raise ValueError(f"{settings['model_name']} is not supported")
    if device == "cuda":
        model.load_state_dict(
            torch.load(os.path.join(save_dir, "train", "model.pt"), map_location="cuda")
        )
        model.to(device)
    else:
        model.load_state_dict(torch.load(os.path.join(save_dir, "train", "model.pt")))
    trainloader, evalloader = prepare_dataloader(
        dataset=settings["dataset"],
        train_size=settings["train_size"],
        eval_size=settings["eval_size"],
        batch_size=settings["batch_size"],
        seed=settings["seed"],
        only_initial_shuffle_train=True,
    )

    # evaluate
    if not os.path.exists(os.path.join(save_dir, "eval")):
        os.mkdir(os.path.join(save_dir, "eval"))
    if recons:
        visualize_reconstruction(
            model=model,
            dataloader=trainloader,
            num=num_recons,
            save_path=os.path.join(save_dir, "eval", "recons_train.png"),
            device=device,
        )
        visualize_reconstruction(
            model=model,
            dataloader=evalloader,
            num=num_recons,
            save_path=os.path.join(save_dir, "eval", "recons_eval.png"),
            device=device,
        )
    if latent_dist:
        visualize_latent_distribution(
            model=model,
            dataloader=trainloader,
            save_path=os.path.join(save_dir, "eval", "latent_dist_train.png"),
            device=device,
        )
        visualize_latent_distribution(
            model=model,
            dataloader=evalloader,
            save_path=os.path.join(save_dir, "eval", "latent_dist_eval.png"),
            device=device,
        )
    if change_one_variable:
        os.mkdir(os.path.join(save_dir, "eval", "change_one_variable_train"))
        os.mkdir(os.path.join(save_dir, "eval", "change_one_variable_eval"))
        visualize_change_one_variable(
            model=model,
            dataloader=trainloader,
            save_dir=os.path.join(save_dir, "eval", "change_one_variable_train"),
            device=device,
            num_input=num_change_one_variable,
        )
        visualize_change_one_variable(
            model=model,
            dataloader=evalloader,
            save_dir=os.path.join(save_dir, "eval", "change_one_variable_eval"),
            device=device,
            num_input=num_change_one_variable,
        )
