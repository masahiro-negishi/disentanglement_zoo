import json
import os
import random

import numpy as np
import torch

from ..data.dataset import prepare_dataloader
from ..method.vae import VAE
from .reconstruction import visualize_reconstruction


def eval(save_dir: str, reconstruction: bool, num: int, device: str):
    """evaluate trained model

    Args:
        save_dir (str): path to directory where model is saved and evaluation results will be saved
        reconstruction (bool): whether to do reconstruction
        num (int): # of samples to do reconstruction
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
    model = VAE(channels=3, z_dim=settings["z_dim"])
    if device == "cuda":
        model.load_state_dict(
            torch.load(os.path.join(save_dir, "train", "model.pt"), map_location="cuda")
        )
        model.to(device)
    else:
        model.load_state_dict(torch.load(os.path.join(save_dir, "train", "model.pt")))
    trainloader = prepare_dataloader(
        dataset=settings["dataset"],
        train_size=settings["train_size"],
        batch_size=settings["batch_size"],
    )

    # evaluate
    if not os.path.exists(os.path.join(save_dir, "eval")):
        os.mkdir(os.path.join(save_dir, "eval"))
    if reconstruction:
        visualize_reconstruction(
            model=model,
            dataloader=trainloader,
            num=num,
            save_path=os.path.join(save_dir, "eval", "reconstruction.png"),
            device=device,
        )
