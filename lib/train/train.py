import json
import os
import random

import numpy as np
import torch

from ..data.dataset import prepare_dataloader
from ..method.vae import VAE, BetaVAE, AnnealedVAE
from .loss_curve import plot_loss_curve


def train(
    dataset: str,
    train_size: int,
    eval_size: int,
    batch_size: int,
    model_name: str,
    seed: int,
    z_dim: int,
    device: str,
    lr: float,
    epochs: int,
    train_log: int,
    save: bool,
    save_dir: str = ".",
    **kwargs,
):
    """train a model

    Args:
        dataset (str): dataset name
        train_size (int): # of samples in train set
        eval_size (int): # of samples in eval set
        batch_size (int): batch size for training
        model_name (str): model name
        seed (int): random seed
        z_dim (int): dimension of latent variable
        device (str): device name
        lr (float): learning rate
        epochs (int): # of epochs
        train_log (int): log every train_log epochs. If set to -1, no log.
        save (bool): save model and other configurations or not
        save_dir (str, optional): directory to save model.
        **kwargs: other arguments

    Returns:
        None
    """
    # device check
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda is not available")

    # set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # preapre dataloader
    trainloader, evalloader = prepare_dataloader(
        dataset=dataset,
        train_size=train_size,
        eval_size=eval_size,
        batch_size=batch_size,
        seed=seed,
        only_initial_shuffle_train=False,
    )

    # prepare model
    torch.manual_seed(seed)
    if model_name == "VAE":
        model = VAE(channels=trainloader.dataset.observation_shape[0], z_dim=z_dim).to(
            device
        )
    elif model_name == "BetaVAE":
        model = BetaVAE(
            channels=trainloader.dataset.observation_shape[0],
            z_dim=z_dim,
            beta=kwargs["beta"],
        ).to(device)
    elif model_name == "AnnealedVAE":
        model = AnnealedVAE(
            channels=trainloader.dataset.observation_shape[0],
            z_dim=z_dim,
            c_start=kwargs["c_start"],
            c_end=kwargs["c_end"],
            gamma=kwargs["gamma"],
            epochs=epochs,
        ).to(device)
    else:
        raise ValueError("invalid model name")

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    train_loss_history = []
    train_recons_loss_history = []
    train_kl_loss_history = []
    eval_loss_history = []
    eval_recons_loss_history = []
    eval_kl_loss_history = []
    torch.manual_seed(seed)
    for epoch in range(epochs):
        # training
        model.train()
        train_loss_sum = 0
        train_recons_loss_sum = 0
        train_kl_loss_sum = 0
        for images, _ in trainloader:  # unsupervised learning
            images = images.to(device)
            optimizer.zero_grad()
            lamb, mean, logvar = model(images)
            loss, recon_loss, kl_loss = model.loss(images, lamb, mean, logvar)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(images)
            train_recons_loss_sum += recon_loss.item() * len(images)
            train_kl_loss_sum += kl_loss.item() * len(images)

        # evaluation
        model.eval()
        eval_loss_sum = 0
        eval_recons_loss_sum = 0
        eval_kl_loss_sum = 0
        for images, _ in evalloader:
            images = images.to(device)
            lamb, mean, logvar = model(images)
            loss, recons_loss, kl_loss = model.loss(images, lamb, mean, logvar)
            eval_loss_sum += loss.item() * len(images)
            eval_recons_loss_sum += recon_loss.item() * len(images)
            eval_kl_loss_sum += kl_loss.item() * len(images)

        # update c if model is AnnealedVAE
        if model_name == "AnnealedVAE":
            model.next_epoch()

        # save loss history
        train_loss_history.append(train_loss_sum / len(trainloader.dataset))
        train_recons_loss_history.append(
            train_recons_loss_sum / len(trainloader.dataset)
        )
        train_kl_loss_history.append(train_kl_loss_sum / len(trainloader.dataset))
        eval_loss_history.append(eval_loss_sum / len(evalloader.dataset))
        eval_recons_loss_history.append(eval_recons_loss_sum / len(evalloader.dataset))
        eval_kl_loss_history.append(eval_kl_loss_sum / len(evalloader.dataset))
        if train_log != -1 and (epoch + 1) % train_log == 0:
            print(
                "epoch: {}".format(epoch),
                "train_loss: {:.2f}".format(train_loss_history[-1]),
                "train_recons_loss: {:.2f}".format(train_recons_loss_history[-1]),
                "train_kl_loss: {:.2f}".format(train_kl_loss_history[-1]),
                "eval_loss: {:.2f}".format(eval_loss_history[-1]),
                "eval_recons_loss: {:.2f}".format(eval_recons_loss_history[-1]),
                "eval_kl_loss: {:.2f}".format(eval_kl_loss_history[-1]),
            )

    # save
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(os.path.join(save_dir, "train"))
        # model
        torch.save(
            model.to("cpu").state_dict(), os.path.join(save_dir, "train", "model.pt")
        )
        # settings
        settings = {
            "dataset": dataset,
            "train_size": train_size,
            "eval_size": eval_size,
            "batch_size": batch_size,
            "model_name": model_name,
            "seed": seed,
            "z_dim": z_dim,
            "device": device,
            "lr": lr,
            "epochs": epochs,
            "train_log": train_log,
        }
        if model_name == "BetaVAE":
            settings["beta"] = kwargs["beta"]
        elif model_name == "AnnealedVAE":
            settings["c_start"] = kwargs["c_start"]
            settings["c_end"] = kwargs["c_end"]
            settings["gamma"] = kwargs["gamma"]
        with open(os.path.join(save_dir, "train", "settings.json"), "w") as f:
            json.dump(settings, f)
        # loss curve
        plot_loss_curve(
            train_loss_history,
            train_recons_loss_history,
            train_kl_loss_history,
            os.path.join(save_dir, "train", "train_loss.png"),
            "Train Loss",
        )
        plot_loss_curve(
            eval_loss_history,
            eval_recons_loss_history,
            eval_kl_loss_history,
            os.path.join(save_dir, "train", "eval_loss.png"),
            "Eval Loss",
        )
        with open(os.path.join(save_dir, "train", "loss.json"), "w") as f:
            json.dump(
                {
                    "train_loss": train_loss_history,
                    "train_recons_loss": train_recons_loss_history,
                    "train_kl_loss": train_kl_loss_history,
                    "eval_loss": eval_loss_history,
                    "eval_recons_loss": eval_recons_loss_history,
                    "eval_kl_loss": eval_kl_loss_history,
                },
                f,
            )
