import json
import os
import random

import numpy as np
import torch

from ..data.dataset import prepare_dataloader
from ..method.vae import VAE
from .loss_curve import plot_loss_curve


def train(
    dataset: str,
    train_size: int,
    eval_size: int,
    batch_size: int,
    seed: int,
    z_dim: int,
    device: str,
    lr: float,
    epochs: int,
    train_log: int,
    save: bool,
    save_dir: str = ".",
):
    """train a model

    Args:
        dataset (str): dataset name
        train_size (int): # of samples in train set
        eval_size (int): # of samples in eval set
        batch_size (int): batch size for training
        seed (int): random seed
        z_dim (int): dimension of latent variable
        device (str): device name
        lr (float): learning rate
        epochs (int): # of epochs
        train_log (int): log every train_log epochs. If set to -1, no log.
        save (bool): save model and other configurations or not
        save_dir (str, optional): directory to save model.

    Returns:
        train_loss_history (list): train loss history
        model (a child of nn.Module): trained model
        trainloader (torch.utils.data.DataLoader): train dataloader
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
    model = VAE(channels=trainloader.dataset.observation_shape[0], z_dim=z_dim).to(
        device
    )

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    train_loss_history = []
    eval_loss_history = []
    torch.manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        for images, _ in trainloader:  # unsupervised learning
            images = images.to(device)
            optimizer.zero_grad()
            lamb, mean, logvar = model(images)
            loss = model.loss(images, lamb, mean, logvar)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(images)
        model.eval()
        eval_loss_sum = 0
        for images, _ in evalloader:
            images = images.to(device)
            lamb, mean, logvar = model(images)
            loss = model.loss(images, lamb, mean, logvar)
            eval_loss_sum += loss.item() * len(images)
        train_loss_history.append(train_loss_sum / len(trainloader.dataset))
        eval_loss_history.append(eval_loss_sum / len(evalloader.dataset))
        if train_log != -1 and (epoch + 1) % train_log == 0:
            print(
                f"epoch: {epoch}, train loss: {loss.item()}, eval loss: {loss.item()}"
            )

    # save
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(os.path.join(save_dir, "train"))
        torch.save(
            model.to("cpu").state_dict(), os.path.join(save_dir, "train", "model.pt")
        )
        settings = {
            "dataset": dataset,
            "train_size": train_size,
            "eval_size": eval_size,
            "batch_size": batch_size,
            "seed": seed,
            "z_dim": z_dim,
            "device": device,
            "lr": lr,
            "epochs": epochs,
            "train_log": train_log,
        }
        with open(os.path.join(save_dir, "train", "settings.json"), "w") as f:
            json.dump(settings, f)
        plot_loss_curve(
            train_loss_history,
            os.path.join(save_dir, "train", "train_loss.png"),
            "Train Loss",
        )
        plot_loss_curve(
            train_loss_history,
            os.path.join(save_dir, "train", "eval_loss.png"),
            "Eval Loss",
        )
