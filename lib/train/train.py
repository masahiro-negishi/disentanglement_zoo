import torch

from ..data.dataset import prepare_dataloader
from ..method.vae import VAE


def train(
    dataset: str,
    train_size: int,
    batch_size: int,
    seed: int,
    z_dim: int,
    device: str,
    lr: float,
    epochs: int,
    train_log: int,
    save_model: bool,
    save_path: str = "model.pt",
):
    """train a model

    Args:
        dataset (str): dataset name
        train_size (int): # of samples in train set
        batch_size (int): batch size for training
        seed (int): random seed
        z_dim (int): dimension of latent variable
        device (str): device name
        lr (float): learning rate
        epochs (int): # of epochs
        train_log (int): log every train_log epochs. If set to -1, no log.
        save_model (bool): save model or not
        save_path (str, optional): path to save model.

    Returns:
        train_loss_history (list): train loss history
    """
    # device check
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda is not available")

    # preapre dataloader
    trainloader = prepare_dataloader(
        dataset=dataset, train_size=train_size, batch_size=batch_size, seed=seed
    )

    # prepare model
    model = VAE(channels=trainloader.dataset.observation_shape[0], z_dim=z_dim).to(
        device
    )

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    train_loss_history = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for images, _ in trainloader:  # unsupervised learning
            images = images.to(device)
            optimizer.zero_grad()
            lamb, mean, logvar = model(images)
            loss = model.loss(images, lamb, mean, logvar)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        train_loss_history.append(loss_sum / len(trainloader))
        if train_log != -1 and (epoch + 1) % train_log == 0:
            print(f"epoch: {epoch}, loss: {loss.item()}")

    # save
    if save_model:
        torch.save(model.state_dict(), save_path)

    return train_loss_history
