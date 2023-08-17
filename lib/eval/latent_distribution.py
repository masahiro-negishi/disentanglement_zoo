import matplotlib.pyplot as plt
import torch


def visualize_latent_distribution(
    model,
    dataloader: torch.utils.data.DataLoader,
    save_path: str,
    device: str,
):
    """visualize reconstruction of images

    Args:
        model (a child of nn.Module): model
        dataloader (torch.utils.data.DataLoader): dataloader
        save_path (str): path to save the visualization
        device (str): device
    """ "num must be less than the dataset size"
    model.eval()
    means = []
    logvars = []
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            _, mean, logvar = model.encoder(images)  # (B, z_dim)
            means.append(mean.cpu())
            logvars.append(logvar.cpu())
    means = torch.cat(means, dim=0)  # (len(dataloader.dataset), z_dim)
    logvars = torch.cat(logvars, dim=0)  # (len(dataloader.dataset), z_dim)
    z_dim = means.shape[1]

    # output figure. The first row is mean, and the second row is logvar. Each column is a dimension of z
    # make sure ticks are not overlapped
    fig = plt.figure(figsize=(z_dim, 4))
    mean_left = round(means.min().item())
    mean_center = round(means.mean().item())
    mean_right = round(means.max().item())
    logvar_left = round(logvars.min().item())
    logvar_center = round(logvars.mean().item())
    logvar_right = round(logvars.max().item())
    for i in range(z_dim):
        ax1 = fig.add_subplot(2, z_dim, i + 1)
        ax1.hist(means[:, i], bins=50)
        ax1.set_xticks([mean_left, mean_center, mean_right])
        ax1.set_yticks([])
        ax2 = fig.add_subplot(2, z_dim, i + 1 + z_dim)
        ax2.hist(logvars[:, i], bins=50)
        ax2.set_xticks([logvar_left, logvar_center, logvar_right])
        ax2.set_yticks([])
    plt.savefig(save_path)
