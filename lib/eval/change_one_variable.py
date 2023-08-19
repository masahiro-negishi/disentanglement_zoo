import matplotlib.pyplot as plt
import torch
import os


def visualize_change_one_variable(
    model,
    dataloader: torch.utils.data.DataLoader,
    save_dir: str,
    device: str,
    num_input: int = 2,
    num_sample: int = 7,
):
    """visualize reconstruction of images

    Args:
        model (a child of nn.Module): model
        dataloader (torch.utils.data.DataLoader): dataloader
        save_dir (str): path to the directory to save the visualization
        device (str): device
        num_input (int, optional): # of input images. Defaults to 2.
        num_sample (int, optional): # of samples per one input. Defaults to 7.
    """
    assert num_input <= len(
        dataloader.dataset
    ), "num must be less than the dataset size"
    # ground truth
    model.eval()
    sampled = [torch.tensor([]) for _ in range(model.z_dim)]
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            z, mean, logvar = model.encode(images)
            z_part = z[: min(num_input - sampled[0].shape[0] // num_sample, z.shape[0])]
            for dim in range(model.z_dim):
                # change one dimension
                z_, _ = (
                    z_part.clone().repeat(num_sample, 1).sort(dim=0)
                )  # (z_part.shape[0] * num_sample, z_dim)
                z_[:, dim] = torch.linspace(-3, 3, num_sample, device=device).repeat(
                    z_part.shape[0]
                )  # (z_part.shape[0] * num_sample, z_dim)
                lamb = model.decode(z_)  # (z_part.shape[0] * num_sample, C, H, W)
                sampled[dim] = torch.cat([sampled[dim], lamb.cpu()], dim=0)
            if sampled[0].shape[0] // num_sample == num_input:
                break

    # output figure. The first row is ground truth, and the second row is reconstruction
    for dim in range(model.z_dim):
        fig = plt.figure(figsize=(num_sample, num_input))
        for i in range(num_input):
            for j in range(num_sample):
                ax = fig.add_subplot(num_input, num_sample, i * num_sample + j + 1)
                ax.imshow(sampled[dim][i * num_sample + j].permute(1, 2, 0))
                ax.axis("off")
        fig.savefig(os.path.join(save_dir, f"dim_{dim}.png"))
        plt.close(fig)
