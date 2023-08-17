import matplotlib.pyplot as plt
import torch


def visualize_reconstruction(
    model,
    dataloader: torch.utils.data.DataLoader,
    num: int,
    save_path: str,
    device: str,
):
    """visualize reconstruction of images

    Args:
        model (a child of nn.Module): model
        dataloader (torch.utils.data.DataLoader): dataloader
        num (int): # of samples to visualize
        save_path (str): path to save the visualization
        device (str): device
    """
    assert num <= len(dataloader.dataset), "num must be less than the dataset size"
    model.eval()
    ground_truth = []
    reconstruction = []
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            lamb, _, _ = model(images)
            if len(ground_truth) < num:
                ground_truth.append(
                    images.cpu()[: min(num - len(ground_truth), len(images))]
                )
                reconstruction.append(
                    lamb.cpu()[: min(num - len(reconstruction), len(images))]
                )
            if len(ground_truth) == num:
                break
    ground_truth = torch.cat(ground_truth, dim=0)  # (num, C, H, W)
    reconstruction = torch.cat(reconstruction, dim=0)  # (num, C, H, W)

    # output figure. The first row is ground truth, and the second row is reconstruction
    fig = plt.figure(figsize=(num, 2))
    for i in range(num):
        ax1 = fig.add_subplot(2, num, i + 1)
        ax1.imshow(ground_truth[i].permute(1, 2, 0))
        ax1.axis("off")
        ax2 = fig.add_subplot(2, num, i + 1 + num)
        ax2.imshow(reconstruction[i].permute(1, 2, 0))
        ax2.axis("off")
    plt.savefig(save_path)
