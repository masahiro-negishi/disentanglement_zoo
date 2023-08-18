import matplotlib.pyplot as plt


def plot_loss_curve(
    loss_history: list,
    recons_loss_history: list,
    kl_loss_history: list,
    save_path: str,
    title: str,
):
    plt.plot(loss_history, label="total loss")
    plt.plot(recons_loss_history, label="reconstruction loss")
    plt.plot(kl_loss_history, label="kl loss")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(save_path)
    plt.close()
