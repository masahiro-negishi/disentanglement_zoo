import matplotlib.pyplot as plt


def plot_loss_curve(loss_history: list, save_path: str, title: str):
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(save_path)
    plt.close()
