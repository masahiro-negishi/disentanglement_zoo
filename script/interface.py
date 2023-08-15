import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from lib.train.train import train
from lib.visualize.loss_curve import plot_loss_curve


def main():
    print("start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--z_dim", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--train_log", type=int, required=True)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_path", type=str, default="model.pt")
    parser.add_argument("--plot_loss", action="store_true")
    parser.add_argument("--loss_figure_path", type=str, default="loss.png")
    args = parser.parse_args()

    train_loss_history = train(
        dataset=args.dataset,
        train_size=args.train_size,
        batch_size=args.batch_size,
        seed=args.seed,
        z_dim=args.z_dim,
        device=args.device,
        lr=args.lr,
        epochs=args.epochs,
        train_log=args.train_log,
        save_model=args.save_model,
        save_path=args.save_path,
    )

    if args.plot_loss:
        plot_loss_curve(
            train_loss_history,
            args.loss_figure_path,
            f"dataset={args.dataset}, seed={args.seed}",
        )


if __name__ == "__main__":
    main()
