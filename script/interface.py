import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from lib.train.train import train


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
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_dir", type=str, default="model.pt")
    args = parser.parse_args()

    train(
        dataset=args.dataset,
        train_size=args.train_size,
        batch_size=args.batch_size,
        seed=args.seed,
        z_dim=args.z_dim,
        device=args.device,
        lr=args.lr,
        epochs=args.epochs,
        train_log=args.train_log,
        save=args.save,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
