import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from lib.eval.eval import eval
from lib.train.train import train


def main():
    print("start")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="exec_type")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--dataset", type=str, required=True)
    parser_train.add_argument("--train_size", type=int, required=True)
    parser_train.add_argument("--eval_size", type=int, required=True)
    parser_train.add_argument("--batch_size", type=int, required=True)
    parser_train.add_argument("--seed", type=int, required=True)
    parser_train.add_argument("--z_dim", type=int, required=True)
    parser_train.add_argument("--device", type=str, required=True)
    parser_train.add_argument("--lr", type=float, required=True)
    parser_train.add_argument("--epochs", type=int, required=True)
    parser_train.add_argument("--train_log", type=int, required=True)
    parser_train.add_argument("--save", action="store_true")
    parser_train.add_argument("--save_dir", type=str, default=".")

    parser_eval = subparsers.add_parser("eval")
    parser_eval.add_argument("--save_dir", type=str, required=True)
    parser_eval.add_argument("--device", type=str, required=True)
    parser_eval.add_argument("--recons", action="store_true")
    parser_eval.add_argument("--num", type=int, default=5)
    parser_eval.add_argument("--latent_dist", action="store_true")

    args = parser.parse_args()

    if args.exec_type == "train":
        train(
            dataset=args.dataset,
            train_size=args.train_size,
            eval_size=args.eval_size,
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
    else:
        eval(
            save_dir=args.save_dir,
            recons=args.recons,
            num=args.num,
            latent_dist=args.latent_dist,
            device=args.device,
        )


if __name__ == "__main__":
    main()
