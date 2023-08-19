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
    parser_eval = subparsers.add_parser("eval")
    subparsers_model = parser_train.add_subparsers(dest="model_name")
    _ = subparsers_model.add_parser("VAE")
    parser_BetaVAE = subparsers_model.add_parser("BetaVAE")

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

    parser_BetaVAE.add_argument("--beta", type=float, required=True)

    parser_eval.add_argument("--save_dir", type=str, required=True)
    parser_eval.add_argument("--device", type=str, required=True)
    parser_eval.add_argument("--recons", action="store_true")
    parser_eval.add_argument("--num_recons", type=int, default=5)
    parser_eval.add_argument("--latent_dist", action="store_true")
    parser_eval.add_argument("--change_one_variable", action="store_true")
    parser_eval.add_argument("--num_change_one_variable", type=int, default=2)

    args = parser.parse_args()
    kwargs = args.__dict__

    if args.exec_type == "train":
        kwargs.pop("exec_type")
        train(**kwargs)
    else:
        kwargs.pop("exec_type")
        eval(**kwargs)


if __name__ == "__main__":
    main()
