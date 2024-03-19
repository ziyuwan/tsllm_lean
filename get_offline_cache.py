from prover.evaluate import _get_theorems
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data-path", type=str, required=True)
parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)

args = parser.parse_args()

x = _get_theorems(
    args.data_path,
    args.split,
    None,
    None,
    None,
    None
)