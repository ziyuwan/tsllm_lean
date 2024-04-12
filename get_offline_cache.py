from prover.evaluate import _get_theorems
from argparse import ArgumentParser
from lean_dojo import Dojo
from tqdm.contrib.concurrent import process_map

parser = ArgumentParser()
parser.add_argument("--data-path", type=str, required=True)
parser.add_argument(
    "--split", type=str, choices=["train", "val", "test"], required=True
)
parser.add_argument("--num-build-workers", type=int, default=1)

args = parser.parse_args()

repo, thms, poss = _get_theorems(args.data_path, args.split, None, None, None, None)


def run_build_dojo(thm):
    with Dojo(thm, hard_timeout=10) as (dojo, init_state):
        pass


process_map(run_build_dojo, thms, max_workers=args.num_build_workers, chunksize=1)
