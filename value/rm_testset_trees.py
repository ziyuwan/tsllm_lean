from pathlib import Path
import json
from prover.evaluate import _get_theorems_from_files
from argparse import ArgumentParser
import os


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    _, test_theorems, _ = _get_theorems_from_files(
        args.data_path,
        split="test",
        file_path=None,
        full_name=None,
        name_filter=None,
        num_theorems=None,
    )
    _, val_theorems, _ = _get_theorems_from_files(
        args.data_path,
        split="val",
        file_path=None,
        full_name=None,
        name_filter=None,
        num_theorems=None,
    )
    theorems = test_theorems + val_theorems

    target_dir = Path(args.target_dir)
    files_to_rm = []
    for thm in theorems:
        tree_path = target_dir / f"{thm.uid}.jsonl"
        tree_path1 = target_dir / f"{thm.uid1}.jsonl"
        if tree_path.exists():
            print("File exists:", tree_path)
            files_to_rm.append(tree_path)
        elif tree_path1.exists():
            print("File exists:", tree_path1)
            files_to_rm.append(tree_path1)

    print("Total Number of files to remove: {}".format(len(files_to_rm)))
    if not args.dry_run:
        for file_path in files_to_rm:
            os.remove(file_path)
