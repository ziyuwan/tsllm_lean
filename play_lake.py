from lean_dojo.container import Mount, NativeContainer, get_container
from lean_dojo.data_extraction.trace import get_traced_repo_path
from lean_dojo.interaction.dojo import (
    CommandState,
    Dojo,
    DojoInitError,
    State,
    TacticState,
    Theorem,
)
from prover.evaluate import _get_theorems
from argparse import ArgumentParser
from tqdm.contrib.concurrent import process_map
import re
from pathlib import Path
import re
import os
import sys
import json
import time
import signal
import shutil
from pathlib import Path
from loguru import logger
from tempfile import mkdtemp
from shutil import ignore_patterns
from subprocess import TimeoutExpired
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Dict, Any, Optional
from lean_dojo.constants import (
    TMP_DIR,
    LEAN3_PACKAGES_DIR,
    TACTIC_TIMEOUT,
    TACTIC_CPU_LIMIT,
    TACTIC_MEMORY_LIMIT,
    CACHE_DIR,
)
import pdb

parser = ArgumentParser()
parser.add_argument(
    "--data-path", type=str, default="data/leandojo_benchmark_4/random/"
)
parser.add_argument(
    "--split", type=str, choices=["train", "val", "test"], default="test"
)
# parser.add_argument("--num-build-workers", type=int, default=1)

args = parser.parse_args()

repo, thms, poss = _get_theorems(args.data_path, args.split, None, None, None, 1)

dojo = Dojo(thms[0], 60)


# Work in a temporary directory.
dojo.origin_dir = Path.cwd()
dojo.tmp_dir = Path(mkdtemp(dir=TMP_DIR))

dojo.build_cache_dir = Path(CACHE_DIR / f"lean4_build_cache/{dojo.entry.repo.name}")

try:
    print("BUILDING {} ...".format(dojo.build_cache_dir))
    dojo._install_handlers()

    # Copy and `cd` into the repo.
    traced_repo_path = get_traced_repo_path(dojo.repo)

    if not dojo.build_cache_dir.exists():
        dojo.build_cache_dir.mkdir(parents=True)
        os.chdir(dojo.build_cache_dir)
        shutil.copytree(
            traced_repo_path,
            dojo.repo.name,
            ignore=ignore_patterns("*.dep_paths", "*.ast.json", "*.trace.xml"),
        )
        os.chdir(dojo.repo.name)

        # Replace the human-written proof with a `repl` tactic.
        try:
            traced_file = dojo._locate_traced_file(traced_repo_path)
        except FileNotFoundError:
            raise DojoInitError(
                f"Cannot find the *.ast.json file for {dojo.entry} in {traced_repo_path}."
            )

        dojo._modify_file(traced_file)

        # The REPL code cannot be used to interact with its own dependencies.
        unsupported_deps = dojo._get_unsupported_deps(traced_repo_path)

        # Run the modified file in a container.
        dojo.container = get_container()
        if dojo.uses_lean3 and isinstance(dojo.container, NativeContainer):
            logger.warning(
                "Docker is strongly recommended when using LeanDojo with Lean 3. See https://leandojo.readthedocs.io/en/latest/user-guide.html#advanced-running-within-docker."
            )
        logger.debug(f"Launching the proof using {type(dojo.container)}")
        mts = [Mount(Path.cwd(), Path(f"/workspace/{dojo.repo.name}"))]
        if dojo.repo.uses_lean3:
            cmd = f"lean {dojo.file_path}"
            cpu_limit = TACTIC_CPU_LIMIT
            memory_limit = TACTIC_MEMORY_LIMIT
        else:
            dojo.container.run(
                "lake build Lean4Repl",
                mts,
                as_current_user=True,
                capture_output=True,
                work_dir=f"/workspace/{dojo.repo.name}",
                cpu_limit=None,
                memory_limit=None,
                envs={},
            )

            # assert re.fullmatch(r"\d+g", TACTIC_MEMORY_LIMIT)
            # memory_limit = 1024 * int(TACTIC_MEMORY_LIMIT[:-1])
            # cmd = f"lake env lean --threads={TACTIC_CPU_LIMIT} --memory={memory_limit} {dojo.file_path}"
            # cpu_limit = memory_limit = None

    print("FINISH_BUILDING")
    print(dojo.build_cache_dir / dojo.repo.name)

    shutil.copytree(
        dojo.build_cache_dir / dojo.repo.name,
        dojo.tmp_dir / dojo.repo.name,
    )
    print("Copying to {}".format(dojo.tmp_dir / dojo.repo.name))
    assert re.fullmatch(r"\d+g", TACTIC_MEMORY_LIMIT)
    memory_limit = 1024 * int(TACTIC_MEMORY_LIMIT[:-1])
    cmd = f"lake env lean --threads={TACTIC_CPU_LIMIT} --memory={memory_limit} {dojo.file_path}"
    print("Try runing:")
    print("\tcd {}".format(dojo.tmp_dir / dojo.repo.name))
    print(cmd)

except Exception as ex:
    import pdb

    pdb.set_trace()
    os.chdir(dojo.origin_dir)
    shutil.rmtree(dojo.tmp_dir)
    raise ex
