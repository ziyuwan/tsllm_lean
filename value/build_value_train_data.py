import json
import jsonlines
from pathlib import Path
from argparse import ArgumentParser
from prover.search_tree import Status
from typing import Dict

TOTAL_POSITIVE = 0
TOTAL_NEGATIVE = 0
TOTAL_NEUTRAL = 0


def tranverse_and_write(
    node: Dict, writer, tactic: str = None, state_before: str = None
):
    if node["type"] == "InternalTreeNode":
        state = node["state"]["ts"]
        if node["status"] == "Status.OPEN":
            v = 0.0
            global TOTAL_NEUTRAL
            if node["out_edges"]:
                TOTAL_NEUTRAL += 1
        elif node["status"] == "Status.FAILED":
            v = -1.0
            global TOTAL_NEGATIVE
            TOTAL_NEGATIVE += 1
        elif node["status"] == "Status.PROVED":
            v = 1.0
            global TOTAL_POSITIVE
            TOTAL_POSITIVE += 1
        if v != 0 or node["out_edges"]:
            writer.write(
                {
                    "state": state,
                    "value": v,
                    "tactic": tactic,
                    "state_before": state_before,
                }
            )
            cnt = 1
        else:
            cnt = 0
        if node["out_edges"] is not None:
            for out_edge in node["out_edges"]:
                cnt += tranverse_and_write(
                    out_edge["dst"], writer, out_edge["tactic"], state
                )
    else:
        cnt = 0

    return cnt


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--merge_dir", type=str, required=True)
    parser.add_argument("--save_jsonl_path", type=str, required=True)

    args = parser.parse_args()

    MERGE_DIR = Path(args.merge_dir)
    WRITE_PATH = Path(args.save_jsonl_path)

    num_proved = num_failed = num_discarded = 0
    num_timeout = 0
    llm_time = 0.0
    env_time = 0.0
    num_total_nodes = 0
    num_searched_nodes = 0

    cnt = 0

    writer = jsonlines.open(WRITE_PATH, "w")

    n_processed = 0
    for file_path in MERGE_DIR.glob(r"*.json"):
        # print(file_path.as_posix())
        data = json.load(open(file_path, "r"))
        r = data["result"]
        if str(r["status"]) == str(Status.PROVED):
            num_proved += 1
        elif str(r["status"]) == str(Status.OPEN):
            num_timeout += 1
        else:
            num_failed += 1
        cnt += 1
        llm_time += r["actor_time"]
        env_time += r["environment_time"]
        num_total_nodes += r["num_total_nodes"]
        num_searched_nodes += r["num_searched_nodes"]

        n_processed += tranverse_and_write(data["search_tree"], writer)

        # if n_processed > 100: break

    print("Proved: {:.4%}({}/{})".format(num_proved / cnt, num_proved, cnt))
    print("#Failed:", num_failed)
    print("#Discarded:", num_discarded)
    print("#Timeout:", num_timeout)
    print("avg LLM Time:", llm_time / cnt)
    print("avg Env Time:", env_time / cnt)
    print("avg Total Nodes:", num_total_nodes / cnt)
    print("avg Searched Nodes:", num_searched_nodes / cnt)
    print("==  " * 10)
    print("Total Searched Nodes", num_searched_nodes)
    print("Num processed nodes: {}".format(n_processed))
    print(
        "POSTIVE: {}, NEGATIVE: {}, NEUTRAL: {}".format(
            TOTAL_POSITIVE, TOTAL_NEGATIVE, TOTAL_NEUTRAL
        )
    )

    writer.close()
