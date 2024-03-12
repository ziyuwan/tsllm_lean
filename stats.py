import pickle as pkl
from argparse import ArgumentParser
from typing import List
from prover.search_tree import Status
from mcts_prover.mcts import SearchResult

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pkl-path", type=str, required=True)
    args = parser.parse_args()  
    
    x: List[SearchResult] = pkl.load(open(args.pkl_path, "rb"))
    
    num_proved = num_failed = num_discarded = 0
    num_timeout = 0
    
    llm_time = 0.
    env_time = 0.
    
    num_total_nodes = 0
    num_searched_nodes = 0
    cnt = 0
    for r in x:
        # print(r)
        # import pdb; pdb.set_trace()
        if r is None:
            num_discarded += 1
            continue
        elif str(r.status) == str(Status.PROVED):
            num_proved += 1
        elif str(r.status) == str(Status.OPEN):
            num_timeout += 1
        else:
            num_failed += 1
        cnt += 1
        llm_time += r.actor_time
        env_time += r.environment_time
        num_total_nodes += r.num_total_nodes
        num_searched_nodes += r.num_searched_nodes
    
    
    print("Proved:", num_proved)
    print("Failed:", num_failed)
    print("Discarded:", num_discarded)
    print("Timeout:", num_timeout)
    print("LLM Time:", llm_time/cnt)
    print("Env Time:", env_time/cnt)
    print("Total Nodes:", num_total_nodes/cnt)
    print("Searched Nodes:", num_searched_nodes/cnt)