from mcts_prover.mcts_node import (
    Node,
    Edge,
    InternalTreeNode,
    ErrorNode,
    ProofFinishedNode,
    Status,
)
from lean_dojo import TacticState
from typing import Optional


def _find_tree_root(node: Node):
    if not node.is_root:
        return _find_tree_root(node.in_edge.src)
    else:
        return node


def _node_to_dict(node: Node):
    if isinstance(node, ProofFinishedNode):
        res = {
            "type": "ProofFinishedNode",
        }
    elif isinstance(node, ErrorNode):
        res = {
            "type": "ErrorNode",
        }
    elif isinstance(node, InternalTreeNode):
        res = {
            "type": "InternalTreeNode",
            "state": _tactic_state_to_dict(node.state),
            "out_edges": (
                [_directed_edge_to_dict(e) for e in node.out_edges]
                if node.out_edges
                else None
            ),
        }
    else:
        raise ValueError(f"Unknown node type: {node}")

    res.update(
        {
            "value_sum": node._value_sum,
            "visit_count": node.visit_count,
            "value": node.value,
            "status": str(node.status),
            # FIXME(ziyu): handle math.inf -> Infinity in json files.
            "distance_to_proof": str(node.distance_to_proof),
        }
    )

    return res


def _directed_edge_to_dict(edge: Edge):
    if edge is None:
        return None
    return {
        # "src": _node_to_dict(edge.src),
        "type": "OutEdge",
        "tactic": edge.tactic,
        "logp": edge.logp,
        "dst": _node_to_dict(edge.dst),
    }


def _tactic_state_to_dict(state: TacticState):
    if isinstance(state, TacticState):
        ts = state.pp
        return {
            "ts": ts,
            "id": state.id,
            "message": state.message,
        }

    else:
        ts = state.unsolved_tactic_state
        return {
            "ts": ts,
        }
