import math
import numpy as np
from mcts_prover.mcts_node import ProofFinishedNode, ErrorNode
from mcts_prover.mcts_node import InternalTreeNode


def compute_puct(
    parent_node: InternalTreeNode, pb_c_base: float, pb_c_init: float
) -> np.ndarray:
    assert parent_node.out_edges
    values = []
    logps = []
    vis_cnts = []
    for edge in parent_node.out_edges:
        child_node = edge.dst
        value = child_node.value
        values.append(value)
        logps.append(edge.logp)
        vis_cnts.append(child_node.visit_count)

    values = np.array(values)
    vis_cnts = np.array(vis_cnts)
    priors = np.exp(np.array(logps))

    pb_c = math.log((parent_node.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c = pb_c * np.sqrt(parent_node.visit_count) / (vis_cnts + 1)
    prior_score = pb_c * priors

    assert len(prior_score.shape) == 1, f"prior_score: {prior_score}"
    assert len(values.shape) == 1, f"values: {values}"

    return prior_score + values


# TODO(ziyu): try vectorize it.

# _ucb_score(self, parent: Node, child: Node) -> float:
#         """
#         Overview:
#             Compute UCB score. The score for a node is based on its value, plus an exploration bonus based on the prior.
#         Arguments:
#             - parent (:obj:`Class Node`): Current node.
#             - child (:obj:`Class Node`): Current node's child.
#         Returns:
#             - score (:obj:`Bool`): The UCB score.
#         """
#         pb_c = (
#             math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base)
#             + self._pb_c_init
#         )
#         pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

#         prior_score = pb_c * child.prior_p
#         value_score = child.value

#         return prior_score + value_score
