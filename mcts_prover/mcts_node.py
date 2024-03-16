import math
from lean_dojo.interaction.dojo import (
    LeanError,
    ProofFinished,
    ProofGivenUp,
    TacticState,
)
from enum import Enum
from abc import ABC, abstractmethod
from prover.search_tree import Node, Status, Edge
from dataclasses import dataclass
from functools import total_ordering
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Iterable, Union


class Status(Enum):
    """Status of a node or a proof search."""

    PROVED = "Proved"  # This node (or search) has at least one known proof.
    FAILED = "Failed"  # This node (or search) has exhausted its options and cannot be proved within the current run.
    OPEN = "Open"  # This node (or search) has not been proven or given up on yet.

class Node(ABC):
    _value_sum: float = 0.0
    visit_count: int = 0
    # virtual visit count

    # The edge from which src -> self
    in_edge: Optional["Edge"] = None

    def backup(self, value: float):
        if isinstance(self, InternalTreeNode):
            self._value_sum += value
            self.visit_count += 1
        if not self.is_root:
            self.in_edge.src.backup(value)
    
    @property
    @abstractmethod
    def status(self) -> Status:
        raise NotImplementedError

    @property
    @abstractmethod
    def distance_to_proof(self) -> int:
        "The smallest number of steps to a proof."
        raise NotImplementedError

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_root(self) -> bool:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def is_leaf(self) -> bool:
        raise NotImplementedError


@dataclass
class ProofFinishedNode(Node):
    inner: ProofFinished
    status = Status.PROVED
    distance_to_proof = 0
    is_terminal = True
    is_leaf = True
    is_root = False
    value = 1.0


@dataclass
class ErrorNode(Node):
    inner: Union[LeanError, TimeoutError, ProofGivenUp]
    status = Status.FAILED
    distance_to_proof = math.inf
    is_terminal = True
    is_leaf = True
    is_root = False
    value = -1.0


@total_ordering
@dataclass(unsafe_hash=True)
class InternalTreeNode(Node):
    """
    An internal node in the search tree, representing a nonterminal state.

    Nodes are sorted by _inverse_ priority, for compatibility with the `heapq` library.
    That is, node_a < node_b is true if node_a has _higher_ priority than node_b.
    """

    # Goal state this node represents. Two nodes are considered equal if their states
    # are equal; this is the only hashed field and must not be changed.
    state: TacticState = field(compare=True)

    critic_value: float = field(default=None, init=True, compare=False)
    cum_logp: float = field(default=0.0, init=True, compare=False)


    # All edges out of this node that we've considered, or None for unexplored nodes.
    # When a node is explored, this list is populated, and must not change after that.
    _out_edges: Optional[List["Edge"]] = field(
        default=None, init=False, compare=False, repr=False
    )

    # A node is proved if any child is proved, and failed if every child is failed
    # (or there are no children). A node that is proved or failed cannot change status
    # because nothing is ever added to out_edges. _status is recomputed on an as-needed
    # basis by children, since proving or failing a child may prove or fail this node.
    _status: Status = field(default=Status.OPEN, init=False, compare=False, repr=True)

    is_terminal = False  # type: ignore[override]

    # Number of steps separating this node from the end of a proof along the
    # optimal path. If unproved, infinity. Updated as needed by children.
    _distance_to_proof: float = field(
        default=math.inf, init=False, compare=False, repr=False
    )

    @property
    def out_edges(self):
        return self._out_edges

    # This setter implements exploring this node
    @out_edges.setter
    def out_edges(self, out_edges: Iterable["Edge"]) -> Optional[List["Edge"]]:
        if self.is_explored:
            raise RuntimeError("Node is already explored.")

        self._out_edges = list(out_edges)
        self._recompute_status()
        self._recompute_distance_to_proof()

    # A node is considered explored if we've evaluated the actor in the node to generate
    # a list of candidate children. Explored nodes are never re-searched.
    @property
    def is_explored(self) -> bool:
        return self.out_edges is not None

    @property
    def status(self) -> Status:
        return self._status

    @status.setter
    def status(self, s):
        self._status = s

    def _recompute_status(self):
        """
        Recursively update the status of the current node and its ancestors.
        """
        assert self.is_explored and self.out_edges is not None

        # If this node is proved or failed, nothing can change that
        if self._status != Status.OPEN:
            return

        # If any child is proved, this node is proved, and so are parents recursively
        if any(edge.dst.status == Status.PROVED for edge in self.out_edges):
            self._status = Status.PROVED

        # If all children failed, this node is failed. This may fail some parents too.
        if all(edge.dst.status == Status.FAILED for edge in self.out_edges):
            self._status = Status.FAILED

        # If this node was proved or failed, parents may need to recompute.
        # This is guaranteed to terminate because only open nodes can change, and
        # there are a finite number of open nodes in the tree.
        if self._status != Status.OPEN and not self.is_root:
            self.in_edge.src._recompute_status()

    @property
    def distance_to_proof(self) -> float:
        return self._distance_to_proof

    def _recompute_distance_to_proof(self):
        """
        Recursively update the distance_to_proof of the current node and its ancestors.
        """
        if self.out_edges:
            distance = min(edge.distance_to_proof() for edge in self.out_edges)
        else:
            distance = math.inf

        if distance < self._distance_to_proof:
            self._distance_to_proof = distance
            if not self.is_root:
                self.in_edge.src._recompute_distance_to_proof()

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return self.critic_value
        else:
            return self._value_sum / self.visit_count



    def __lt__(self, other: "InternalTreeNode") -> bool:
        return self.value < other.value

    def extract_proof(self) -> Optional[List["Edge"]]:
        """
        Extract a proof of the current node as a sequence of edges.
        """
        if self.status != Status.PROVED:
            return None
        assert self.is_explored

        proving_edge = min(
            self.out_edges,
            key=Edge.distance_to_proof,
        )

        if proving_edge.dst.is_terminal:
            # Base case: this edge is all that's required to finish the proof
            assert isinstance(proving_edge.dst, ProofFinishedNode)
            return [proving_edge]
        else:
            # Recursive case: prove the child, then add this edge
            assert isinstance(proving_edge.dst, InternalTreeNode)
            child_proof = proving_edge.dst.extract_proof()
            assert child_proof
            return [proving_edge, *child_proof]
    
    @property
    def is_root(self):
        if self.in_edge is None:
            return True
        else:
            return False

    @property
    def is_leaf(self):
        if self._out_edges is None:
            return True
        else:
            assert len(self.out_edges) > 0
            return None

    #########
    # Debug #
    #########

    def check_invariants(self):
        """
        Perform some sanity checks.
        """
        if not self.is_explored:
            assert self.status == Status.OPEN
            return  # Nothing more can be said about unexplored nodes

        assert self.in_edge.dst is self

        if self.out_edges == []:
            assert self.status == Status.FAILED
        else:
            for edge in self.out_edges:  # type: ignore
                assert edge.src is self

        if self.status == Status.PROVED:
            assert self.out_edges
            assert any(edge.dst.status == Status.PROVED for edge in self.out_edges)
            assert self.in_edge.dst.status == Status.PROVED

            proof_by_steps = self.extract_proof()
            assert proof_by_steps is not None
            assert self.distance_to_proof == len(proof_by_steps)

        elif self.status == Status.FAILED:
            assert self.out_edges is not None
            assert all(edge.dst.status == Status.FAILED for edge in self.out_edges)
            assert self.distance_to_proof == math.inf
            assert self.extract_proof() == None
        elif self.status == Status.OPEN:
            assert self.out_edges
            assert not any(edge.dst.status == Status.PROVED for edge in self.out_edges)
            assert not all(edge.dst.status == Status.FAILED for edge in self.out_edges)
            assert self.distance_to_proof == math.inf
            assert self.extract_proof() == None


@dataclass
class Edge:
    """An edge in the search tree, representing a tactic."""

    tactic: str
    logp: float
    src: InternalTreeNode = field(repr=False)
    dst: Node = field(repr=False)

    def distance_to_proof(self) -> float:
        return 1 + self.dst.distance_to_proof
