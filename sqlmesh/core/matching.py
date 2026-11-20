"""Minimum-cost bipartite matching used by the ``minimum_change`` table diff.

The problem solved here is deliberately small and bounded: after all fully-equal row copies
have been removed within a business key, at most ``max_unmatched_rows`` unmatched copies per
side per key are sent to the host (see :class:`sqlmesh.core.table_diff.TableDiff`). Each
source copy must be paired with at most one target copy, every pair has an integer cost (the
number of non-key cells that differ under the engine's comparison semantics) and we want a
maximum-cardinality pairing with the minimum total cost.

The implementation is a successive-shortest-path min-cost max-flow over the standard
``source -> left nodes -> right nodes -> sink`` network, using the Bellman-Ford/SPFA
relaxation. With both partitions bounded by the configured cap the graph never exceeds a few
hundred nodes, so avoiding an external dependency (e.g. scipy) is worth the simple algorithm.

Tie-breaking is part of the contract rather than an accident of iteration order. Every edge
cost is encoded as a pair ``(primary, secondary)`` where:

* ``primary`` is the actual number of differing cells, multiplied by a constant larger than
  any achievable secondary contribution, so minimizing the encoded cost first minimizes the
  real total change count;
* ``secondary`` is the target copy's stable ordinal. Among pairings with the same real total
  cost, the one with the smallest sum of target ordinals wins; because every left node always
  carries the same constant offset, this deterministically prefers matching earlier source
  ordinals to earlier target ordinals (lexicographically smallest assignment), independent of
  the physical order in which the database happens to return rows.
"""

from __future__ import annotations

import typing as t
from collections import deque


class _Edge(t.NamedTuple):
    to: int
    rev: int
    capacity: int
    cost: int


class MinCostMaxFlow:
    """A tiny min-cost max-flow solver for bounded bipartite assignment problems."""

    def __init__(self, n: int, inf: int = 10**18) -> None:
        self.n = n
        self.inf = inf
        self.graph: t.List[t.List[_Edge]] = [[] for _ in range(n)]

    def add_edge(self, fr: int, to: int, capacity: int, cost: int) -> None:
        forward = _Edge(to=to, rev=len(self.graph[to]), capacity=capacity, cost=cost)
        backward = _Edge(to=fr, rev=len(self.graph[fr]), capacity=0, cost=-cost)
        self.graph[fr].append(forward)
        self.graph[to].append(backward)

    def flow(self, source: int, sink: int, required_flow: int) -> t.Tuple[int, int]:
        """Push ``required_flow`` units while minimizing total cost.

        Returns ``(flow, cost)``. Raises ``ValueError`` if the requested flow is infeasible.
        """
        n = self.n
        total_flow = 0
        total_cost = 0

        while total_flow < required_flow:
            # SPFA: the residual graph contains negative edges, but no negative cycles.
            dist = [self.inf] * n
            prev_node = [-1] * n
            prev_edge = [-1] * n
            in_queue = [False] * n
            dist[source] = 0
            queue = deque([source])
            in_queue[source] = True

            while queue:
                v = queue.popleft()
                in_queue[v] = False
                for i, edge in enumerate(self.graph[v]):
                    if edge.capacity > 0 and dist[edge.to] > dist[v] + edge.cost:
                        dist[edge.to] = dist[v] + edge.cost
                        prev_node[edge.to] = v
                        prev_edge[edge.to] = i
                        if not in_queue[edge.to]:
                            queue.append(edge.to)
                            in_queue[edge.to] = True

            if dist[sink] == self.inf:
                raise ValueError(
                    f"Min-cost matching is infeasible: could only pair {total_flow} of "
                    f"{required_flow} copies."
                )

            # Augment by one unit so the recorded path always represents a single pair.
            v = sink
            while v != source:
                u = prev_node[v]
                edge = self.graph[u][prev_edge[v]]
                self.graph[u][prev_edge[v]] = edge._replace(capacity=edge.capacity - 1)
                reverse = self.graph[v][edge.rev]
                self.graph[v][edge.rev] = reverse._replace(capacity=reverse.capacity + 1)
                v = u

            total_flow += 1
            total_cost += dist[sink]

        return total_flow, total_cost


def minimum_change_assignment(
    costs: t.Sequence[t.Sequence[int]],
    secondary_weight: int = 10**6,
) -> t.Tuple[t.List[t.Tuple[int, int]], int]:
    """Pair left copies with right copies minimizing total cell changes.

    Args:
        costs: A matrix ``costs[i][j]`` giving the number of differing non-key cells between
            left copy ``i`` and right copy ``j``.
        secondary_weight: Encoding factor for the deterministic tie-breaker. It must exceed
            the maximum achievable sum of target ordinals (``n_right ** 2``) so that a single
            extra changed cell always outweighs every ordering preference.

    Returns:
        A tuple of ``(pairs, total_cost)`` where ``pairs`` is a list of
        ``(left_index, right_index)`` and ``total_cost`` is the exact minimum number of
        changed cells across the pairing. Unpaired copies on either side are simply absent
        from ``pairs`` and are reported as side-only by the caller.

    The stable tie-break rule: among all minimum-cost pairings, choose the one minimizing the
    sum of target ordinals; equivalently (each left node contributes the same constant), the
    lexicographically smallest assignment when left copies are read in their stable order.
    """
    n_left = len(costs)
    n_right = len(costs[0]) if n_left else 0

    if n_left == 0 or n_right == 0:
        return [], 0

    source = 0
    left_start = 1
    right_start = left_start + n_left
    sink = right_start + n_right
    network = MinCostMaxFlow(sink + 1)

    for i in range(n_left):
        network.add_edge(source, left_start + i, 1, 0)
    for j in range(n_right):
        network.add_edge(right_start + j, sink, 1, 0)

    for i, row in enumerate(costs):
        for j, cell_cost in enumerate(row):
            encoded_cost = cell_cost * secondary_weight + j
            network.add_edge(left_start + i, right_start + j, 1, encoded_cost)

    required_flow = min(n_left, n_right)
    _, encoded_total = network.flow(source, sink, required_flow)

    pairs: t.List[t.Tuple[int, int]] = []
    for i in range(n_left):
        left_node = left_start + i
        for edge in network.graph[left_node]:
            # Forward edges point at right nodes with zero initial residual when saturated.
            if right_start <= edge.to < sink and edge.capacity == 0:
                pairs.append((i, edge.to - right_start))
                break

    pairs.sort()
    total_cost = sum(costs[i][j] for i, j in pairs)
    # Sanity check that the encoding never distorted the real optimum.
    assert encoded_total // secondary_weight == total_cost
    return pairs, total_cost
