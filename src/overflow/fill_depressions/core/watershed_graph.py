from numba import int64, float32
from numba.experimental import jitclass
from numba.typed import Dict  # pylint: disable=no-name-in-module
from numba.types import DictType
from overflow.util.numba_types import DictInt64Float32


@jitclass([("_edges", DictType(int64, DictType(int64, float32)))])
class WatershedGraph:
    """
    A class representing a graph of watershed connections between tiles.

    Attributes:
    - _edges (DictType(int64, DictType(int64, float32))): A dictionary representing the graph of watershed connections.

    Methods:
    - __init__(self): Initializes the graph with an empty dictionary.
    - __contains__(self, edge: tuple[int, int]) -> bool: Checks if an edge is in the graph.
    - __getitem__(self, edge: tuple[int, int]) -> float: Gets the weight of an edge in the graph.
    - __setitem__(self, edge: tuple[int, int], weight: float): Sets the weight of an edge in the graph.
    - update(self, other: WatershedGraph): Updates the graph with the edges from another graph.
    - neighbors(self, node: int): Gets the neighbors of a node in the graph.
    """

    _edges: dict

    def __init__(self):
        self._edges = Dict.empty(key_type=int64, value_type=DictInt64Float32)

    def __contains__(self, edge: tuple[int, int]) -> bool:
        u, v = edge
        return u in self._edges and v in self._edges[u]

    def __getitem__(self, edge: tuple[int, int]) -> float:
        u, v = edge
        return self._edges[u][v]

    def __setitem__(self, edge: tuple[int, int], weight: float):
        # adds both edges u -> v and v -> u since the graph is undirected
        u, v = edge
        if u not in self._edges:
            self._edges[u] = Dict.empty(int64, float32)
        self._edges[u][v] = weight
        if v not in self._edges:
            self._edges[v] = Dict.empty(int64, float32)
        self._edges[v][u] = weight

    def update(self, other: "WatershedGraph"):
        """
        Updates the graph with the edges from another graph.
        """
        # pylint: disable=protected-access
        for u, neighbors in other._edges.items():
            for v, weight in neighbors.items():
                self[u, v] = weight

    def neighbors(self, node: int):
        """
        Gets the neighbors of a node in the graph.
        """
        return self._edges[node]
