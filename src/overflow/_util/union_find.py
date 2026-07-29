from numba.experimental import jitclass
from numba.typed import Dict  # type: ignore[attr-defined]
from numba.types import DictType, int64


@jitclass(
    [
        ("parent", DictType(int64, int64)),
        ("size", DictType(int64, int64)),
    ]
)
class UnionFind:
    """
    A disjoint-set (union-find) structure over int64 keys.

    Keys are added lazily, so the structure does not need to know the universe of
    labels up front. This matters for tiled processing, where labels are handed out
    per tile from a sparse global range and only the ones that actually appear on a
    tile perimeter ever need to be joined.

    Uses union by size and iterative path compression, giving near constant time
    amortized find and union. Path compression is written as an explicit two pass
    loop because numba jitclass methods cannot recurse.

    Attributes:
        parent (DictType(int64, int64)): Maps each key to its parent key. A key that
            maps to itself is the representative (root) of its set.
        size (DictType(int64, int64)): Maps each root key to the number of keys in
            its set. Entries for non root keys are stale and must not be read.
    """

    def __init__(self):
        self.parent = Dict.empty(key_type=int64, value_type=int64)
        self.size = Dict.empty(key_type=int64, value_type=int64)

    def add(self, x: int64) -> None:
        """Add a key as a singleton set if it is not already present.

        Args:
            x (int64): The key to add.
        """
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = int64(1)

    def find(self, x: int64) -> int64:
        """Return the representative key of the set containing x.

        Adds x as a singleton if it is not already present. Compresses the path
        from x to the root so subsequent lookups are direct.

        Args:
            x (int64): The key to look up.

        Returns:
            int64: The representative key of x's set.
        """
        self.add(x)
        # walk up to the root
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # second pass: point every key on the path directly at the root
        current = x
        while self.parent[current] != root:
            next_key = self.parent[current]
            self.parent[current] = root
            current = next_key
        return root

    def union(self, a: int64, b: int64) -> None:
        """Merge the sets containing a and b.

        Args:
            a (int64): A key in the first set.
            b (int64): A key in the second set.
        """
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        # attach the smaller tree under the larger one to keep paths short
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]
