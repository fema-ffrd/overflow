# Breach Algorithm

## Overview

The breach algorithm removes depressions from a DEM by carving flow paths through elevation barriers. Overflow implements two complementary breaching strategies:

1. **Single-Cell Pit Breaching** - Fast local resolution for simple pits
2. **Least-Cost Path Breaching** - Dijkstra-based pathfinding for complex depressions

## Single-Cell Pit Breaching

### Purpose

Identifies and resolves pits that can be solved by modifying a single intermediate cell. This serves as an efficient preprocessing step before the more computationally expensive least-cost path algorithm.

### Pit Identification

For each cell $c$ with elevation $z_c$, examine all 8-connected neighbors $N(c)$:

$$
\text{is\_sink}(c) = \begin{cases}
\text{true} & \text{if } \forall n \in N(c): z_n \geq z_c \text{ or } n \text{ is nodata} \\
\text{false} & \text{otherwise}
\end{cases}
$$

$$
\text{is\_flat}(c) = \begin{cases}
\text{true} & \text{if } \forall n \in N(c): z_n = z_c \\
\text{false} & \text{otherwise}
\end{cases}
$$

A cell is marked as a pit if $\text{is\_sink}(c) \land \neg\text{is\_flat}(c)$.

### Local Resolution

For each identified pit, search a 16-cell neighborhood at radius 2. If any cell $t$ in this neighborhood satisfies $z_t \leq z_c$ or is nodata:

1. Identify the intermediate cell $m$ between $c$ and $t$
2. Set the intermediate elevation:

$$
z_m = \frac{z_c + z_t}{2}
$$

When breaching toward nodata, use an epsilon gradient:

$$
z_t = z_c - 2\epsilon
$$

where $\epsilon$ is a small gradient constant.

## Least-Cost Path Breaching

### Purpose

Finds optimal breach paths for complex, multi-cell pits using Dijkstra's shortest path algorithm with an elevation-based cost function.

### Data Structures

| Structure | Description |
|-----------|-------------|
| Priority Queue | Min-heap storing (cost, row, col) tuples |
| Cost Grid | 2D array of size $(2r+1) \times (2r+1)$ storing accumulated path costs |
| Parent Pointers | Two 2D arrays storing the predecessor cell for path reconstruction |

Where $r$ is the search radius parameter.

### Cost Function

The cost to move from cell $c$ to neighbor $n$ is defined as:

$$
\text{cost}(c \to n) = w_{c,n} \cdot (z_n - z_{\text{pit}})
$$

Where:

- $z_{\text{pit}}$ is the elevation of the original pit cell
- $w_{c,n}$ is a distance weight:

$$
w_{c,n} = \begin{cases}
1 & \text{cardinal directions} \\
\sqrt{2} & \text{diagonal directions}
\end{cases}
$$

The total path cost is the sum of edge costs:

$$
\text{Cost}(P) = \sum_{i=1}^{|P|-1} \text{cost}(p_i \to p_{i+1})
$$

**Properties:**

- Negative edge costs are permitted (descending terrain reduces cost)
- Paths through valleys are preferred over paths over ridges
- Cost is relative to pit elevation, creating a pit-centric coordinate system

### Algorithm

```python
def breach_pit(dem, pit_row, pit_col, search_radius):
    initial_elevation = dem[pit_row, pit_col]

    # Initialize cost grid to infinity
    costs = array((2*r+1, 2*r+1), fill=infinity)
    costs[pit] = 0

    # Initialize parent pointers
    parent = array((2*r+1, 2*r+1, 2), fill=UNVISITED)

    # Priority queue: (cost, row, col)
    queue = MinHeap()
    queue.push((0, pit_row, pit_col))

    while not queue.empty():
        cost, row, col = queue.pop()

        # Termination: found breach point
        if dem[row, col] < initial_elevation or is_nodata(row, col):
            return reconstruct_path(row, col, parent)

        for neighbor in neighbors_8(row, col):
            if out_of_bounds(neighbor):
                continue

            weight = sqrt(2) if is_diagonal(neighbor) else 1.0
            edge_cost = weight * (dem[neighbor] - initial_elevation)
            new_cost = cost + edge_cost

            if new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                parent[neighbor] = (row, col)
                queue.push((new_cost, neighbor))

    return None  # No path found within search radius
```

### Path Reconstruction

Reconstruct the path by following parent pointers from the breach point back to the pit:

```python
def reconstruct_path(breach_row, breach_col, parent):
    path = []
    current = (breach_row, breach_col)

    while current != UNVISITED:
        path.append(current)
        current = parent[current]

    return path  # Ordered from breach point to pit
```

### Gradient Application

Apply a linear elevation gradient along the breach path. For a path $P = [p_1, p_2, \ldots, p_k]$ where $p_1$ is the breach point and $p_k$ is the pit:

**Case A: Breaching to valid terrain**

$$
z_{p_i} = \min\left(z_{p_i}, \; z_{p_1} + \frac{i-1}{k-1}(z_{p_k} - z_{p_1})\right) \quad \text{for } i = 1, \ldots, k-1
$$

**Case B: Breaching to nodata**

$$
z_{p_i} = \min\left(z_{p_i}, \; z_{p_k} - (k-i) \cdot \epsilon\right) \quad \text{for } i = 2, \ldots, k
$$

## Tiled Processing

For large DEMs, processing is divided into overlapping tiles:

1. **Chunking**: DEM split into tiles of size $s \times s$ (default $s = 512$)
2. **Buffering**: Each tile includes a buffer of $r$ pixels (search radius) on all edges
3. **Parallel Execution**: Tiles processed concurrently
4. **Result Writing**: Only the unbuffered core region written to output

The buffer ensures that breach paths near tile edges have full search neighborhoods available.

## Complexity

Let $n$ be the total number of cells, $p$ the number of pits, and $r$ the search radius.

| Operation | Time Complexity |
|-----------|-----------------|
| Single-cell pit detection | $O(n)$ |
| Single-cell resolution | $O(p)$ |
| Least-cost path (per pit) | $O(r^2 \log r^2)$ |
| Total | $O(n + p \cdot r^2 \log r)$ |

## Implementation Notes

- **Non-deterministic**: Pits are processed in parallel without elevation ordering, which may produce different (but valid) results between runs
- **Requires post-processing**: Run the fill algorithm after breaching to resolve any remaining depressions
- **Memory efficient**: Cost arrays are allocated per-tile rather than globally
