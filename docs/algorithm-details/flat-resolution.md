# Flat Resolution Algorithm

## Overview

The flat resolution algorithm assigns flow directions to cells within flat regions (areas of constant elevation) where the standard D8 steepest-descent method cannot determine flow. It creates a synthetic gradient by combining two distance-based surfaces.

## References

Barnes, R., Lehman, C., Mulla, D. (2014). "An Efficient Assignment of Drainage Direction Over Flat Surfaces in Raster Digital Elevation Models." *Computers & Geosciences*, 62, 128-135. [PDF](https://rbarnes.org/sci/2014_flats.pdf)

Zhou, G., Song, L., Liu, Y. (2021). "Parallel Assignment of Flow Directions Over Flat Surfaces in Massive Digital Elevation Models." *Computers & Geosciences*, 159, 105015. [DOI](https://doi.org/10.1016/j.cageo.2021.105015)

## Core Concept

The algorithm constructs a **flat mask** that encodes directional preference without modifying the original DEM. This mask is the superposition of two gradients:

1. **Away from higher terrain** ($G_{\text{high}}$): Values increase with distance from cells adjacent to higher ground
2. **Toward lower terrain** ($G_{\text{low}}$): Values increase with distance from cells adjacent to lower ground (drainage outlets)

The combination directs flow away from ridges and toward natural drainage points.

## Definitions

**Flat region**: A connected set of cells $F$ where:
$$
\forall c_1, c_2 \in F: z_{c_1} = z_{c_2} \text{ and } c_1 \sim c_2 \text{ (connected)}
$$

**High edge**: Cells in $F$ adjacent to higher terrain
$$
H = \{c \in F : \exists n \in N(c), n \notin F, z_n > z_c\}
$$

**Low edge**: Cells in $F$ adjacent to lower terrain or nodata
$$
L = \{c \in F : \exists n \in N(c), z_n < z_c \text{ or } n \text{ is nodata}\}
$$

## Algorithm Overview

The algorithm proceeds in four phases:

1. **FlatEdges**: Identify high-edge and low-edge cells
2. **AwayFromHigher**: Propagate gradient from high edges
3. **TowardLower**: Propagate gradient from low edges and combine
4. **AssignDirections**: Use flat mask to determine D8 directions

## Phase 1: Identify Flat Edges

Scan all cells with undefined flow direction (code 8) and classify as high-edge or low-edge:

```python
def flat_edges(dem, fdr):
    high_edges = []
    low_edges = []

    for cell in all_cells():
        if fdr[cell] != UNDEFINED:
            continue

        for neighbor in neighbors_8(cell):
            if dem[neighbor] > dem[cell]:
                high_edges.append(cell)
                break
            if dem[neighbor] < dem[cell] or is_nodata(neighbor):
                low_edges.append(cell)
                break

    return high_edges, low_edges
```

## Phase 2: Gradient Away from Higher Terrain

Use breadth-first search from high-edge cells. Each cell records its distance (in "loops" or BFS iterations) from the nearest high edge:

```python
def away_from_higher(flat_mask, high_edges, dem, fdr):
    queue = Queue()
    loops = 1

    for cell in high_edges:
        flat_mask[cell] = 1
        queue.push(cell)

    while not queue.empty():
        loops += 1
        size = queue.size()

        for _ in range(size):
            current = queue.pop()

            for neighbor in neighbors_8(current):
                if fdr[neighbor] != UNDEFINED:
                    continue
                if dem[neighbor] != dem[current]:
                    continue
                if flat_mask[neighbor] != 0:
                    continue

                flat_mask[neighbor] = loops
                queue.push(neighbor)
```

After this phase:
$$
G_{\text{high}}(c) = \text{BFS distance from } c \text{ to nearest high edge}
$$

## Phase 3: Gradient Toward Lower Terrain

Propagate from low-edge cells and combine with the existing gradient. The combination formula ensures cells preferentially flow toward drainage points while avoiding high terrain:

```python
def toward_lower(flat_mask, low_edges, dem, fdr):
    # Negate existing values
    for cell in all_cells():
        if flat_mask[cell] > 0:
            flat_mask[cell] = -flat_mask[cell]

    queue = Queue()
    loops = 1

    for cell in low_edges:
        if flat_mask[cell] < 0:
            # Already visited by away_from_higher
            flat_mask[cell] += MAX_FLAT_HEIGHT + 2 * loops
        else:
            flat_mask[cell] = 2 * loops
        queue.push(cell)

    while not queue.empty():
        loops += 1
        size = queue.size()

        for _ in range(size):
            current = queue.pop()

            for neighbor in neighbors_8(current):
                if fdr[neighbor] != UNDEFINED:
                    continue
                if dem[neighbor] != dem[current]:
                    continue

                if flat_mask[neighbor] < 0:
                    # Combine gradients
                    flat_mask[neighbor] += MAX_FLAT_HEIGHT + 2 * loops
                elif flat_mask[neighbor] == 0:
                    flat_mask[neighbor] = 2 * loops

                queue.push(neighbor)
```

### Gradient Combination

The final flat mask value for cell $c$ combines both gradients:

$$
M(c) = G_{\text{low}}(c) \cdot 2 + G_{\text{high}}(c) + K
$$

Where:
- $G_{\text{low}}(c)$ = BFS distance to nearest low edge
- $G_{\text{high}}(c)$ = BFS distance to nearest high edge
- $K$ = constant offset to ensure positive values

The factor of 2 on $G_{\text{low}}$ gives priority to draining toward outlets over moving away from ridges.

## Phase 4: Assign Flow Directions

Use the flat mask as a synthetic elevation surface. Flow direction is assigned toward the neighbor with the minimum mask value:

```python
def assign_flat_directions(flat_mask, fdr, dem):
    for cell in all_cells():
        if fdr[cell] != UNDEFINED:
            continue

        min_value = infinity
        min_direction = -1

        for direction in range(8):
            neighbor = get_neighbor(cell, direction)

            if dem[neighbor] != dem[cell]:
                if dem[neighbor] < dem[cell] or is_nodata(neighbor):
                    # Direct drainage - use immediately
                    min_direction = direction
                    break
                continue

            distance = sqrt(2) if is_diagonal(direction) else 1.0
            slope = (flat_mask[cell] - flat_mask[neighbor]) / distance

            if slope > 0 and flat_mask[neighbor] < min_value:
                min_value = flat_mask[neighbor]
                min_direction = direction

        if min_direction >= 0:
            fdr[cell] = min_direction
```

## Tiled Processing

For large DEMs, the algorithm uses graph-based connectivity to solve flat regions spanning multiple tiles. The tiled implementation is based on Zhou et al. (2021), which extends the Barnes et al. (2014) algorithm to support parallel processing of massive DEMs.

### Local Graph Construction

For each tile:
1. Identify flat cells on tile perimeter
2. Compute distances between perimeter cells via BFS
3. Record connections to high/low terrain

### Global Graph Solving

1. Join adjacent tiles by connecting matching perimeter cells
2. Solve shortest paths from all perimeter cells to high/low terrain using Dijkstra's algorithm
3. Use precomputed distances when creating flat masks

### Distance Computation

For two perimeter cells $p_1$ and $p_2$ in the same flat region within a tile:

$$
d(p_1, p_2) = \text{BFS distance through flat cells}
$$

If no path exists through flat cells, an estimate using Chebyshev distance is used:

$$
d_{\text{Chebyshev}}(p_1, p_2) = \max(|r_1 - r_2|, |c_1 - c_2|)
$$

This is valid when the path can traverse the flat region diagonally.

## Properties

The resolved flat regions satisfy:

1. **Complete resolution**: Every undefined cell receives a valid direction (0-7)
2. **Consistency**: Flow paths within flats do not form cycles
3. **Natural drainage**: Flow preferentially moves toward low edges

## Complexity

Let $n$ be the number of cells and $f$ the number of flat cells.

| Operation | Time Complexity |
|-----------|-----------------|
| Edge identification | $O(f)$ |
| BFS propagation | $O(f)$ |
| Direction assignment | $O(f)$ |
| Total | $O(f)$ |

For tiled processing with graph solving:

| Operation | Time Complexity |
|-----------|-----------------|
| Local BFS | $O(f_t)$ per tile |
| Global Dijkstra | $O(p \log p)$ |

Where $f_t$ is flat cells per tile and $p$ is total perimeter cells.

## Performance Considerations

Tiled flat resolution is the most computationally intensive process in Overflow due to:

- Potentially large flat regions spanning many tiles
- Graph construction overhead
- Global shortest-path computation

For DEMs with large flat areas (lakes, filled depressions), smaller tile sizes (e.g., 512) are recommended.

## See Also

- [Flow Direction](flow-direction.md) - Initial flow direction computation
- [Fill](fill.md) - Depression filling creates flat regions requiring resolution
