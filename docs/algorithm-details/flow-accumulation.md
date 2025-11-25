# Flow Accumulation Algorithm

## Overview

The flow accumulation algorithm computes the upstream drainage area for each cell. The output value represents the count of cells whose flow paths traverse the cell, providing the basis for stream network extraction and watershed analysis.

## Reference

Barnes, R. (2017). "Parallel Non-divergent Flow Accumulation For Trillion Cell Digital Elevation Models On Desktops Or Clusters." *Environmental Modelling & Software*, 92, 202-212. [arXiv:1608.04431](https://arxiv.org/abs/1608.04431)

## Core Concept

Flow accumulation uses **topological sorting** to process cells in dependency order. A cell can only be processed after all cells that flow into it have been processed. This ensures correct accumulation without iteration.

## Data Structures

| Structure | Description |
|-----------|-------------|
| Inflow Count | Integer grid counting upstream neighbors for each cell |
| Accumulation | Integer grid storing cumulative flow for each cell |
| Queue | FIFO queue for cells ready to process (inflow count = 0) |
| Links | For tiled processing: where perimeter cells drain to |

## Algorithm

### Dependency Calculation

For each cell, count how many neighbors flow into it:

$$
\text{inflow}(c) = |\{n \in N(c) : \text{downstream}(n) = c\}|
$$

where $\text{downstream}(n)$ is the cell that $n$ flows to according to the flow direction raster.

### Topological Sort Processing

```python
def flow_accumulation(fdr):
    rows, cols = fdr.shape
    accumulation = zeros((rows, cols), dtype=int64)
    inflow_count = zeros((rows, cols), dtype=uint8)
    queue = Queue()

    # Phase 1: Calculate inflow counts
    for cell in all_cells():
        downstream = get_downstream(cell, fdr)
        if downstream is not None:
            inflow_count[downstream] += 1

    # Phase 2: Initialize queue with source cells
    for cell in all_cells():
        if inflow_count[cell] == 0:
            queue.push(cell)

    # Phase 3: Process in topological order
    while not queue.empty():
        cell = queue.pop()

        # Each cell contributes itself
        accumulation[cell] += 1

        # Pass accumulation to downstream cell
        downstream = get_downstream(cell, fdr)
        if downstream is None:
            continue

        accumulation[downstream] += accumulation[cell]
        inflow_count[downstream] -= 1

        # If downstream has no more pending inflows, it's ready
        if inflow_count[downstream] == 0:
            queue.push(downstream)

    return accumulation
```

### Downstream Cell Lookup

Given a flow direction code $d \in \{0, 1, \ldots, 7\}$:

$$
\text{downstream}(r, c) = (r + \Delta r_d, c + \Delta c_d)
$$

Using the offset table from [Flow Direction](flow-direction.md).

## Properties

The output accumulation raster satisfies:

1. **Minimum value**: Every cell has accumulation $\geq 1$ (counts itself)
2. **Conservation**: Total inflow equals total outflow for interior cells
3. **Monotonic increase**: Accumulation increases along flow paths

$$
\text{acc}(c) = 1 + \sum_{n : \text{downstream}(n) = c} \text{acc}(n)
$$

## Tiled Processing

For large rasters, the algorithm uses a three-phase approach:

### Phase 1: Local Accumulation

Process each tile independently:

1. Run single-tile flow accumulation
2. Build **links** for perimeter cells: where they ultimately drain within the tile

```python
def follow_path(cell, fdr, tile_bounds):
    """Follow flow from perimeter cell to determine destination."""
    current = cell

    while True:
        downstream = get_downstream(current, fdr)

        if downstream is None:
            return FLOW_TERMINATES

        if out_of_tile(downstream, tile_bounds):
            return FLOW_EXTERNAL

        if on_perimeter(downstream, tile_bounds):
            return downstream  # Exits via another perimeter cell

        current = downstream
```

### Phase 2: Global Graph Construction

Build a graph connecting tiles via their perimeter cells:

**Vertices**: Perimeter cells from all tiles
**Edges**: Flow connections between perimeter cells

For each perimeter cell $p$:
- If $p$ flows external (leaves tile): create edge to corresponding cell in adjacent tile
- If $p$ flows to another perimeter cell $p'$: create internal edge

The global graph is then processed using the same topological sort:

```python
def solve_global_graph(tiles, links):
    global_acc = {}
    global_offset = {}
    inflow_count = {}
    queue = Queue()

    # Build dependency counts
    for tile in tiles:
        for perimeter_cell in tile.perimeter:
            link = links[perimeter_cell]
            if link != FLOW_TERMINATES:
                inflow_count[link] += 1

    # Initialize queue with sources
    for cell in all_perimeter_cells():
        if inflow_count[cell] == 0:
            queue.push(cell)

    # Topological processing
    while not queue.empty():
        cell = queue.pop()
        # ... accumulate and propagate
```

### Phase 3: Finalization

Apply global offsets to each tile:

For perimeter cell $p$ with global offset $\delta_p$, follow the flow path within the tile and add $\delta_p$ to all cells along the path:

```python
def finalize_tile(tile, global_offset, links):
    for perimeter_cell in tile.perimeter:
        if perimeter_cell in global_offset:
            offset = global_offset[perimeter_cell]
            current = perimeter_cell

            while current is not None:
                tile.accumulation[current] += offset
                downstream = get_downstream(current)

                # Stop if we hit another global node
                if downstream in global_acc:
                    break

                current = downstream
```

## Perimeter Structure

Tile perimeters are stored as 1D arrays in clockwise order:

```
Top edge (left to right)
  -> Right edge (top to bottom)
    -> Bottom edge (right to left)
      -> Left edge (bottom to top)
```

Total perimeter size: $2(w + h) - 4$ cells for a $w \times h$ tile.

## Global Coordinate Mapping

For a tile at grid position $(t_r, t_c)$ with chunk size $s$:

$$
\text{global\_index}(r, c) = (t_r \cdot s + r) \cdot (n_c \cdot s) + (t_c \cdot s + c)
$$

This provides a unique identifier for inter-tile graph construction.

## Complexity

Let $n$ be the total number of cells.

**Single-tile processing:**

| Operation | Time | Space |
|-----------|------|-------|
| Inflow counting | $O(n)$ | $O(n)$ |
| Queue processing | $O(n)$ | $O(n)$ |
| Total | $O(n)$ | $O(n)$ |

**Tiled processing** with $t$ tiles of size $s \times s$:

| Operation | Time | Space |
|-----------|------|-------|
| Local accumulation | $O(s^2)$ per tile | $O(s^2)$ |
| Graph construction | $O(p)$ | $O(p)$ |
| Graph solving | $O(p)$ | $O(p)$ |
| Finalization | $O(s^2)$ per tile | $O(1)$ |

Where $p = O(t \cdot s)$ is the total perimeter cells.

## Input Requirements

The flow direction raster must satisfy:

1. **No undefined cells**: All non-nodata cells have direction 0-7
2. **Acyclic**: No cycles in the flow graph

These requirements are automatically met when using `flow_direction()` with `resolve_flats=True` on a properly conditioned (filled) DEM.

## See Also

- [Flow Direction](flow-direction.md) - Computing flow directions
- [Stream Extraction](stream-extraction.md) - Using accumulation to extract streams
- [Basin Delineation](basin-delineation.md) - Watershed analysis
