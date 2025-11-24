# Basin Delineation Algorithm

## Overview

Basin delineation assigns each cell a label identifying its drainage outlet. The algorithm traces flow paths upstream from outlets, creating watershed boundaries that define the contributing area for each drainage point.

## Core Concept

The algorithm uses **upstream breadth-first search (BFS)** from outlet cells. Starting at each outlet, it propagates upstream through the flow direction network, assigning basin labels to all cells that drain to that outlet.

## Definitions

**Outlet cell**: A cell where flow exits the domain or terminates:

$$
\text{is\_outlet}(c) = \text{downstream}(c) \text{ is nodata or out of bounds}
$$

**Drainage point**: A user-specified outlet location with an associated basin ID.

**Upstream neighbor**: A cell $n$ is upstream of $c$ if $n$ flows directly to $c$:

$$
\text{upstream}(n, c) \iff \text{downstream}(n) = c
$$

## Data Structures

| Structure | Description |
|-----------|-------------|
| Labels Array | Integer grid storing basin ID for each cell |
| Drainage Points | Map from (row, col) to basin ID |
| Basin Graph | Directed graph of basin connectivity |
| Queue | FIFO queue for BFS traversal |

## Algorithm

### Phase 1: Outlet Identification

Scan all cells to find outlets:

```python
def find_outlets(fdr):
    outlets = []

    for cell in all_cells():
        if is_nodata(fdr[cell]):
            continue

        downstream = get_downstream(cell, fdr)

        if downstream is None or is_nodata(fdr[downstream]):
            outlets.append(cell)

    return outlets
```

### Phase 2: Upstream Labeling

For each outlet, propagate labels upstream using BFS:

```python
def label_watersheds(fdr, drainage_points):
    labels = zeros(fdr.shape, dtype=int64)
    basin_graph = {}
    queue = Queue()

    # Start from all outlets
    for outlet in find_outlets(fdr):
        basin_id = generate_unique_id(outlet)
        labels[outlet] = basin_id
        queue.push((outlet, basin_id))

    while not queue.empty():
        cell, current_basin = queue.pop()

        # Check if this cell is a drainage point
        if cell in drainage_points:
            new_basin = drainage_points[cell]
            basin_graph[new_basin] = current_basin  # Record downstream connection
            current_basin = new_basin

        # Find and label upstream neighbors
        for neighbor in neighbors_8(cell):
            if labels[neighbor] != 0:
                continue  # Already labeled

            if not is_upstream(neighbor, cell, fdr):
                continue  # Not flowing to current cell

            labels[neighbor] = current_basin
            queue.push((neighbor, current_basin))

    return labels, basin_graph
```

### Upstream Neighbor Generator

Iterate through neighbors and check if they flow to the current cell:

```python
def upstream_neighbors(cell, fdr):
    row, col = cell

    for direction in range(8):
        dr, dc = OFFSETS[direction]
        neighbor = (row + dr, col + dc)

        if out_of_bounds(neighbor):
            continue

        # Check if neighbor's flow direction points to cell
        neighbor_direction = fdr[neighbor]
        if neighbor_direction == NODATA or neighbor_direction == UNDEFINED:
            continue

        # Opposite direction should point back to cell
        opposite = (direction + 4) % 8
        if neighbor_direction == opposite:
            yield neighbor
```

## Drainage Point Handling

### Coordinate Transformation

Drainage points are typically provided in geographic coordinates. Convert to raster indices:

$$
c = \lfloor (x - x_0) / \Delta x \rfloor
$$

$$
r = \lfloor (y_0 - y) / |\Delta y| \rfloor
$$

where $(x_0, y_0)$ is the raster origin and $(\Delta x, \Delta y)$ are pixel dimensions.

### Snapping to Flow Accumulation

Drainage points may not fall exactly on stream channels. Snapping moves them to the highest accumulation cell within a search radius:

```python
def snap_to_stream(point, fac, radius):
    best_cell = point
    max_acc = fac[point]

    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            cell = (point[0] + dr, point[1] + dc)

            if fac[cell] > max_acc:
                max_acc = fac[cell]
                best_cell = cell

    return best_cell
```

### Basin ID Assignment

Each drainage point receives a unique basin ID. The graph records downstream relationships:

$$
\text{graph}[B_i] = B_j \iff \text{basin } B_i \text{ drains to basin } B_j
$$

## Basin Graph

The basin graph is a directed acyclic graph (DAG) representing drainage relationships:

- **Vertices**: Basin IDs
- **Edges**: Downstream connections

For drainage points $D = \{d_1, d_2, \ldots, d_k\}$ with basin IDs $B = \{b_1, b_2, \ldots, b_k\}$:

$$
(b_i, b_j) \in E \iff \exists \text{ flow path from } d_i \text{ to } d_j
$$

## Polygon Generation

Convert labeled raster to vector polygons using boundary tracing.

### Boundary Cell Detection

A cell is on the basin boundary if any 8-connected neighbor has a different label:

$$
\text{is\_boundary}(c) = \exists n \in N_8(c) : \text{label}(n) \neq \text{label}(c)
$$

### Moore-Neighbor Tracing

Trace the boundary of each basin using the Moore-Neighbor algorithm:

1. Find a boundary cell (leftmost cell in topmost row)
2. Start facing "up" (north)
3. Turn clockwise until finding a cell with the same label
4. Move to that cell, record its position
5. Repeat until returning to the starting cell

```python
def trace_boundary(start, labels, basin_id):
    polygon = [start]
    current = start
    direction = 0  # Start facing up

    while True:
        # Search clockwise for next boundary cell
        for i in range(8):
            check_dir = (direction + 5 + i) % 8  # Start from back-left
            neighbor = get_neighbor(current, check_dir)

            if labels[neighbor] == basin_id:
                current = neighbor
                direction = check_dir
                break

        if current == start and len(polygon) > 1:
            break

        polygon.append(current)

    return polygon
```

Reference: [Moore-Neighbor Tracing](https://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/square.html)

## Tiled Processing

For large rasters, basins are labeled per-tile and then connected:

### Phase 1: Local Labeling

Each tile runs the watershed labeling algorithm independently:

- Labels offset by tile index to ensure uniqueness
- Perimeter cells recorded for cross-tile connection

### Phase 2: Cross-Tile Connection

Examine adjacent tile boundaries:

For tiles $A$ and $B$ sharing an edge, for each pair of adjacent cells $(a, b)$:

```python
if fdr[a] points to b:
    graph[labels[a]] = labels[b]
elif fdr[b] points to a:
    graph[labels[b]] = labels[a]
```

### Phase 3: Label Finalization

Walk the graph to assign final basin IDs:

```python
def finalize_labels(labels, graph, drainage_points):
    # Find final basin for each intermediate label
    final_basin = {}

    for label in all_labels:
        current = label

        while current in graph and current not in drainage_points:
            current = graph[current]

        final_basin[label] = current

    # Apply final labels
    for cell in all_cells():
        if labels[cell] in final_basin:
            labels[cell] = final_basin[labels[cell]]
```

## Output Modes

### Drainage Point Basins Only

Label only cells draining to specified drainage points. Other cells receive nodata.

### All Basins

Label all cells, including those draining off-map:

$$
\text{label}(c) = \begin{cases}
b_i & \text{if } c \text{ drains to drainage point } d_i \\
\text{outlet\_id}(c) & \text{if } c \text{ drains off-map}
\end{cases}
$$

## Complexity

Let $n$ be total cells and $d$ be drainage point count.

| Operation | Time Complexity |
|-----------|-----------------|
| Outlet finding | $O(n)$ |
| Upstream BFS | $O(n)$ |
| Graph walking | $O(d)$ |
| Polygon tracing | $O(p)$ |

Where $p$ is total boundary perimeter.

## See Also

- [Flow Direction](flow-direction.md) - Computing flow directions
- [Stream Extraction](stream-extraction.md) - Generating drainage points at junctions
- [Flow Length](flow-length.md) - Computing flow path lengths within basins
