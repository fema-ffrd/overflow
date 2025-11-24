# Flow Length Algorithm

## Overview

The flow length algorithm computes the longest upstream flow path distance for each cell. This metric is used in hydrological modeling to estimate time of concentration, characterize watershed shape, and identify the hydraulically most distant point.

## Core Concept

The algorithm uses **multi-basin upstream BFS** to propagate distance values from drainage points (outlets) through the flow network. Each cell records the maximum distance from the outlet along any upstream path.

## Definitions

**Flow length**: The distance along the flow path from a cell to its basin outlet:

$$
L(c) = \sum_{i=0}^{k-1} d(p_i, p_{i+1})
$$

where $P = [p_0 = c, p_1, \ldots, p_k = \text{outlet}]$ is the flow path and $d(p_i, p_{i+1})$ is the step distance.

**Longest flow path**: The path from outlet to the hydraulically most distant cell in the basin.

## Distance Calculation

### Projected Coordinates

For rasters in a projected coordinate system (e.g., UTM), use Euclidean distance:

$$
d(c_1, c_2) = \sqrt{(\Delta c \cdot \delta_x)^2 + (\Delta r \cdot \delta_y)^2}
$$

where $\delta_x$ and $\delta_y$ are pixel dimensions in map units.

For D8 flow (single-cell steps):

$$
d_{\text{step}} = \begin{cases}
\delta_x & \text{horizontal step} \\
\delta_y & \text{vertical step} \\
\sqrt{\delta_x^2 + \delta_y^2} & \text{diagonal step}
\end{cases}
$$

### Geographic Coordinates

For rasters in geographic coordinates (latitude/longitude), use the Haversine formula:

$$
d(c_1, c_2) = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)
$$

where:
- $R$ = Earth's radius (approximately 6,378,137 m)
- $\phi_1, \phi_2$ = latitudes in radians
- $\lambda_1, \lambda_2$ = longitudes in radians
- $\Delta\phi = \phi_2 - \phi_1$
- $\Delta\lambda = \lambda_2 - \lambda_1$

### Coordinate Conversion

Convert cell indices to geographic coordinates:

$$
\lambda = \lambda_0 + (c + 0.5) \cdot \Delta\lambda
$$

$$
\phi = \phi_0 + (r + 0.5) \cdot \Delta\phi
$$

where $(\lambda_0, \phi_0)$ is the origin and $(\Delta\lambda, \Delta\phi)$ are pixel dimensions in degrees.

## Algorithm

### Phase 1: Initialize Drainage Points

Pre-claim drainage point cells to prevent race conditions in parallel processing:

```python
def initialize_drainage_points(drainage_points, flow_length, basin_labels):
    for (row, col), basin_id in drainage_points.items():
        flow_length[row, col] = 0.0
        basin_labels[row, col] = basin_id
```

### Phase 2: Upstream BFS

For each drainage point, propagate upstream using BFS:

```python
def compute_flow_length(fdr, drainage_points, is_geographic):
    flow_length = full(fdr.shape, -1.0)  # -1 = unvisited
    basin_labels = zeros(fdr.shape, dtype=int64)
    max_cells = {}  # basin_id -> (row, col, max_distance)

    # Initialize
    initialize_drainage_points(drainage_points, flow_length, basin_labels)

    # Process each drainage point
    for (start_row, start_col), basin_id in drainage_points.items():
        queue = Queue()
        queue.push((start_row, start_col))
        max_dist = 0.0
        max_row, max_col = start_row, start_col

        while not queue.empty():
            row, col = queue.pop()
            current_dist = flow_length[row, col]

            for neighbor in upstream_neighbors((row, col), fdr):
                step_dist = calculate_distance(
                    (row, col), neighbor, is_geographic
                )
                new_dist = current_dist + step_dist

                if basin_labels[neighbor] == 0:
                    # Unclaimed cell - claim it
                    basin_labels[neighbor] = basin_id
                    flow_length[neighbor] = new_dist
                    queue.push(neighbor)

                    if new_dist > max_dist:
                        max_dist = new_dist
                        max_row, max_col = neighbor

                elif basin_labels[neighbor] == basin_id:
                    # Same basin - update if longer path found
                    if new_dist > flow_length[neighbor]:
                        flow_length[neighbor] = new_dist
                        queue.push(neighbor)

                        if new_dist > max_dist:
                            max_dist = new_dist
                            max_row, max_col = neighbor

        max_cells[basin_id] = (max_row, max_col, max_dist)

    return flow_length, basin_labels, max_cells
```

## Longest Flow Path Extraction

### Basin Graph Construction

Build a graph of upstream basin relationships:

```python
def build_basin_graph(drainage_points, fdr, basin_labels):
    graph = {}  # basin_id -> [upstream_basin_ids]

    for (row, col), basin_id in drainage_points.items():
        downstream = get_downstream((row, col), fdr)

        if downstream is not None:
            downstream_basin = basin_labels[downstream]
            if downstream_basin != basin_id:
                if downstream_basin not in graph:
                    graph[downstream_basin] = []
                graph[downstream_basin].append(basin_id)

    return graph
```

### Path Tracing

For a given outlet, find all upstream basins and select the one with the longest flow path:

```python
def find_longest_flow_path(outlet_basin, graph, max_cells, fdr):
    # Find all upstream basins
    upstream_basins = find_all_upstream(outlet_basin, graph)

    # Find basin with maximum distance
    best_basin = outlet_basin
    best_distance = max_cells[outlet_basin][2]

    for basin_id in upstream_basins:
        if max_cells[basin_id][2] > best_distance:
            best_distance = max_cells[basin_id][2]
            best_basin = basin_id

    # Trace path from max cell to outlet
    max_row, max_col, _ = max_cells[best_basin]
    path = trace_downstream((max_row, max_col), fdr, outlet_cell)

    return path, best_distance
```

### Upstream Basin Discovery

Use BFS on the basin graph to find all upstream basins:

```python
def find_all_upstream(outlet_basin, graph):
    visited = set()
    queue = Queue()
    queue.push(outlet_basin)

    while not queue.empty():
        basin = queue.pop()

        if basin in visited:
            continue

        visited.add(basin)

        if basin in graph:
            for upstream_basin in graph[basin]:
                queue.push(upstream_basin)

    return visited
```

### Downstream Path Tracing

Trace the flow path from a cell to the outlet:

```python
def trace_downstream(start, fdr, outlet):
    path = [start]
    current = start

    while current != outlet:
        downstream = get_downstream(current, fdr)
        if downstream is None:
            break
        path.append(downstream)
        current = downstream

    return path
```

## Output

### Flow Length Raster

A floating-point raster where each cell contains its flow distance to the outlet:

- **Units**: Map units (meters for projected, meters for geographic via Haversine)
- **Nodata**: -1 for cells not draining to any drainage point

### Longest Flow Path Vectors

LineString features representing the longest flow path for each basin:

- **Geometry**: Path from hydraulically most distant cell to outlet
- **Attributes**: Basin ID, path length

## Properties

The flow length raster satisfies:

1. **Monotonic increase**: Flow length increases upstream

$$
L(\text{upstream}(c)) > L(c)
$$

2. **Zero at outlets**: Drainage points have flow length 0

$$
L(d) = 0 \quad \forall d \in D
$$

3. **Path additivity**: Flow length equals sum of step distances

$$
L(c) = L(\text{downstream}(c)) + d(c, \text{downstream}(c))
$$

## Complexity

Let $n$ be total cells and $b$ be basin count.

| Operation | Time Complexity |
|-----------|-----------------|
| Upstream BFS | $O(n)$ |
| Graph construction | $O(b)$ |
| Upstream basin search | $O(b)$ |
| Path tracing | $O(l)$ |

Where $l$ is the length of the longest flow path.

## Coordinate System Detection

The algorithm automatically detects whether the raster uses projected or geographic coordinates by examining the spatial reference:

- **Geographic**: Units are degrees, use Haversine distance
- **Projected**: Units are linear (meters, feet), use Euclidean distance

## See Also

- [Basin Delineation](basin-delineation.md) - Defining drainage basins
- [Flow Direction](flow-direction.md) - Computing flow directions
- [Flow Accumulation](flow-accumulation.md) - Computing drainage area
