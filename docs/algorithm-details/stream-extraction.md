# Stream Extraction Algorithm

## Overview

The stream extraction algorithm converts flow accumulation data into vector stream networks. Cells exceeding a threshold are classified as streams, then vectorized into polylines with junction points at confluences and sources.

## Core Concept

Stream extraction is a three-step process:

1. **Classification**: Apply threshold to identify stream cells
2. **Node Detection**: Find topologically significant points (sources, confluences)
3. **Vectorization**: Trace stream segments between nodes

## Stream Classification

A cell belongs to the stream network if its flow accumulation meets or exceeds the threshold:

$$
\text{is\_stream}(c) = \begin{cases}
\text{true} & \text{if } \text{acc}(c) \geq T \\
\text{false} & \text{otherwise}
\end{cases}
$$

where $T$ is the accumulation threshold parameter.

**Threshold selection:**

- Lower thresholds extract more tributaries (denser network)
- Higher thresholds extract only main channels (sparser network)
- Common heuristics: $T = 100$ to $T = 1000$ cells, or based on contributing area

## Node Detection

Nodes are topologically significant points in the stream network:

**Source nodes**: Stream cells with no upstream stream neighbors

$$
\text{is\_source}(c) = \text{is\_stream}(c) \land |\{n \in N(c) : \text{upstream}(n, c) \land \text{is\_stream}(n)\}| = 0
$$

**Confluence nodes**: Stream cells with multiple upstream stream neighbors

$$
\text{is\_confluence}(c) = \text{is\_stream}(c) \land |\{n \in N(c) : \text{upstream}(n, c) \land \text{is\_stream}(n)\}| > 1
$$

where $\text{upstream}(n, c)$ is true if cell $n$ flows directly to cell $c$.

### Upstream Neighbor Detection

A neighbor $n$ is upstream of cell $c$ if $n$'s flow direction points to $c$:

```python
def is_upstream(neighbor, cell, fdr):
    """Check if neighbor flows into cell."""
    downstream_of_neighbor = get_downstream(neighbor, fdr)
    return downstream_of_neighbor == cell
```

### Algorithm

```python
def find_nodes(stream_raster, fdr):
    nodes = []

    for cell in all_cells():
        if not stream_raster[cell]:
            continue

        # Count upstream stream neighbors
        upstream_count = 0
        for neighbor in neighbors_8(cell):
            if stream_raster[neighbor] and is_upstream(neighbor, cell, fdr):
                upstream_count += 1

        if upstream_count == 0:  # Source
            nodes.append(cell)
        elif upstream_count > 1:  # Confluence
            nodes.append(cell)

    return nodes
```

## Stream Tracing

Each stream segment connects two nodes. Starting from each node, trace downstream until reaching another node or the domain boundary:

```python
def trace_stream(start_node, fdr, stream_raster, node_set):
    path = [start_node]
    current = start_node

    while True:
        downstream = get_downstream(current, fdr)

        if downstream is None:
            # Reached boundary
            break

        if not stream_raster[downstream]:
            # Left stream network
            break

        path.append(downstream)

        if downstream in node_set and downstream != start_node:
            # Reached another node
            break

        current = downstream

    return path
```

## Coordinate Conversion

Raster cell indices are converted to geographic coordinates using the geotransform:

$$
x = x_0 + (c + 0.5) \cdot \Delta x
$$

$$
y = y_0 + (r + 0.5) \cdot \Delta y
$$

where:
- $(x_0, y_0)$ is the origin (top-left corner)
- $(\Delta x, \Delta y)$ are pixel dimensions
- $(r, c)$ are row and column indices
- The $+0.5$ offset places coordinates at cell centers

## Output Structure

The algorithm produces two vector layers:

### Streams Layer (LineString)

Each feature represents a stream segment between nodes:

- **Geometry**: LineString with vertices at cell centers
- **Attributes**: Feature ID, optional length

### Junctions Layer (Point)

Each feature represents a topologically significant point:

- **Sources**: Headwater points (upstream extent)
- **Confluences**: Where tributaries join
- **Outlets**: Where streams exit the domain

## Tiled Processing

For large rasters, streams are extracted per-tile and then merged:

### Per-Tile Extraction

1. Extract streams and junctions within tile
2. Track tile coordinates for global positioning
3. Record stream endpoints for cross-tile merging

### Cross-Tile Merging

When streams cross tile boundaries, they are split into separate features. Post-processing merges these:

1. **Identify edge junctions**: Junctions with exactly 2 stream endpoints
2. **Classify merge type**: Determine endpoint orientations
3. **Merge geometries**: Concatenate coordinates in correct order

### Merge Types

For two stream segments meeting at a junction:

| Type | Description | Action |
|------|-------------|--------|
| 0 | downstream $\to$ upstream | Append second to first |
| 1 | upstream $\to$ downstream | Prepend second to first |
| 2 | upstream $\gets$ upstream | Reverse one, concatenate |
| 3 | downstream $\gets$ downstream | Reverse one, concatenate |

```python
def merge_streams(stream1, stream2, merge_type):
    if merge_type == 0:
        return stream1.coords + stream2.coords[1:]
    elif merge_type == 1:
        return stream2.coords + stream1.coords[1:]
    elif merge_type == 2:
        return list(reversed(stream1.coords)) + stream2.coords[1:]
    elif merge_type == 3:
        return stream1.coords + list(reversed(stream2.coords))[1:]
```

### Spatial Hashing

To efficiently find matching endpoints across tiles, a spatial hash is used:

$$
h(x, y) = \lfloor x / \delta \rfloor \cdot P + \lfloor y / \delta \rfloor
$$

where $\delta$ is the grid cell size and $P$ is a large prime.

## Complexity

Let $n$ be total cells, $s$ be stream cells, and $j$ be junction count.

| Operation | Time Complexity |
|-----------|-----------------|
| Classification | $O(n)$ |
| Node detection | $O(s)$ |
| Stream tracing | $O(s)$ |
| Coordinate conversion | $O(s)$ |
| Total | $O(n)$ |

For tiled processing with merging:

| Operation | Time Complexity |
|-----------|-----------------|
| Per-tile extraction | $O(n)$ total |
| Edge junction finding | $O(j)$ |
| Stream merging | $O(j \cdot l)$ |

where $l$ is average stream segment length.

## Network Properties

The extracted stream network forms a directed tree (or forest):

1. **Connectivity**: All stream cells are connected
2. **Hierarchy**: Flow direction defines parent-child relationships
3. **Single outlet**: Each connected component has one outlet (unless truncated by boundary)

## See Also

- [Flow Accumulation](flow-accumulation.md) - Computing accumulation for thresholding
- [Basin Delineation](basin-delineation.md) - Using stream junctions as drainage points
