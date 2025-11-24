# Flow Direction Algorithm

## Overview

The flow direction algorithm assigns each cell a D8 direction code indicating the steepest descent to one of eight neighbors. This is the foundational step for all downstream hydrological analysis.

## D8 Flow Model

The D8 (deterministic eight-neighbor) model directs all flow from a cell to the single neighbor with the steepest downhill slope. Each cell receives exactly one flow direction.

### Direction Encoding

Directions are encoded as integers 0-9:

| Code | Direction | Offset $(\Delta c, \Delta r)$ |
|------|-----------|------------------------------|
| 0 | East | $(1, 0)$ |
| 1 | Northeast | $(1, -1)$ |
| 2 | North | $(0, -1)$ |
| 3 | Northwest | $(-1, -1)$ |
| 4 | West | $(-1, 0)$ |
| 5 | Southwest | $(-1, 1)$ |
| 6 | South | $(0, 1)$ |
| 7 | Southeast | $(1, 1)$ |
| 8 | Undefined | — |
| 9 | Nodata | — |

Direction indices proceed counter-clockwise from East.

### Visual Representation

```
  3  |  2  |  1
-----+-----+-----
  4  |  X  |  0
-----+-----+-----
  5  |  6  |  7
```

## Slope Calculation

For a cell $c$ at position $(r, c)$ with elevation $z_c$, the slope to neighbor $n$ at position $(r + \Delta r, c + \Delta c)$ is:

$$
S_{c \to n} = \frac{z_c - z_n}{d_{c,n}}
$$

where $d_{c,n}$ is the distance between cell centers:

$$
d_{c,n} = \begin{cases}
1 & \text{cardinal directions (N, S, E, W)} \\
\sqrt{2} & \text{diagonal directions (NE, NW, SE, SW)}
\end{cases}
$$

The distance normalization ensures that diagonal and cardinal flows are weighted appropriately. Without it, diagonal neighbors would be unfairly penalized due to their greater physical distance.

## Algorithm

For each cell, compute slopes to all valid neighbors and select the steepest descent:

```python
def flow_direction(dem, nodata):
    rows, cols = dem.shape
    fdr = empty((rows, cols), dtype=uint8)

    for row in range(rows):
        for col in range(cols):
            if is_nodata(dem[row, col]) or is_nan(dem[row, col]):
                fdr[row, col] = 9  # Nodata
                continue

            max_slope = -infinity
            max_direction = -1

            for direction in range(8):
                dr, dc = OFFSETS[direction]
                nr, nc = row + dr, col + dc

                if out_of_bounds(nr, nc):
                    continue

                if is_nodata(dem[nr, nc]) or is_nan(dem[nr, nc]):
                    continue

                distance = sqrt(2) if is_diagonal(direction) else 1.0
                slope = (dem[row, col] - dem[nr, nc]) / distance

                if slope > max_slope:
                    max_slope = slope
                    max_direction = direction

            if max_slope <= 0:
                fdr[row, col] = 8  # Undefined (flat or pit)
            else:
                fdr[row, col] = max_direction

    return fdr
```

## Flat Regions and Undefined Flow

When all neighbors have elevation $\geq z_c$ (local minimum or flat), no valid downhill direction exists. These cells receive code 8 (undefined).

**Flat region**: All neighbors have equal elevation
$$
\forall n \in N(c): z_n = z_c \implies \text{fdr}_c = 8
$$

**Pit/sink**: All neighbors have greater elevation
$$
\forall n \in N(c): z_n > z_c \implies \text{fdr}_c = 8
$$

To resolve undefined flow in flat regions, apply the [Flat Resolution](flat-resolution.md) algorithm.

## Nodata Handling

- **Input nodata**: Cells with nodata elevation receive direction code 9
- **Nodata neighbors**: Excluded from slope calculation (flow cannot enter nodata)
- **Propagation**: Nodata status propagates to flow direction output

## Properties

The D8 flow direction raster satisfies:

1. **Deterministic**: Each cell has exactly one flow direction
2. **Unique outflow**: Water from any cell flows to exactly one neighbor
3. **Acyclic** (on conditioned DEMs): Following flow directions never returns to the starting cell

Formally, for a properly conditioned DEM:

$$
\text{fdr}_c \in \{0, 1, \ldots, 7\} \implies z_{\text{downstream}(c)} < z_c
$$

## Chunked Processing

For large rasters, processing is performed in tiles with a 1-pixel buffer:

1. Read tile with 1-pixel overlap on all edges
2. Compute flow directions for interior cells
3. Buffer ensures all 8 neighbors are available for interior cells
4. Write interior (non-buffer) region to output

## Complexity

| Metric | Value |
|--------|-------|
| Time | $O(n)$ |
| Space | $O(n)$ |
| Per-cell operations | 8 neighbor comparisons |

Where $n$ is the number of cells.

## Limitations

1. **Flat regions**: Cannot resolve flow in areas of constant elevation without additional processing
2. **Single flow path**: Does not support flow dispersion (unlike D-infinity or multiple flow direction models)
3. **Grid artifact sensitivity**: Results depend on DEM resolution and alignment

## See Also

- [Flat Resolution](flat-resolution.md) - Resolving undefined flow directions
- [Fill](fill.md) - Depression removal before flow direction
- [Flow Accumulation](flow-accumulation.md) - Computing drainage area from flow directions
