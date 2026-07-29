# Burn Algorithm

## Overview

The burn algorithm rewrites a DEM inside regions defined by a co-registered mask raster. It reads the DEM and the mask and writes a modified DEM in which the selected regions carry an imposed elevation.

Two of the three methods are trivial. `constant` and `relative` compute each output cell from the DEM value and mask value at that cell alone, so they need one pass over the data and nothing else. The `statistic` method is the interesting one: each contiguous region takes a statistic of the DEM beneath it, which no single cell and no single tile can determine on its own. For large rasters processed tile by tile, a region that crosses a tile boundary appears as two or more independent fragments, and the algorithm has to recognize them as one region before it can compute anything.

## Definitions

Let $M$ be the mask raster, $Z$ the DEM, and $S$ the set of selected mask values.

A cell $c$ is **selected** when $M_c \in S$ and $M_c$ is neither the mask's nodata value nor NaN.

A **region** is an equivalence class of selected cells under the transitive closure of the adjacency relation

$$
c \sim d \iff d \in N(c) \land M_c = M_d
$$

where $N(c)$ is the 8-connected neighborhood by default, or the 4-connected neighborhood when `connectivity=4`. Requiring $M_c = M_d$ rather than merely requiring both to be selected is what keeps two touching classes in separate regions.

For a region $R$ the burned elevation is

$$
b(R) = \operatorname{stat}\left(\{Z_c : c \in R,\ Z_c \neq \text{nodata}\}\right) - \delta
$$

with $\operatorname{stat} \in \{\min, \max, \operatorname{mean}\}$ and $\delta$ the burn offset. When the set is empty the region has no defined value and is left untouched.

## Mergeable Statistics

The tiled algorithm works because $\min$, $\max$ and $\operatorname{mean}$ can all be computed from four quantities that combine associatively. For a region split into fragments $R_1, \dots, R_k$, accumulate per fragment

$$
\left(\min_i,\ \max_i,\ \Sigma_i = \sum_{c \in R_i} Z_c,\ n_i = |R_i|\right)
$$

and recover the whole region's statistics as

$$
\min = \min_i \min_i, \qquad
\max = \max_i \max_i, \qquad
\operatorname{mean} = \frac{\sum_i \Sigma_i}{\sum_i n_i}
$$

No fragment needs to know about any other while it is being processed, and the fragments never need to be re-read. This is the property that decides which statistics the tool can offer: a median or an arbitrary percentile is **not** recoverable this way without retaining every value in the region, which is why they are not supported.

## Data Types and Structures

### Input/Output Types

| Parameter | Type | Description | No-Data Value |
|-----------|------|-------------|---------------|
| dem | any float or integer | Input elevation raster | required |
| mask | any numeric | Binary or classified region raster | optional |
| output | same as input dem | Burned elevation raster | inherited from dem |
| labels | int64 | Temporary per-cell region label | 0 |

### Internal Structures

| Structure | Type | Purpose |
|-----------|------|---------|
| `MaskSelection` | jitclass | Which mask values are selected and, for the direct methods, their burn values |
| `labels` | int64 2D array | Region label per cell, 0 outside any region |
| `stats` | float64 array, shape $(n, 4)$ | Per-region $(\min, \max, \Sigma, n)$ for one tile |
| `UnionFind` | jitclass over `Dict[int64, int64]` | Disjoint set joining labels that turn out to be one region |
| `label_perimeters` | int64 2D array | Region labels around each tile's perimeter |
| `mask_perimeters` | float64 2D array | Mask values around each tile's perimeter |
| `burn_lookup` | `Dict[int64, float64]` | Final label to burned elevation map |

## Algorithm

### Region labeling

Within a tile, regions are found by flood fill in the shape of Algorithm 4 *LabelFlats* from [Barnes, Lehman & Mulla (2014)](https://rbarnes.org/sci/2014_flats.pdf), with equality of mask value standing in for equality of elevation. Cells are scanned in row-major order; the first unlabeled selected cell seeds a new label, and a FIFO queue drives the fill outward across neighbors sharing that cell's mask value.

A single queue is allocated once and reused across regions rather than per region, which matters when a mask holds many small blobs.

### Local statistics

A second scan over the labeled tile accumulates $(\min, \max, \Sigma, n)$ per label, skipping DEM nodata and NaN. This is a single sequential pass; it is not parallelized within the tile because the accumulators would race.

## Tiled Processing

### Globally unique labels without renumbering

Each tile draws its labels from a private range,

$$
\text{label\_offset}(t) = s^2 \cdot t + 1
$$

for tile index $t$ and chunk size $s$. Since a tile of $s \times s$ cells can hold at most $s^2$ regions, the ranges cannot overlap, so labels from different tiles are distinct without any communication or renumbering pass. Label 0 is reserved for "not in a region". This is the same trick the tiled depression fill uses for watershed labels.

### Joining regions across seams

Only tile perimeters can participate in a cross-tile adjacency, so each tile stashes its perimeter labels and perimeter mask values, and nothing else survives the first pass. The solve then walks the tile grid, and for each tile joins it to its eastern and southern neighbors plus the two diagonal pairs meeting at its southeastern corner:

```
+ - - + - - +
|  A  |  B  |
+ - - * - - +
|  C  |  D  |
+ - - + - - +
```

A joins B along a vertical seam and C along a horizontal one; the corner joins A-D and B-C cover the diagonal adjacencies at the point marked `*`, which no edge scan reaches. Every adjacency in the grid is covered exactly once, and the bounds guards keep it correct for a tile grid that is a single tile wide or tall.

Along a seam, index $i$ in one tile's perimeter sits directly against index $i$ in the other's. With 8-connectivity a cell also touches its two diagonal neighbors across the seam, so the scan compares $i$ against $i-1$, $i$ and $i+1$; with 4-connectivity it compares only $i$. Two labels are joined when both cells are labeled and hold the same mask value, exactly the rule the within-tile flood fill applies.

Joins go into a union-find structure with union by size and path-compressed find, giving near constant amortized time per join.

### Edge padding

Tiles hanging off the right or bottom edge of the raster are padded out to a full square by the reader, and the labeler is told how many rows and columns hold real data so padded cells are never labeled. Without that bound, padding sharing a selected mask value could flood-fill into a real region and then join two genuinely disconnected regions across a seam, giving both the wrong statistic.

### Solve and apply

Every label is resolved to its representative, the four statistics are merged per representative, and the result is expanded back into a flat label-to-value lookup so the final pass is one dict read per cell. The apply pass re-reads the **original** DEM, not the partially written output, so statistics can never be contaminated by values the tool itself wrote.

## Complexity

Let $N$ be the number of cells, $R$ the number of region fragments and $T$ the number of tiles with chunk size $s$.

| Phase | Time | Memory |
|-------|------|--------|
| `constant` / `relative` | $O(N)$ | $O(s^2)$ per worker |
| Label and accumulate | $O(N)$ | $O(s^2)$ per worker |
| Seam joining | $O(T s\,\alpha(R))$ | $O(T s)$ perimeters |
| Solve | $O(R\,\alpha(R))$ | $O(R)$ |
| Apply | $O(N)$ | $O(s^2)$ per worker |

$\alpha$ is the inverse Ackermann function, effectively constant. The solve is proportional to the number of region fragments rather than to raster size, so a mask covering a small fraction of a very large DEM stays cheap.

Tiles are processed concurrently by numba kernels compiled with `nogil`, using the same thread pool and back-pressure pattern as the other tiled tools. Raster writes and the merge of per-tile statistics into the global dicts are serialized under a lock, since numba typed dicts are not safe to insert into concurrently.

## Input Requirements

- The DEM must declare a nodata value.
- The mask must have the same shape and geotransform as the DEM. A nodata value is recommended but not required.
- Mask values are compared by exact equality, so a floating point mask must hold exact class values.
- Naming the mask's own nodata value as a selected value is rejected.

## See Also

- [Burn User Guide](../user-guide/terrain-conditioning/burn.md) - Parameters and usage
- [Fill](fill.md) - The tiled depression fill this design borrows its tiling structure from
- [Breach](breach.md) - The other terrain conditioning operation
