# Algorithm Details

This section provides in-depth documentation of the algorithms implemented in Overflow. Each page covers the mathematical foundations, data structures, and implementation details for a specific hydrological processing step.

## Terrain Conditioning

Prepare raw DEMs for hydrological analysis by removing artificial depressions:

- [Breach Algorithm](breach.md) - Remove depressions by carving least-cost flow paths
- [Fill Algorithm](fill.md) - Fill depressions using priority-flood

## Flow Routing

Determine how water moves across the terrain:

- [Flow Direction](flow-direction.md) - Compute D8 flow directions
- [Flat Resolution](flat-resolution.md) - Resolve undefined flow in flat regions
- [Flow Accumulation](flow-accumulation.md) - Calculate upstream contributing area

## Feature Extraction

Derive hydrographic features from the flow network:

- [Stream Extraction](stream-extraction.md) - Extract stream networks from flow accumulation
- [Basin Delineation](basin-delineation.md) - Delineate drainage basins
- [Flow Length](flow-length.md) - Calculate upstream flow length and longest flow paths
