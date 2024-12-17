import numpy as np

# constants used in the overflow module
DEFAULT_SEARCH_RADIUS = 200
DEFAULT_MAX_PITS = 24
UNVISITED_INDEX = -1
EPSILON_GRADIENT = 1e-5  # small value to apply to gradient of breaching to nodata cells
DEFAULT_CHUNK_SIZE = 2000

#   3  |   2    |  1
# ------------------
#   4  |   8   |  0
# ------------------
#   5  |   6   |  7
FLOW_DIRECTION_EAST = 0
FLOW_DIRECTION_NORTH_EAST = 1
FLOW_DIRECTION_NORTH = 2
FLOW_DIRECTION_NORTH_WEST = 3
FLOW_DIRECTION_WEST = 4
FLOW_DIRECTION_SOUTH_WEST = 5
FLOW_DIRECTION_SOUTH = 6
FLOW_DIRECTION_SOUTH_EAST = 7
FLOW_DIRECTION_UNDEFINED = 8
FLOW_DIRECTION_NODATA = 9
# numba does not support global constant dictionaries
# so we do lookups by ordering indicies in the array to
# match the flow direction codes
# see https://github.com/numba/numba/issues/6488
NEIGHBOR_OFFSETS = np.array(
    [
        (0, 1),  # FLOW_EAST
        (-1, 1),  # FLOW_NORTH_EAST
        (-1, 0),  # FLOW_NORTH
        (-1, -1),  # FLOW_NORTH_WEST
        (0, -1),  # FLOW_WEST
        (1, -1),  # FLOW_SOUTH_WEST
        (1, 0),  # FLOW_SOUTH
        (1, 1),  # FLOW_SOUTH_EAST
    ]
)
NEIGHBOR_OFFSETS_8 = NEIGHBOR_OFFSETS
NEIGHBOR_OFFSETS_4 = np.array(
    [
        (0, 1),  # FLOW_EAST
        (-1, 0),  # FLOW_NORTH
        (0, -1),  # FLOW_WEST
        (1, 0),  # FLOW_SOUTH
    ]
)
FLOW_DIRECTIONS = np.array(
    [
        FLOW_DIRECTION_EAST,
        FLOW_DIRECTION_NORTH_EAST,
        FLOW_DIRECTION_NORTH,
        FLOW_DIRECTION_NORTH_WEST,
        FLOW_DIRECTION_WEST,
        FLOW_DIRECTION_SOUTH_WEST,
        FLOW_DIRECTION_SOUTH,
        FLOW_DIRECTION_SOUTH_EAST,
        FLOW_DIRECTION_UNDEFINED,
        FLOW_DIRECTION_NODATA,
    ],
    dtype=np.uint8,
)

FLOW_ACCUMULATION_NODATA = -9999
FLOW_TERMINATES = (-1, -1)
FLOW_EXTERNAL = (-2, -2)
EDGE_LABEL = np.int64(1)  # edge label of fill depressions algorithm
# Define flags for each side
TOP = 0b0001  # Binary representation of 1
RIGHT = 0b0010  # Binary representation of 2
BOTTOM = 0b0100  # Binary representation of 4
LEFT = 0b1000  # Binary representation of 8
