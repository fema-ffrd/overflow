from enum import IntEnum, unique

from overflow._util.constants import (
    FLOW_DIRECTION_EAST,
    FLOW_DIRECTION_NODATA,
    FLOW_DIRECTION_NORTH,
    FLOW_DIRECTION_NORTH_EAST,
    FLOW_DIRECTION_NORTH_WEST,
    FLOW_DIRECTION_SOUTH,
    FLOW_DIRECTION_SOUTH_EAST,
    FLOW_DIRECTION_SOUTH_WEST,
    FLOW_DIRECTION_UNDEFINED,
    FLOW_DIRECTION_WEST,
)


@unique
class FlowDirection(IntEnum):
    """D8 flow direction codes.

    These codes represent the eight cardinal and intercardinal directions
    plus special values for undefined flow and nodata cells.

    The numeric values correspond to the index in the neighbor offset array,
    starting from East (0) and going counter-clockwise.

    | 3 | 2 | 1 |
    | :-: | :-: | :-: |
    | 4 | 8 | 0 |
    | 5 | 6 | 7 |

    """

    EAST = FLOW_DIRECTION_EAST
    NORTH_EAST = FLOW_DIRECTION_NORTH_EAST
    NORTH = FLOW_DIRECTION_NORTH
    NORTH_WEST = FLOW_DIRECTION_NORTH_WEST
    WEST = FLOW_DIRECTION_WEST
    SOUTH_WEST = FLOW_DIRECTION_SOUTH_WEST
    SOUTH = FLOW_DIRECTION_SOUTH
    SOUTH_EAST = FLOW_DIRECTION_SOUTH_EAST
    UNDEFINED = FLOW_DIRECTION_UNDEFINED
    NODATA = FLOW_DIRECTION_NODATA
