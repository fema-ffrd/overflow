from numba import int64, float32
from numba.types import UniTuple, ListType, DictType, Tuple, Array


Int64List = ListType(int64)
Int64Pair = UniTuple(int64, 2)
Int64PairList = ListType(Int64Pair)
Int64PairListList = ListType(Int64PairList)
DictInt64Float32 = DictType(int64, float32)
Int64Array3D = Array(int64, 3, "C")
Int64Array3DList = ListType(Int64Array3D)
