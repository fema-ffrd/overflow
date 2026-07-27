from numba import float32, float64, int64  # type: ignore[attr-defined]
from numba.types import Array, DictType, ListType, Tuple, UniTuple

Int64List = ListType(int64)
Int64Pair = UniTuple(int64, 2)
Int64PairList = ListType(Int64Pair)
Int64PairListList = ListType(Int64PairList)
DictInt64Float32 = DictType(int64, float32)
Int64Array3D = Array(int64, 3, "C")
Int64Array3DList = ListType(Int64Array3D)
IndexOffsetPair = Tuple((int64, float64))
