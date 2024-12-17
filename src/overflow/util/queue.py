from numba import int64
from numba.types import ListType
from numba.typed import List  # pylint: disable=no-name-in-module
from numba.experimental import jitclass
from overflow.util.raster import GridCellFloat32, GridCellInt64
from overflow.util.numba_types import Int64Pair


def create_queue_class(numba_type):
    spec = [
        ("_front", int64),
        ("_size", int64),
        ("_data", ListType(numba_type)),
    ]

    @jitclass(spec)
    class Queue:
        """
        A resizable First-In-First-Out (FIFO) queue implemented with a circular buffer.
        The complexity of the push and pop operations is amortized O(1) due to the occasional resizing.
        This custom class is needed since numba does not yet support the built-in
        collections.deque class.

        Attributes
        ----------
        _front : int
            The index of the front of the queue.
        _size : int
            The current size of the queue.
        _data : np.ndarray
            The underlying data array storing the elements of the queue.
        """

        def __init__(self, data: list):
            """
            Initialize the queue with the given data.

            Parameters
            ----------
            data : list or np.ndarray
                The initial data to populate the queue.
            """
            if len(data) == 0:
                raise ValueError("list must not be empty")
            self._front = 0
            self._data = List.empty_list(numba_type)
            for value in data:
                self._data.append(value)
            self._size = len(self._data)

        def push(self, value):
            """
            Add a value to the end of the queue.

            Parameters
            ----------
            value : int
                The value to add to the queue.
            """
            if self._size == len(self._data):
                self._resize(2 * len(self._data))
            self._data[(self._front + self._size) % len(self._data)] = value
            self._size += 1

        def pop(self):
            """
            Remove and return the value at the front of the queue.

            Returns
            -------
            int
                The value at the front of the queue.

            Raises
            ------
            IndexError
                If the queue is empty.
            """
            if self._size == 0:
                raise IndexError("pop from an empty queue")
            value = self._data[self._front]
            self._front = (self._front + 1) % len(self._data)
            self._size -= 1
            if self._size != 0 and self._size < len(self._data) // 4:
                self._resize(len(self._data) // 2)
            return value

        def _resize(self, new_capacity):
            """
            Resize the underlying data array to the given capacity.

            Parameters
            ----------
            new_capacity : int
                The new capacity for the data array.
            """
            new_data = List.empty_list(numba_type)
            old_len = min(len(self._data), new_capacity)

            for i in range(old_len):
                new_data.append(self._data[(self._front + i) % len(self._data)])

            if old_len < new_capacity:
                # fill remaining with junk data
                # since python has no concept of uninitialized memory
                junk = self._data[0]  # always at least one element
                for i in range(new_capacity - old_len):
                    new_data.append(junk)

            self._data = new_data
            self._front = 0

        def __len__(self):
            """
            Return the current size of the queue.

            Returns
            -------
            int
                The current size of the queue.
            """
            return self._size

    return Queue


# pylint: disable=no-member
Int64PairQueue = create_queue_class(Int64Pair)
# pylint: disable=no-member
GridCellInt64Queue = create_queue_class(GridCellInt64.class_type.instance_type)
# pylint: disable=no-member
GridCellFloat32Queue = create_queue_class(GridCellFloat32.class_type.instance_type)
