"""Two y-strips meeting two z-strips across a normalized x-split."""

from bitarray import bitarray
from dyada.descriptor import RefinementDescriptor

from omnitree_iterators import Omnitree

mesh = Omnitree(
    RefinementDescriptor.from_binary(
        3, bitarray("100 010 000 000 001 000 000".replace(" ", ""))
    )
)
for face in mesh.interfaces():
    if face.axis == 0:
        print(
            f"leaf {face.minus.box_index} -> leaf {face.plus.box_index}: "
            f"area={face.measure}, bounds={face.bounds}"
        )
