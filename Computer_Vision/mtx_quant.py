"""
Problem:
Quantize uniformly a floating-point matrix to integer values
in a specific range, such as 0–255,
commonly used in image processing.
"""
import numpy as np
from numpy.typing import NDArray


def quantize(mtx:NDArray, bits:int=8):
    type_map = {8:np.uint8,16:np.uint16,32:np.uint32,64:np.uint64}
    if bits not in type_map:
        raise ValueError(f"Invalid number of bits: {bits}")
    min_value, max_value = 0, 2**bits - 1
    min_mtx = np.min(mtx)
    max_mtx = np.max(mtx)
    # Map the mtx to [0, 1] range
    norm_mtx = (mtx - min_mtx) / (max_mtx - min_mtx)
    q_mtx = (
            norm_mtx * (max_value - min_value) + min_value
    ).astype(type_map[bits])
    return q_mtx, (min_mtx, max_mtx)

def dequantize(q_mtx:NDArray, min_mtx:float,max_mtx:float, bits:int = 8):
    type_map = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    min_value, max_value = 0, 2 ** bits - 1
    norm_mtx = (q_mtx.astype(type_map[bits]) - min_value) / (max_value - min_value)
    mtx = min_mtx + (max_mtx - min_mtx) * norm_mtx
    return mtx

if __name__ == "__main__":
    mtx = np.linspace(0, 1, 64).reshape(8, 8)
    bits = 8
    q_mtx, (min_mtx, max_mtx) = quantize(mtx,bits)
    mtx_rec = dequantize(q_mtx,min_mtx,max_mtx, bits)
    print(mtx, end="\n\n")
    print(q_mtx)
    print(q_mtx.dtype)
    print(mtx_rec)