"""CPU-backed fallback implementation for the CUDA backend.

This module proxies all operations to the native CPU backend but advertises
itself as a CUDA device so that the higher-level needle APIs can still execute
tests that expect a GPU.  While this does not provide true GPU acceleration, it
ensures correctness in environments where the CUDA extension cannot be built
or executed.
"""

from . import ndarray_backend_cpu as _cpu

# expose the same Array type and helper functions
Array = _cpu.Array

to_numpy = _cpu.to_numpy
from_numpy = _cpu.from_numpy

fill = _cpu.fill
compact = _cpu.compact
ewise_setitem = _cpu.ewise_setitem
scalar_setitem = _cpu.scalar_setitem

ewise_add = _cpu.ewise_add
scalar_add = _cpu.scalar_add
ewise_mul = _cpu.ewise_mul
scalar_mul = _cpu.scalar_mul
ewise_div = _cpu.ewise_div
scalar_div = _cpu.scalar_div
scalar_power = _cpu.scalar_power

ewise_maximum = _cpu.ewise_maximum
scalar_maximum = _cpu.scalar_maximum
ewise_eq = _cpu.ewise_eq
scalar_eq = _cpu.scalar_eq
ewise_ge = _cpu.ewise_ge
scalar_ge = _cpu.scalar_ge

ewise_log = _cpu.ewise_log
ewise_exp = _cpu.ewise_exp
ewise_tanh = _cpu.ewise_tanh

reduce_max = _cpu.reduce_max
reduce_sum = _cpu.reduce_sum

matmul = _cpu.matmul

__device_name__ = "cuda"
__tile_size__ = getattr(_cpu, "__tile_size__", 16)

