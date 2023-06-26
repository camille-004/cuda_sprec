"""Orthogonal Matching Pursuit CUDA invocation."""
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from cusprec.algorithms.base import Algorithm
from cusprec.constants import BLOCK_SIZE
from cusprec.data.dataset import Dataset


class OMP(Algorithm):
    """Orthogonal Matching Pursuit algorithm implemented in PyCUDA.

    Parameters
    ----------
    data : Dataset
        The dataset containing the original signal, measurement matrix, and
        observed signal.

    Attributes
    ----------
    data : Dataset
        The dataset containing the original signal, measurement matrix, and
        observed signal.
    m : int
        Number of measurements.
    n : int
        Number of original signal samples.
    k : int
        Sparsity level.
    A : ndarray
        Measurement matrix.
    x : ndarray
        Original signal.
    b : ndarray
        Observed signal.
    A_gpu : cuda.DeviceAllocation
        GPU memory allocated for the measurement matrix.
    x_gpu : cuda.DeviceAllocation
        GPU memory allocated for the original signal.
    b_gpu : cuda.DeviceAllocation
        GPU memory allocated for the observed signal.
    r : ndarray
        Residual of the original signal.
    s : ndarray
        Recovered signal.
    r_gpu : cuda.DeviceAllocation
        GPU memory allocated for the residual.
    s_gpu : cuda.DeviceAllocation
        GPU memory allocated for the recovered signal.

    Methods
    -------
    execute() -> None:
        Executes the Orthogonal Matching Pursuit algorithm.

    get_results() -> np.ndarray:
        Returns the recovered signal.

    cleanup() -> None:
        Frees up the GPU memory allocated.
    """

    def __init__(self, data: Dataset):
        self.data = data
        self.m = data.m()
        self.n = data.n()
        self.k = data.k()
        self.A = data.A
        self.x = data.x
        self.b = data.b

        self.A_gpu = cuda.mem_alloc(self.A.nbytes)
        self.x_gpu = cuda.mem_alloc(self.x.nbytes)
        self.b_gpu = cuda.mem_alloc(self.b.nbytes)

        cuda.memcpy_htod(self.A_gpu, self.A)
        cuda.memcpy_htod(self.x_gpu, self.x)
        cuda.memcpy_htod(self.b_gpu, self.b)

        self.r = self.x.copy()
        self.s = np.zeros_like(self.b)

        self.r_gpu = cuda.mem_alloc(self.r.nbytes)
        self.s_gpu = cuda.mem_alloc(self.s.nbytes)

        cuda.memcpy_htod(self.r_gpu, self.r)
        cuda.memcpy_htod(self.s_gpu, self.s)

    def execute(self) -> None:
        """Execute the OMP algorithm.

        Returns
        -------
        None
        """
        ops = self.read_kernel("ops.cu")
        dot_product = ops.get_function("dotProduct")
        scalar_multiply = ops.get_function("scalarMultiply")
        vector_add = ops.get_function("vectorAdd")
        vector_subtract = ops.get_function("vectorSubtract")

        threads_per_block = 256
        blocks_per_grid = (self.m + threads_per_block - 1) // threads_per_block
        grid_size = blocks_per_grid

        for i in range(self.k):
            c = np.zeros_like(self.b)
            c_gpu = cuda.mem_alloc(c.nbytes)
            dot_product(
                self.A_gpu,
                self.r_gpu,
                c_gpu,
                np.int32(self.m),
                block=BLOCK_SIZE,
                grid=(grid_size, 1),
            )

            # Copy result to CPU.
            cuda.memcpy_dtoh(c, c_gpu)

            max_idx = np.argmax(c)
            max_correlation = c[max_idx]

            d = self.A[:, max_idx]
            d_gpu = cuda.mem_alloc(d.nbytes)
            cuda.memcpy_htod(d_gpu, d.copy())
            scalar_multiply(
                d_gpu,
                np.float64(max_correlation),
                np.int32(self.n),
                block=BLOCK_SIZE,
                grid=(grid_size, 1),
            )
            vector_add(
                self.s_gpu,
                d_gpu,
                self.s_gpu,
                np.int32(self.m),
                block=BLOCK_SIZE,
                grid=(grid_size, 1),
            )
            vector_subtract(
                self.r_gpu,
                d_gpu,
                self.r_gpu,
                np.int32(self.n),
                block=BLOCK_SIZE,
                grid=(grid_size, 1),
            )

        cuda.memcpy_dtoh(self.b, self.b_gpu)
        cuda.memcpy_dtoh(self.s, self.s_gpu)
        self.cleanup()

    def get_results(self) -> np.ndarray:
        """Return the recovered signal.

        Returns
        -------
        np.ndarray
            The recovered signal.
        """
        return self.s

    def cleanup(self) -> None:
        """Free up the GPU memory allocated.

        Returns
        -------
        None
        """
        self.A_gpu.free()
        self.x_gpu.free()
        self.b_gpu.free()
