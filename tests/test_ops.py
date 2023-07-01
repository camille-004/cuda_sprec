"""Unit tests for CUDA kernel operations."""
import unittest

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from cusprec.constants import BLOCK_SIZE, KERNEL_PATH

N = 256
SIZE = 1000

grid = N + BLOCK_SIZE[0] - 1 // BLOCK_SIZE[0]
with open(KERNEL_PATH / "algorithms" / "ops.cu") as kernel_file:
    kernel = kernel_file.read()


class TestKernelOps(unittest.TestCase):
    """Test kernel operations."""

    def setUp(self) -> None:
        """Set up kernel functions."""
        mod = SourceModule(kernel)
        self.dot_product = mod.get_function("dot")
        self.add = mod.get_function("add")
        self.subtract = mod.get_function("sub")
        self.scalar_multiply = mod.get_function("scalarMultiply")

    def test_dot_product(self):
        """Test dot product kernel."""
        a = np.random.randint(0, SIZE, N).astype(np.int32)
        b = np.random.randint(0, SIZE, N).astype(np.int32)
        c = np.empty(1, dtype=np.int32)

        dev_a = cuda.mem_alloc(a.nbytes)
        dev_b = cuda.mem_alloc(b.nbytes)
        dev_c = cuda.mem_alloc(c.nbytes)

        cuda.memcpy_htod(dev_a, a)
        cuda.memcpy_htod(dev_b, b)

        self.dot_product(dev_a, dev_b, dev_c, block=BLOCK_SIZE, grid=(grid, 1))
        cuda.memcpy_dtoh(c, dev_c)

        self.assertEqual(c[0], np.dot(a, b))

    def test_vector_add(self):
        """Test vector addition kernel."""
        a = np.random.randint(0, SIZE, N).astype(np.int32)
        b = np.random.randint(0, SIZE, N).astype(np.int32)
        c = np.empty(N, dtype=np.int32)

        dev_a = cuda.mem_alloc(a.nbytes)
        dev_b = cuda.mem_alloc(b.nbytes)
        dev_c = cuda.mem_alloc(c.nbytes)

        cuda.memcpy_htod(dev_a, a)
        cuda.memcpy_htod(dev_b, b)

        self.add(dev_a, dev_b, dev_c, block=BLOCK_SIZE, grid=(grid, 1))
        cuda.memcpy_dtoh(c, dev_c)

        self.assertTrue(np.array_equal(c, np.add(a, b)))

    def test_vector_sub(self):
        """Test vector subtraction kernel."""
        a = np.random.randint(0, SIZE, N).astype(np.int32)
        b = np.random.randint(0, SIZE, N).astype(np.int32)
        c = np.empty(N, dtype=np.int32)

        dev_a = cuda.mem_alloc(a.nbytes)
        dev_b = cuda.mem_alloc(b.nbytes)
        dev_c = cuda.mem_alloc(c.nbytes)

        cuda.memcpy_htod(dev_a, a)
        cuda.memcpy_htod(dev_b, b)

        self.subtract(dev_a, dev_b, dev_c, block=BLOCK_SIZE, grid=(grid, 1))
        cuda.memcpy_dtoh(c, dev_c)

        self.assertTrue(np.array_equal(c, np.subtract(a, b)))

    def test_scalar_multiply(self):
        """Test scalar-vector multiplication kernel."""
        scalar = 2.5
        a = np.random.randint(0, SIZE, N).astype(np.float32)

        dev_a = cuda.mem_alloc(a.nbytes)
        dev_c = cuda.mem_alloc(a.nbytes)

        cuda.memcpy_htod(dev_a, a)

        self.scalar_multiply(
            dev_a, np.float32(scalar), dev_c, block=BLOCK_SIZE, grid=(grid, 1)
        )
        c = np.empty(N, dtype=np.float32)
        cuda.memcpy_dtoh(c, dev_c)

        self.assertTrue(np.array_equal(c, a * scalar))
