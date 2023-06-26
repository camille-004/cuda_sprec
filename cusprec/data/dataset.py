"""Dataset definitions."""
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from cusprec.constants import KERNEL_PATH


class Dataset:
    """Interface for dataset definitions."""

    def __init__(self, n: int, m: int, s: int) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        n : int
            Number of original signal samples.
        m : int
            Number of measurements.
        s : int
            Sparsity level.
        """
        self.n = n
        self.m = m
        self.s = s

    def generate_data(self) -> Any:
        """Generate the dataset by executing a CUDA kernel.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.

        Returns
        -------
        Any
            The generated dataset.
        """
        raise NotImplementedError("Subclasses must implement generate_data().")

    def plot(self) -> None:
        """Plot the generated dataset.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement plot().")


class BasicDataset(Dataset):
    """Basic dataset."""

    def __init__(self, n: int, m: int, s: int) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        n : int
            Number of original signal samples.
        m : int
            Number of measurements.
        s : int
            Sparsity level.
        """
        super().__init__(n, m, s)
        self.x = None
        self.A = None
        self.b = None

        with open(KERNEL_PATH / "data" / "basic.cu") as kernel_file:
            kernel = kernel_file.read()

        mod = SourceModule(kernel)
        self.func = mod.get_function("initialize")
        self.generate_data()

    def generate_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the dataset.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The generated dataset consisting of the sparse signal `x`,
            measurement matrix `A`, and observed signal `b`.
        """
        rand_vals = np.random.randn(self.n * (self.m + 1)).astype(np.float32)
        rand_indices = np.random.choice(self.n, self.s, replace=False).astype(
            np.int32
        )
        x = np.zeros(self.n, dtype=np.float32)
        x[rand_indices] = rand_vals[: self.s]

        # Allocate memory.
        # For x, Allocate n * 4, (4 bytes for np.float32).
        num_bytes = np.dtype("float32").itemsize
        x_gpu = cuda.mem_alloc(self.n * num_bytes)
        A_gpu = cuda.mem_alloc(self.m * self.n * num_bytes)
        b_gpu = cuda.mem_alloc(self.m * num_bytes)
        rand_vals_gpu = cuda.mem_alloc(rand_vals.nbytes)
        rand_indices_gpu = cuda.mem_alloc(rand_indices.nbytes)

        # Copy from host to device.
        cuda.memcpy_htod(rand_vals_gpu, rand_vals)
        cuda.memcpy_htod(rand_indices_gpu, rand_indices)

        # Generate data.
        num_blocks = (self.n + 255) // 256
        self.func(
            x_gpu,
            A_gpu,
            b_gpu,
            rand_vals_gpu,
            rand_indices_gpu,
            np.int32(self.m),
            np.int32(self.n),
            np.int32(self.s),
            block=(256, 1, 1),
            grid=(num_blocks, 1),
        )

        # Copy back to CPU.
        x = np.empty(self.n, dtype=np.float32)
        A = np.empty((self.m, self.n), dtype=np.float32)
        b = np.empty(self.m, dtype=np.float32)
        cuda.memcpy_dtoh(x, x_gpu)
        cuda.memcpy_dtoh(A, A_gpu)
        cuda.memcpy_dtoh(b, b_gpu)
        self.x = x
        self.A = A
        self.b = b
        return self.x, self.A, self.b

    def plot(self) -> None:
        """Plot the generated dataset."""
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.stem(self.x)
        plt.title("Sparse Signal (x)")

        plt.subplot(1, 3, 2)
        plt.imshow(self.A, aspect="auto", cmap="viridis")
        plt.colorbar(label="Value")
        plt.title("Measurement Matrix (A)")

        plt.subplot(1, 3, 3)
        plt.stem(self.b)
        plt.title("Observed Signal (b)")

        plt.tight_layout()
        plt.show()
