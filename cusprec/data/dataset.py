"""Dataset definitions."""
from abc import ABC, abstractmethod, abstractproperty
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from cusprec.constants import BLOCK_SIZE, GETTER_ERROR, KERNEL_PATH


class Dataset(ABC):
    """Interface for dataset definitions."""

    @abstractmethod
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
        pass

    @abstractmethod
    def plot(self) -> None:
        """Plot the generated dataset.

        Returns
        -------
        None
        """
        pass

    @property
    @abstractmethod
    def n(self) -> int:
        """Getter method for n.

        Returns
        -------
        int
            Value of n.
        """
        return self._n

    @property
    @abstractmethod
    def m(self) -> int:
        """Getter method for m.

        Returns
        -------
        int
            Value of m.
        """
        return self._m

    @property
    @abstractmethod
    def k(self) -> int:
        """Getter method for k.

        Returns
        -------
        int
            Value of k.
        """
        return self._k

    @property
    @abstractmethod
    def x(self) -> np.ndarray:
        """Getter method for x.

        Returns
        -------
        np.ndarray
            Sparse signal x.
        """
        if self._x is None:
            raise ValueError(GETTER_ERROR)
        return self._x

    @property
    @abstractmethod
    def A(self) -> np.ndarray:
        """Getter method for A.

        Returns
        -------
        np.ndarray
            Measurement matrix A.
        """
        if self._A is None:
            raise ValueError(GETTER_ERROR)
        return self._A

    @property
    @abstractmethod
    def b(self) -> np.ndarray:
        """Getter method for b.

        Returns
        -------
        np.ndarray
            Observed signal b.
        """
        if self._b is None:
            raise ValueError(GETTER_ERROR)
        return self._b


class BasicDataset(Dataset):
    """Basic dataset."""

    def __init__(self, n: int, m: int, k: int) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        n : int
            Number of original signal samples.
        m : int
            Number of measurements.
        k : int
            Sparsity level.
        """
        self._n = n
        self._m = m
        self._k = k
        self._x = None
        self._A = None
        self._b = None

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
        rand_vals = np.random.randn(self._n * (self._m + 1)).astype(np.float32)
        rand_indices = np.random.choice(
            self._n, self._k, replace=False
        ).astype(np.int32)
        x = np.zeros(self._n, dtype=np.float32)
        x[rand_indices] = rand_vals[: self._k]

        # Allocate memory.
        # For x, Allocate n * 4, (4 bytes for np.float32).
        num_bytes = np.dtype("float32").itemsize
        x_gpu = cuda.mem_alloc(self._n * num_bytes)
        A_gpu = cuda.mem_alloc(self._m * self._n * num_bytes)
        b_gpu = cuda.mem_alloc(self._m * num_bytes)
        rand_vals_gpu = cuda.mem_alloc(rand_vals.nbytes)
        rand_indices_gpu = cuda.mem_alloc(rand_indices.nbytes)

        # Copy from host to device.
        cuda.memcpy_htod(rand_vals_gpu, rand_vals)
        cuda.memcpy_htod(rand_indices_gpu, rand_indices)

        # Generate data.
        num_blocks = (self._n + 255) // 256
        self.func(
            x_gpu,
            A_gpu,
            b_gpu,
            rand_vals_gpu,
            rand_indices_gpu,
            np.int32(self._m),
            np.int32(self._n),
            np.int32(self._k),
            block=BLOCK_SIZE,
            grid=(num_blocks, 1),
        )

        # Copy back to CPU.
        x = np.empty(self._n, dtype=np.float32)
        A = np.empty((self._m, self._n), dtype=np.float32)
        b = np.empty(self._m, dtype=np.float32)
        cuda.memcpy_dtoh(x, x_gpu)
        cuda.memcpy_dtoh(A, A_gpu)
        cuda.memcpy_dtoh(b, b_gpu)
        self._x = x
        self._A = A
        self._b = b
        return self._x, self._A, self._b

    def plot(self) -> None:
        """Plot the generated dataset.

        Returns
        -------
        None.
        """
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

    @property
    def n(self) -> int:
        """Getter method for n.

        Returns
        -------
        int
            Value of n.
        """
        return self._n

    @property
    def m(self) -> int:
        """Getter method for m.

        Returns
        -------
        int
            Value of m.
        """
        return self._m

    @property
    def k(self) -> int:
        """Getter method for k.

        Returns
        -------
        int
            Value of k.
        """
        return self._k

    @property
    def x(self) -> np.ndarray:
        """Getter method for x.

        Returns
        -------
        np.ndarray
            Sparse signal x.
        """
        if self._x is None:
            raise ValueError(GETTER_ERROR)
        return self._x

    @property
    def A(self) -> np.ndarray:
        """Getter method for A.

        Returns
        -------
        np.ndarray
            Measurement matrix A.
        """
        if self._A is None:
            raise ValueError(GETTER_ERROR)
        return self._A

    @property
    def b(self) -> np.ndarray:
        """Getter method for b.

        Returns
        -------
        np.ndarray
            Observed signal b.
        """
        if self._b is None:
            raise ValueError(GETTER_ERROR)
        return self._b
