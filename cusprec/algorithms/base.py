"""Base class for sparse recovery algorithm."""
from abc import ABC, abstractmethod

import numpy as np
from pycuda.compiler import SourceModule

from cusprec.constants import KERNEL_PATH


class Algorithm(ABC):
    """Base class for sparse recovery algorithms.

    This class serves as a base for implementing sparse recovery algorithms.
    It defines the common interface and behavior expected from derived
    algorithm classes.
    """

    def read_kernel(self, kernel_name: str) -> SourceModule:
        """Read the CUDA kernel and wrap it in a SourceModule.

        Parameters
        ----------
        kernel_name : str
            Name of the kernel file.

        Returns
        -------
        SourceModule
            SourceModule containing the CUDA kernel.
        """
        with open(KERNEL_PATH / "algorithms" / kernel_name) as kernel_file:
            kernel = kernel_file.read()
        return SourceModule(kernel)

    @abstractmethod
    def execute(self) -> None:
        """Execute the sparse recovery algorithm.

        This method executes the sparse recovery algorithm using the provided
        input data and configuration. The implementation of this method
        should define the specific logic for executing the algorithm.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def get_results(self) -> np.ndarray:
        """Retrieve the results of the sparse recovery algorithm.

        This method retrieves the results of the algorithm after it has been
        executed. The format and contents of the results depend on the
        specific implementation.

        Returns
        -------
        np.ndarray
            The results of the algorithm.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Perform cleanup tasks after executing the algorithm.

        This method performs any necessary cleanup tasks after the algorithm
        has been executed. It can be used to release resources or reset
        internal state.

        Returns
        -------
        None
        """
        pass
