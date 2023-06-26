"""Base class for sparse recovery algorithm."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

from cusprec.data.dataset import Dataset


class AlgorithmType(Enum):
    """Enumeration of algorithm types."""

    MATCHING_PURSUIT = 1
    CONVEX_RELAXATION = 2


class Algorithm(ABC):
    """Base class for sparse recovery algorithms.

    This class serves as a base for implementing sparse recovery algorithms.
    It defines the common interface and behavior expected from derived
    algorithm classes.

    Parameters
    ----------
    kernel : cuda.Function
        The CUDA kernel function for algorithm execution.
    dataset : Dataset
        The input dataset for the algorithm.
    algorithm_type : AlgorithmType
        The type of the algorithm.
    params : dict[str, Any]
        Additional parameters for configuring the algorithm.

    Attributes
    ----------
    kernel : cuda.Function
        The CUDA kernel function for algorithm execution.
    dataset : Dataset
        The input dataset for the algorithm.
    algorithm_type : AlgorithmType
        The type of the algorithm.
    params : dict[str, Any]
        Additional parameters for configuring the algorithm.
    """

    def __init__(
        self,
        kernel: cuda.Function,
        dataset: Dataset,
        algorithm_type: AlgorithmType,
        params: dict[str, Any],
    ) -> None:
        self.kernel = kernel
        self.dataset = dataset
        self.algorithm_type = algorithm_type
        self.params = params

    @abstractmethod
    def exec(self) -> None:
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
