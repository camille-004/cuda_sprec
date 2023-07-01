"""Constants for cuSPREC package."""
from pathlib import Path

PACKAGE_PATH: Path = Path(__file__).parent
KERNEL_PATH: Path = PACKAGE_PATH / "kernels"

THREADS_PER_BLOCK: int = 256
BLOCK_SIZE: tuple = (256, 1, 1)

GETTER_ERROR: str = (
    "x, A, and b have not been generated yet. Call " "generate_data() first."
)
