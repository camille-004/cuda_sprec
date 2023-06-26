"""Constants for cuSPREC package."""
from pathlib import Path

PACKAGE_PATH = Path(__file__).parent
KERNEL_PATH = PACKAGE_PATH / "kernels"

GETTER_ERROR = (
    "x, A, and b have not been generated yet. Call " "generate_data() first."
)
