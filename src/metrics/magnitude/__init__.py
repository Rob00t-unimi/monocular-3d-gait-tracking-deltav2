"""Magnitude and amplitude error metrics."""

from src.metrics.magnitude.nrmse import calculate_rmse, calculate_nrmse
from src.metrics.magnitude.rom_error import (
    calculate_rom,
    calculate_rom_error,
    ROMResult,
)

__all__ = [
    "calculate_rmse",
    "calculate_nrmse",
    "calculate_rom",
    "calculate_rom_error",
    "ROMResult",
]
