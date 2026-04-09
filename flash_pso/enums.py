"""Type-safe enumerations for FlashPSO configuration.

IntEnum values pass directly to Triton kernels as constexpr integers.
"""
from enum import IntEnum, Enum


class OptionType(IntEnum):
    CALL = 0
    PUT = 1


class ExerciseStyle(IntEnum):
    """How exercise boundaries are parameterized.

    SCALAR:    One boundary per timestep on the basket price.
               PSO dims = num_time_steps.
    PER_ASSET: One boundary per asset per timestep.
               PSO dims = num_assets * num_time_steps.
    """
    SCALAR = 0
    PER_ASSET = 1


class OptionStyle(str, Enum):
    STANDARD = "standard"
    ASIAN = "asian"
    BASKET = "basket"


class RNGType(str, Enum):
    """Random number generation method for MC path simulation.

    PHILOX:  Philox counter-based RNG + Box-Muller. Supports compute-on-the-fly.
    SOBOL:   Scrambled Sobol quasi-random + inverse CDF. Precompute-only.
    """
    PHILOX = "philox"
    SOBOL = "sobol"
