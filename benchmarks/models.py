from dataclasses import dataclass
from enum import Enum, auto

class Method(Enum):
    FLASH_PSO = auto()
    OPENCL_PSO = auto()
    OPENCL_LSMC = auto()
    QUANTLIB = auto()
    NATIVE_BINOMIAL = auto()
    FLASH_PSO_CPU = auto()

@dataclass
class BenchmarkResult:
    name: str
    method: Method
    runs: int
    mean_price: float
    bias: float
    std_dev: float
    std_error: float
    rmse: float
    mean_iters: float           # <--- Added
    mean_init_time_ms: float
    mean_exec_time_ms: float
    mean_iter_time_ms: float
    mean_wall_time_s: float     # <--- Changed to Mean Per-Run
