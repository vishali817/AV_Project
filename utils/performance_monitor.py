import time
import psutil
import os
from dataclasses import dataclass, field
from typing import Dict

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()

    def elapsed_ms(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

def get_cpu_usage() -> float:
    """Returns the CPU usage percentage."""
    return psutil.cpu_percent(interval=None)

def get_memory_usage() -> float:
    """Returns the current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)
