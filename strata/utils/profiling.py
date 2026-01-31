from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass
class TimingStats:
    """
    Accumulated timing statistics for a named operation. Tracks total time, call count, and 
    computes average duration across all recorded invocations.
    """
    name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0


    @property
    def avg_time(self) -> float:
        """Returns: Average duration in seconds per call"""
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count


    def record(self, duration: float) -> None:
        """Record a timing measurement.

        :param duration: Duration in seconds
        """
        self.total_time += duration
        self.call_count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)


    def reset(self) -> None:
        """Reset all statistics to initial values"""
        self.total_time = 0.0
        self.call_count = 0
        self.min_time = float("inf")
        self.max_time = 0.0


class Timer:
    """Simple timer for measuring execution duration"""

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._elapsed: float = 0.0
        self._running: bool = False


    def start(self) -> None:
        """Start the timer"""
        self._start_time = perf_counter()
        self._running = True


    def stop(self) -> float:
        """Stop the timer and return elapsed time"""
        if self._running:
            self._elapsed = perf_counter() - self._start_time
            self._running = False
        return self._elapsed


    @property
    def elapsed(self) -> float:
        """Get elapsed time (in seconds) without stopping"""
        if self._running:
            return perf_counter() - self._start_time
        return self._elapsed


    def __enter__(self) -> "Timer":
        """Context manager entry"""
        self.start()
        return self


    def __exit__(self, *args) -> None:
        """Context manager exit"""
        self.stop()


@dataclass
class Profiler:
    """Provides named timing contexts and summary reporting for performance analysis"""
    stats: dict[str, TimingStats] = field(default_factory=dict)
    enabled: bool = True


    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        """Time a code block under the given name.

        :param name: Name for the timing category
        :returns: Context manager for timing
        """
        if not self.enabled:
            yield
            return

        if name not in self.stats:
            self.stats[name] = TimingStats(name)
        start = perf_counter()
        yield
        duration = perf_counter() - start
        self.stats[name].record(duration)


    def get_stats(self, name: str) -> TimingStats | None:
        """Get timing statistics for a name.

        :param name: Name of the timing category
        :returns: TimingStats or None if not found
        """
        return self.stats.get(name)


    def reset(self) -> None:
        """Reset all timing statistics"""
        self.stats.clear()


    def summary(self) -> dict[str, dict]:
        """Returns: Dictionary of timing summaries"""
        result = {}
        for name, stats in self.stats.items():
            result[name] = {
                "total_time_ms": stats.total_time * 1000,
                "avg_time_ms": stats.avg_time * 1000,
                "call_count": stats.call_count,
                "min_time_ms": stats.min_time * 1000 if stats.call_count > 0 else 0,
                "max_time_ms": stats.max_time * 1000,
            }
        return result


def compute_throughput(num_tokens: int, duration_seconds: float) -> float:
    """Compute tokens per second throughput.

    :param num_tokens: Number of tokens processed
    :param duration_seconds: Time duration in seconds
    :returns: Tokens per second
    """
    if duration_seconds <= 0:
        return 0.0
    return num_tokens / duration_seconds
