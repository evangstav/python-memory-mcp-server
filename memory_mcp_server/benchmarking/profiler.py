"""Performance profiling for Memory MCP Server."""

import asyncio
import functools
import statistics
import time
from typing import Any, Dict, List, TypeVar

T = TypeVar("T")


class OperationMetrics:
    """Stores metrics for a specific operation."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.execution_times: List[float] = []
        self.error_count = 0

    def add_execution(self, time_ms: float, success: bool = True):
        """Add an execution record."""
        self.execution_times.append(time_ms)
        if not success:
            self.error_count += 1

    @property
    def avg_time_ms(self) -> float:
        """Get average execution time in milliseconds."""
        return statistics.mean(self.execution_times) if self.execution_times else 0

    @property
    def p95_time_ms(self) -> float:
        """Get 95th percentile execution time in milliseconds."""
        if not self.execution_times:
            return 0
        if len(self.execution_times) < 2:
            return self.execution_times[0] if self.execution_times else 0
        return statistics.quantiles(sorted(self.execution_times), n=20)[-1]

    @property
    def max_time_ms(self) -> float:
        """Get maximum execution time in milliseconds."""
        return max(self.execution_times) if self.execution_times else 0

    @property
    def error_rate(self) -> float:
        """Get error rate as a percentage."""
        if not self.execution_times:
            return 0
        return (self.error_count / len(self.execution_times)) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "operation": self.operation_name,
            "count": len(self.execution_times),
            "avg_time_ms": self.avg_time_ms,
            "p95_time_ms": self.p95_time_ms,
            "max_time_ms": self.max_time_ms,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
        }


class PerformanceProfiler:
    """Tracks performance metrics for memory operations."""

    def __init__(self):
        self.metrics: Dict[str, OperationMetrics] = {}
        self.enabled = False

    def enable(self):
        """Enable profiling."""
        self.enabled = True

    def disable(self):
        """Disable profiling."""
        self.enabled = False

    def clear(self):
        """Clear all metrics."""
        self.metrics = {}

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics as dictionary."""
        return {name: metrics.to_dict() for name, metrics in self.metrics.items()}

    def track(self, operation_name: str):
        """Decorator to track performance of a function."""

        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)

                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    if operation_name not in self.metrics:
                        self.metrics[operation_name] = OperationMetrics(operation_name)
                    self.metrics[operation_name].add_execution(elapsed_ms, success=True)
                    return result
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    if operation_name not in self.metrics:
                        self.metrics[operation_name] = OperationMetrics(operation_name)
                    self.metrics[operation_name].add_execution(
                        elapsed_ms, success=False
                    )
                    raise e

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    if operation_name not in self.metrics:
                        self.metrics[operation_name] = OperationMetrics(operation_name)
                    self.metrics[operation_name].add_execution(elapsed_ms, success=True)
                    return result
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    if operation_name not in self.metrics:
                        self.metrics[operation_name] = OperationMetrics(operation_name)
                    self.metrics[operation_name].add_execution(
                        elapsed_ms, success=False
                    )
                    raise e

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator


# Initialize global profiler
profiler = PerformanceProfiler()
