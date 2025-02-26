"""Tests for performance profiler."""

import asyncio
import time

import pytest

from memory_mcp_server.benchmarking.profiler import (
    OperationMetrics,
    PerformanceProfiler,
)


def test_operation_metrics():
    metrics = OperationMetrics("test_operation")

    # Add successful executions
    metrics.add_execution(10.5)
    metrics.add_execution(15.2)
    metrics.add_execution(12.8)

    # Add failed execution
    metrics.add_execution(20.1, success=False)

    # Check metrics
    assert metrics.operation_name == "test_operation"
    assert len(metrics.execution_times) == 4
    assert metrics.error_count == 1

    # Check calculated stats
    assert round(metrics.avg_time_ms, 1) == 14.7
    assert round(metrics.max_time_ms, 1) == 20.1
    assert round(metrics.error_rate, 1) == 25.0

    # Test to_dict method
    metrics_dict = metrics.to_dict()
    assert metrics_dict["operation"] == "test_operation"
    assert metrics_dict["count"] == 4
    assert metrics_dict["error_count"] == 1
    assert round(metrics_dict["avg_time_ms"], 1) == 14.7


def test_profiler_initialization():
    profiler = PerformanceProfiler()

    # Initial state
    assert profiler.enabled == False
    assert len(profiler.metrics) == 0

    # Enable/disable
    profiler.enable()
    assert profiler.enabled == True

    profiler.disable()
    assert profiler.enabled == False


def test_profiler_synchronous_tracking():
    profiler = PerformanceProfiler()
    profiler.enable()

    # Define a test function
    @profiler.track("sync_operation")
    def test_function(x, y):
        time.sleep(0.01)  # Small delay to have measurable execution time
        return x + y

    # Call function
    result = test_function(5, 3)

    # Check result and metrics
    assert result == 8
    assert "sync_operation" in profiler.metrics
    assert profiler.metrics["sync_operation"].operation_name == "sync_operation"
    assert len(profiler.metrics["sync_operation"].execution_times) == 1
    assert profiler.metrics["sync_operation"].error_count == 0

    # Call again
    test_function(10, 20)

    # Check updated metrics
    assert len(profiler.metrics["sync_operation"].execution_times) == 2

    # Test error tracking
    @profiler.track("error_operation")
    def error_function():
        time.sleep(0.01)
        raise ValueError("Test error")

    # Call error function
    try:
        error_function()
    except ValueError:
        pass

    # Check error metrics
    assert "error_operation" in profiler.metrics
    assert profiler.metrics["error_operation"].error_count == 1

    # When profiler is disabled, it doesn't track
    profiler.disable()
    test_function(1, 2)
    assert (
        len(profiler.metrics["sync_operation"].execution_times) == 2
    )  # Still 2, not 3


@pytest.mark.asyncio
async def test_profiler_asynchronous_tracking():
    profiler = PerformanceProfiler()
    profiler.enable()

    # Define a test async function
    @profiler.track("async_operation")
    async def test_async_function(x, y):
        await asyncio.sleep(0.01)  # Small delay to have measurable execution time
        return x * y

    # Call async function
    result = await test_async_function(4, 5)

    # Check result and metrics
    assert result == 20
    assert "async_operation" in profiler.metrics
    assert profiler.metrics["async_operation"].operation_name == "async_operation"
    assert len(profiler.metrics["async_operation"].execution_times) == 1
    assert profiler.metrics["async_operation"].error_count == 0

    # Test error tracking
    @profiler.track("async_error")
    async def async_error_function():
        await asyncio.sleep(0.01)
        raise ValueError("Test async error")

    # Call error function
    try:
        await async_error_function()
    except ValueError:
        pass

    # Check error metrics
    assert "async_error" in profiler.metrics
    assert profiler.metrics["async_error"].error_count == 1


def test_profiler_get_metrics():
    profiler = PerformanceProfiler()
    profiler.enable()

    @profiler.track("test_op")
    def test_function():
        time.sleep(0.01)
        return True

    # Call function
    test_function()

    # Get metrics
    metrics = profiler.get_metrics()

    # Check metrics format
    assert "test_op" in metrics
    assert isinstance(metrics["test_op"], dict)
    assert metrics["test_op"]["operation"] == "test_op"
    assert metrics["test_op"]["count"] == 1
    assert metrics["test_op"]["error_count"] == 0

    # Check metric values - ensure they exist but don't validate specific values
    # which could be variable depending on system performance
    assert "avg_time_ms" in metrics["test_op"]
    assert "p95_time_ms" in metrics["test_op"]
    assert "max_time_ms" in metrics["test_op"]
    assert "error_rate" in metrics["test_op"]

    # Clear metrics
    profiler.clear()
    assert len(profiler.get_metrics()) == 0
