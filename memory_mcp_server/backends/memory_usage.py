"""Memory usage tracking for backend operations."""

import os

import psutil


class MemoryUsageTracker:
    """Track memory usage of the application."""

    @staticmethod
    def get_current_usage_mb() -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    @staticmethod
    def get_system_memory_mb() -> float:
        """Get total system memory in MB."""
        return psutil.virtual_memory().total / (1024 * 1024)

    @staticmethod
    def get_memory_usage_percent() -> float:
        """Get memory usage as percentage of system memory."""
        process = psutil.Process(os.getpid())
        memory_used = process.memory_info().rss
        total_memory = psutil.virtual_memory().total
        return (memory_used / total_memory) * 100
