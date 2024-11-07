from concurrent.futures import ThreadPoolExecutor

__all__ = [
    "executor",
]

executor = ThreadPoolExecutor()
"""Shared executor for all concurrent tasks."""
