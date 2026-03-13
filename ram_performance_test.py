#!/usr/bin/env python3
"""
RAM Performance Test
===================
Measures memory bandwidth - how fast data can be read from and written to RAM.
Tests: sequential read, sequential write, and copy (read+write).
"""

import time
import sys

try:
    import numpy as np
except ImportError:
    print("NumPy is required for RAM testing. Install with:")
    print("  pip install numpy")
    sys.exit(1)


def bytes_to_gib(bytes_val: int) -> float:
    """Convert bytes to GiB."""
    return bytes_val / (1024**3)


def benchmark_read(arr: np.ndarray, duration_seconds: float) -> tuple[int, float]:
    """
    Sequential read: sum array elements. Forces read of every byte.
    Returns: (bytes_read, elapsed_seconds)
    """
    n = arr.nbytes
    iterations = 0
    start = time.perf_counter()
    end_time = start + duration_seconds

    while time.perf_counter() < end_time:
        _ = np.sum(arr)
        iterations += 1

    elapsed = time.perf_counter() - start
    return iterations * n, elapsed


def benchmark_write(arr: np.ndarray, duration_seconds: float) -> tuple[int, float]:
    """
    Sequential write: fill array. Forces write of every byte.
    Returns: (bytes_written, elapsed_seconds)
    """
    n = arr.nbytes
    iterations = 0
    start = time.perf_counter()
    end_time = start + duration_seconds

    while time.perf_counter() < end_time:
        arr.fill(0xAB)
        iterations += 1

    elapsed = time.perf_counter() - start
    return iterations * n, elapsed


def benchmark_copy(src: np.ndarray, dst: np.ndarray, duration_seconds: float) -> tuple[int, float]:
    """
    Copy: read from src, write to dst. Measures combined read+write bandwidth.
    Returns: (bytes_copied, elapsed_seconds)
    """
    n = src.nbytes
    iterations = 0
    start = time.perf_counter()
    end_time = start + duration_seconds

    while time.perf_counter() < end_time:
        np.copyto(dst, src)
        iterations += 1

    elapsed = time.perf_counter() - start
    return iterations * n, elapsed


def main():
    # Use ~1 GiB - large enough to exceed typical CPU cache
    size_gib = 1.0
    size_bytes = int(size_gib * 1024**3)

    print("=" * 50)
    print("  RAM PERFORMANCE TEST")
    print("=" * 50)
    print(f"\nBuffer size: {size_gib} GiB ({size_bytes:,} bytes)")
    print("Tests sequential access - reflects RAM bandwidth.\n")

    duration = 3.0  # seconds per test
    print(f"Each test runs for {duration} seconds...\n")

    # Allocate buffers (touch memory so it's actually resident)
    print("Allocating buffers...")
    arr = np.zeros(size_bytes // 8, dtype=np.float64)  # 8 bytes per float
    arr.fill(1.0)
    dst = np.empty_like(arr)

    results = []

    # Read
    print("  Read test (sum)...")
    bytes_ops, elapsed = benchmark_read(arr, duration)
    bw_gibs = bytes_to_gib(bytes_ops) / elapsed
    results.append(("Read", bw_gibs, bytes_ops, elapsed))

    # Write
    print("  Write test (fill)...")
    bytes_ops, elapsed = benchmark_write(arr, duration)
    bw_gibs = bytes_to_gib(bytes_ops) / elapsed
    results.append(("Write", bw_gibs, bytes_ops, elapsed))

    # Copy
    print("  Copy test (read+write)...")
    bytes_ops, elapsed = benchmark_copy(arr, dst, duration)
    bw_gibs = bytes_to_gib(bytes_ops) / elapsed
    results.append(("Copy", bw_gibs, bytes_ops, elapsed))

    print("\nRESULTS")
    print("-" * 50)
    for name, bw_gibs, bytes_ops, elapsed in results:
        print(f"  {name:6}  {bw_gibs:6.1f} GiB/s  ({elapsed:.2f}s)")
    print("-" * 50)
    print("\nHigher GiB/s = better RAM bandwidth")
    print("=" * 50)


if __name__ == "__main__":
    main()
