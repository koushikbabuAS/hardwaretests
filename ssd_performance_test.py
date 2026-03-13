#!/usr/bin/env python3
"""
SSD Performance Test
===================
Measures storage throughput - how fast data can be written to and read from disk.
Tests: sequential write, sequential read.
Uses fsync on writes so data actually hits the drive (not just OS cache).
"""

import os
import tempfile
import time


def bytes_to_mib(bytes_val: int) -> float:
    """Convert bytes to MiB."""
    return bytes_val / (1024**2)


def benchmark_write(path: str, block_size: int, duration_seconds: float) -> tuple[int, float]:
    """
    Sequential write with fsync. Measures true disk write speed.
    Returns: (bytes_written, elapsed_seconds)
    """
    block = os.urandom(block_size)  # Random data - prevents compression shortcuts
    total_bytes = 0
    start = time.perf_counter()
    end_time = start + duration_seconds

    with open(path, "wb") as f:
        while time.perf_counter() < end_time:
            f.write(block)
            f.flush()
            os.fsync(f.fileno())
            total_bytes += block_size

    elapsed = time.perf_counter() - start
    return total_bytes, elapsed


def benchmark_read(path: str, block_size: int, duration_seconds: float) -> tuple[int, float]:
    """
    Sequential read. Measures disk read speed.
    Returns: (bytes_read, elapsed_seconds)
    """
    total_bytes = 0
    start = time.perf_counter()
    end_time = start + duration_seconds

    with open(path, "rb") as f:
        while time.perf_counter() < end_time:
            block = f.read(block_size)
            if not block:
                f.seek(0)  # Loop back to start
                continue
            total_bytes += len(block)

    elapsed = time.perf_counter() - start
    return total_bytes, elapsed


def main():
    block_size = 1024 * 1024  # 1 MiB blocks
    duration = 5.0

    print("=" * 50)
    print("  SSD PERFORMANCE TEST")
    print("=" * 50)
    print(f"\nBlock size: {block_size // 1024} KiB")
    print(f"Each test runs for {duration} seconds")
    print("Write test uses fsync - measures actual disk speed.\n")

    # Use current directory - tests the drive you're running from
    test_dir = os.getcwd()
    test_path = os.path.join(test_dir, ".ssd_benchmark_temp")

    try:
        # Write test (creates file)
        print("  Write test...")
        bytes_written, elapsed = benchmark_write(test_path, block_size, duration)
        write_mibs = bytes_to_mib(bytes_written) / elapsed
        print(f"    Wrote {bytes_to_mib(bytes_written):.1f} MiB in {elapsed:.2f}s")

        # Read test
        print("  Read test...")
        bytes_read, elapsed = benchmark_read(test_path, block_size, duration)
        read_mibs = bytes_to_mib(bytes_read) / elapsed
        print(f"    Read {bytes_to_mib(bytes_read):.1f} MiB in {elapsed:.2f}s")

        print("\nRESULTS")
        print("-" * 50)
        print(f"  Write:  {write_mibs:6.1f} MiB/s")
        print(f"  Read:   {read_mibs:6.1f} MiB/s")
        print("-" * 50)
        print("\nHigher MiB/s = better SSD performance")
        print(f"Tested: {test_dir}")
        print("=" * 50)

    finally:
        if os.path.exists(test_path):
            os.remove(test_path)


if __name__ == "__main__":
    main()
