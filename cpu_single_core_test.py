#!/usr/bin/env python3
"""
Single-Core CPU Performance Test
================================
Measures how fast your CPU can execute compute-heavy tasks on ONE core.
Uses a single thread - no parallelism. Good for tasks that can't be parallelized.
"""

import time
import math


def cpu_intensive_task(duration_seconds: float = 5.0) -> float:
    """
    Run CPU-intensive math operations for a set duration.
    Uses floating-point and integer math - common in real workloads.
    Returns: operations per second (higher = better single-core performance)
    """
    count = 0
    end_time = time.perf_counter() + duration_seconds

    while time.perf_counter() < end_time:
        # Mix of operations: sqrt, sin, cos, power - keeps CPU busy
        x = count * 1.0000001
        _ = math.sqrt(x) + math.sin(x) * math.cos(x) + x**0.5
        count += 1

    return count


def main():
    print("=" * 50)
    print("  SINGLE-CORE CPU PERFORMANCE TEST")
    print("=" * 50)
    print("\nThis test runs on ONE core only (single thread).")
    print("It measures raw compute speed for sequential tasks.\n")

    duration = 5.0  # seconds to run
    print(f"Running for {duration} seconds...\n")

    start = time.perf_counter()
    operations = cpu_intensive_task(duration)
    elapsed = time.perf_counter() - start

    ops_per_second = operations / elapsed

    print("RESULTS")
    print("-" * 50)
    print(f"  Total operations:  {operations:,.0f}")
    print(f"  Time elapsed:      {elapsed:.2f} seconds")
    print(f"  Operations/sec:    {ops_per_second:,.0f}")
    print("-" * 50)
    print("\nHigher operations/sec = better single-core performance")
    print("=" * 50)


if __name__ == "__main__":
    main()
