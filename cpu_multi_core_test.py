#!/usr/bin/env python3
"""
Multi-Core CPU Performance Test
================================
Measures how well your CPU uses ALL cores in parallel.
Spawns one process per CPU core - shows scaling with core count.
"""

import multiprocessing
import time
import math


def worker_process(duration_seconds: float) -> int:
    """
    Each worker runs this on its own core.
    Same workload as single-core test, but many run in parallel.
    Returns: number of operations completed by this worker
    """
    count = 0
    end_time = time.perf_counter() + duration_seconds

    while time.perf_counter() < end_time:
        x = count * 1.0000001
        _ = math.sqrt(x) + math.sin(x) * math.cos(x) + x**0.5
        count += 1

    return count


def run_worker(duration_seconds: float) -> int:
    """Module-level wrapper - must be picklable for multiprocessing."""
    return worker_process(duration_seconds)


def main():
    num_cores = multiprocessing.cpu_count()

    print("=" * 50)
    print("  MULTI-CORE CPU PERFORMANCE TEST")
    print("=" * 50)
    print(f"\nDetected {num_cores} CPU cores.")
    print("Spawning one worker per core - all run in parallel.\n")

    duration = 5.0  # seconds per worker
    print(f"Each worker runs for {duration} seconds...\n")

    start = time.perf_counter()
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(run_worker, [duration] * num_cores)
    elapsed = time.perf_counter() - start

    total_operations = sum(results)
    ops_per_second = total_operations / elapsed

    print("RESULTS")
    print("-" * 50)
    print(f"  Cores used:        {num_cores}")
    print(f"  Wall-clock time:    {elapsed:.2f} seconds")
    print(f"  Total operations:   {total_operations:,.0f}")
    print(f"  Operations/sec:    {ops_per_second:,.0f}")
    print("-" * 50)
    print("\nHigher operations/sec = better multi-core throughput")
    print("Compare with single-core test to see parallel scaling.")
    print("=" * 50)


if __name__ == "__main__":
    main()
