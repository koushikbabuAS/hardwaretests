#!/usr/bin/env python3
"""
GPU Performance Test
====================
Measures how fast your GPU can execute compute-heavy tasks.
Supports: NVIDIA (CUDA) and Apple Silicon (MPS).
Uses matrix multiplication - a common GPU workload.
"""

import time
import sys

try:
    import torch
except ImportError:
    print("PyTorch is required for GPU testing. Install with:")
    print("  pip install torch")
    sys.exit(1)


def get_device() -> tuple[str, torch.device]:
    """Detect best available GPU. Returns (device_name, device)."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"NVIDIA {name}", torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple Silicon (MPS)", torch.device("mps")
    return None, None


def gpu_benchmark(device: torch.device, duration_seconds: float = 5.0) -> tuple[int, float]:
    """
    Run matrix multiplications on GPU for a set duration.
    Matrix multiply is highly parallel - ideal for GPU.
    Returns: (total_flops, elapsed_seconds)
    """
    # Matrix size: 4096x4096 = ~67M elements per matmul
    # Each matmul does ~2*n^3 FLOPs (n^3 mults + n^3 adds)
    n = 4096
    a = torch.randn(n, n, device=device, dtype=torch.float32)
    b = torch.randn(n, n, device=device, dtype=torch.float32)

    # Warmup - GPU needs a few runs to reach full speed
    for _ in range(3):
        _ = torch.matmul(a, b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    flops_per_matmul = 2 * (n * n * n)  # ~137 billion FLOPs per multiply
    count = 0
    start = time.perf_counter()
    end_time = start + duration_seconds

    # Sync inside loop - GPU ops are async; without sync we'd queue work
    # and the "5 sec" would only measure CPU queueing, not actual GPU time
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    while time.perf_counter() < end_time:
        _ = torch.matmul(a, b)
        sync()
        count += 1

    elapsed = time.perf_counter() - start
    total_flops = count * flops_per_matmul
    return total_flops, elapsed


def main():
    print("=" * 50)
    print("  GPU PERFORMANCE TEST")
    print("=" * 50)

    device_name, device = get_device()
    if device is None:
        print("\nNo GPU detected. This script requires:")
        print("  - NVIDIA GPU with CUDA, or")
        print("  - Apple Silicon (M1/M2/M3) with MPS")
        print("\nFalling back to CPU (torch) for comparison...")
        device_name = "CPU"
        device = torch.device("cpu")

    print(f"\nDevice: {device_name}")
    print("Workload: 4096×4096 matrix multiplication (float32)\n")

    duration = 5.0
    print(f"Running for {duration} seconds...\n")

    total_flops, elapsed = gpu_benchmark(device, duration)

    gflops = total_flops / elapsed / 1e9  # billions of FLOPs per second
    tflops = total_flops / elapsed / 1e12  # trillions of FLOPs per second

    print("RESULTS")
    print("-" * 50)
    print(f"  Matrix multiplies:  {int(total_flops / (2 * 4096**3)):,}")
    print(f"  Total FLOPs:        {total_flops / 1e12:.2f} trillion")
    print(f"  Time elapsed:       {elapsed:.2f} seconds")
    print(f"  Throughput:         {gflops:,.0f} GFLOPS  ({tflops:.2f} TFLOPS)")
    print("-" * 50)
    print("\nHigher GFLOPS/TFLOPS = better GPU performance")
    print("=" * 50)


if __name__ == "__main__":
    main()
