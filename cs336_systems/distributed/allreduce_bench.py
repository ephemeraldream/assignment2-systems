import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.distributed as dist
from datetime import timedelta


def _parse_sizes_mb(sizes_arg: str) -> List[int]:
    if not sizes_arg:
        return [1, 10, 100, 1000]
    sizes: List[int] = []
    for token in sizes_arg.split(","):
        token = token.strip()
        if not token:
            continue
        sizes.append(int(token))
    return sizes


def _dtype_nbytes(dtype: torch.dtype) -> int:
    sizes = {
        torch.float32: 4,
        torch.float64: 8,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int32: 4,
        torch.int64: 8,
    }
    if dtype not in sizes:
        raise ValueError(f"Unsupported dtype for benchmark: {dtype}")
    return sizes[dtype]


def _init_distributed(backend: str) -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    # If not initialized (e.g., when invoked with python -m torch.distributed.run/torchrun), init here.
    if not dist.is_initialized():
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        # Important for NCCL: bind each rank to a specific GPU before init
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available but NCCL backend selected")
            device_index = local_rank % torch.cuda.device_count()
            torch.cuda.set_device(device_index)
        # Shorter timeout to fail fast if something is wrong (e.g., NCCL permissions)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=5))

    # Small safety check
    assert dist.get_world_size() == world_size, "WORLD_SIZE mismatch"
    assert dist.get_rank() == int(os.environ.get("RANK", str(rank))), "RANK mismatch"
    return rank, world_size, local_rank


@torch.no_grad()
def _allocate_tensor(size_mb: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    bytes_total = size_mb * 1024 * 1024
    elem_size = _dtype_nbytes(dtype)
    numel = bytes_total // elem_size
    if numel == 0:
        raise ValueError("Requested size is too small for the chosen dtype")
    return torch.ones(numel, dtype=dtype, device=device)


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _benchmark_single_size(
    tensor: torch.Tensor,
    iterations: int,
    warmup: int,
    device: torch.device,
    op: dist.ReduceOp,
    barrier_each_iter: bool,
) -> List[float]:
    # Warmup
    for _ in range(warmup):
        if barrier_each_iter:
            dist.barrier()
        dist.all_reduce(tensor, op=op, async_op=False)
        _maybe_sync(device)

    latencies_ms: List[float] = []
    # Measure
    dist.barrier()
    for _ in range(iterations):
        if barrier_each_iter:
            dist.barrier()
        _maybe_sync(device)
        t0 = time.perf_counter()
        dist.all_reduce(tensor, op=op, async_op=False)
        _maybe_sync(device)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    dist.barrier()
    return latencies_ms


def _gather_max_across_ranks(latencies_ms: List[float], backend: str, device: torch.device) -> List[float]:
    """Gather per-iteration latencies and take the max across ranks per iteration.

    Using all_gather to avoid Python object collectives for NCCL.
    """
    local = torch.tensor(latencies_ms, dtype=torch.float64, device=(device if backend == "nccl" else torch.device("cpu")))
    world_size = dist.get_world_size()
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    # Move to CPU for reduction
    stacked = torch.stack([t.cpu() for t in gathered], dim=0)
    max_per_iter = torch.max(stacked, dim=0).values
    return max_per_iter.tolist()


def _effective_bandwidth_gbps(bytes_per_rank: int, world_size: int, seconds: float) -> float:
    # For ring all-reduce, each rank transfers 2*(N-1)/N times the tensor size.
    effective_bytes = 2.0 * (world_size - 1) / world_size * bytes_per_rank
    return effective_bytes / seconds / 1e9


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-node multi-process all-reduce benchmark")
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"], help="Distributed backend")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Device for the tensor (default inferred from backend)")
    parser.add_argument("--sizes-mb", type=str, default="1,10,100,1000", help="Comma-separated list of tensor sizes in MB")
    parser.add_argument("--iters", type=int, default=30, help="Number of timed all-reduce iterations per size")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations per size")
    parser.add_argument("--dtype", type=str, default="float32", help="Tensor dtype, e.g., float32")
    parser.add_argument("--no-barrier", action="store_true", help="Disable per-iteration barriers")
    parser.add_argument("--output", type=str, default=None, help="Optional JSONL output file (rank 0 only)")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-JSON logs on non-zero ranks")

    args = parser.parse_args()

    backend = args.backend
    device_str = args.device or ("cuda" if backend == "nccl" else "cpu")
    device = torch.device(device_str)

    dtype = getattr(torch, args.dtype)
    sizes_mb = _parse_sizes_mb(args.sizes_mb)

    rank, world_size, local_rank = _init_distributed(backend)

    # Map each rank to a GPU for NCCL
    if backend == "nccl":
        assert device.type == "cuda", "NCCL requires CUDA device"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank % torch.cuda.device_count())
        else:
            raise RuntimeError("CUDA not available but NCCL backend selected")

    # Limit CPU thread count for more deterministic CPU scaling
    if device.type == "cpu":
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

    op = dist.ReduceOp.SUM
    barrier_each_iter = not args.no_barrier

    out_f = None
    if args.output and rank == 0:
        out_f = open(args.output, "a", buffering=1)

    if not args.quiet and rank == 0:
        print(
            f"Starting all-reduce benchmark | backend={backend} device={device} world_size={world_size} "
            f"sizes_mb={sizes_mb} dtype={dtype} iters={args.iters} warmup={args.warmup}",
            file=sys.stderr,
        )

    results_out = []
    for size_mb in sizes_mb:
        tensor = _allocate_tensor(size_mb=size_mb, device=device, dtype=dtype)

        lat_local = _benchmark_single_size(
            tensor=tensor,
            iterations=args.iters,
            warmup=args.warmup,
            device=device,
            op=op,
            barrier_each_iter=barrier_each_iter,
        )

        # Reduce: take per-iteration max across ranks
        lat_max = _gather_max_across_ranks(lat_local, backend=backend, device=device)

        # Stats on rank 0
        if rank == 0:
            lat_tensor = torch.tensor(lat_max, dtype=torch.float64)
            mean_ms = lat_tensor.mean().item()
            p50_ms = lat_tensor.quantile(0.5).item()
            p95_ms = lat_tensor.quantile(0.95).item()

            bytes_per_rank = size_mb * 1024 * 1024
            bw_gbps = _effective_bandwidth_gbps(bytes_per_rank, world_size, mean_ms / 1000.0)

            record = {
                "backend": backend,
                "device": device_str,
                "world_size": world_size,
                "size_mb": size_mb,
                "dtype": args.dtype,
                "iters": args.iters,
                "warmup": args.warmup,
                "mean_ms": mean_ms,
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
                "bandwidth_GBps": bw_gbps,
            }
            results_out.append(record)
            line = json.dumps(record)
            print(line)
            if out_f is not None:
                out_f.write(line + "\n")

        # Ensure all ranks finished this size before moving on
        dist.barrier()

    if out_f is not None:
        out_f.close()

    # Cleanly tear down process group
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


