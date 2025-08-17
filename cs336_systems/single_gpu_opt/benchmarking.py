import timeit
import torch
import argparse
import os
from logging import getLogger
from cs336_basics.cs336_basics.model import BasicsTransformerLM

logger = getLogger(__name__)


def setup_torch_compile_env():
    """Setup environment variables for torch.compile debugging."""
    env_vars = {
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCH_LOGS": "+dynamo,inductor,graph_breaks", 
        "TORCHDYNAMO_VERBOSE": "1",
        "TORCHINDUCTOR_REPRO_LEVEL": "3",
        "TORCHINDUCTOR_REPRO_DIR": "./ti_repro",
        # Make Inductor more conservative
        "TORCHINDUCTOR_MAX_AUTOTUNE": "0",
        "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM": "0",
        "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE": "0",
        "TORCHINDUCTOR_MAX_AUTOTUNE_CONV": "0"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value



def generate_random_batch(batch_size: int, seq_len: int, vocab_size: int, device: str = "cuda") -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


def benchmarking_script(
    num_layers: int, 
    d_model: int, 
    num_heads: int, 
    d_ff: int, 
    context_length: int, 
    vocab_size: int, 
    batch_size: int, 
    device: str = "cuda",
    warmup_steps: int = 10,
    num_steps: int = 100,
    rope_theta: float = 10000.0,
    profile_memory: bool = False,
    torch_compile: bool = False
):
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    ).to(device)
    
    model.train()
    
    if torch_compile:
        logger.info("Setting up torch.compile environment variables...")
        setup_torch_compile_env()
        logger.info("Compiling model with torch.compile...")
        # Enable TF32 matmul for better perf and stability on Ampere+
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        # Compile with conservative options; allow backend override via env
        backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
        model = torch.compile(
            model,
            backend=backend,
            fullgraph=True,
            dynamic=False,
            mode="reduce-overhead",
        )
        logger.info("Model compilation completed")
    
    input_ids = generate_random_batch(batch_size, context_length, vocab_size, device)
    
    logger.info(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
    logger.info(f"Benchmarking with batch_size={batch_size}, context_length={context_length}, device={device}")
    
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    
    # Memory profiling setup
    if profile_memory and device.startswith("cuda"):
        logger.info("Starting memory profiling...")
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    logger.info(f"Running {warmup_steps} warmup steps...")
    for _ in range(warmup_steps):
        outputs = model(input_ids)
        loss = outputs.sum()
        loss.backward()
        model.zero_grad()
        
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    
    def forward_pass():
        with torch.no_grad():
            outputs = model(input_ids)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return outputs
    
    logger.info(f"Timing forward pass for {num_steps} steps...")
    forward_time = timeit.timeit(forward_pass, number=num_steps)
    forward_time_per_step = forward_time / num_steps
    
    def forward_backward_pass():
        outputs = model(input_ids)
        
        loss = outputs.sum()
        loss.backward()
        
        model.zero_grad()
        
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    
    logger.info(f"Timing forward + backward pass for {num_steps} steps...")
    forward_backward_time = timeit.timeit(forward_backward_pass, number=num_steps)
    forward_backward_time_per_step = forward_backward_time / num_steps
    
    backward_time_per_step = forward_backward_time_per_step - forward_time_per_step
    
    # Memory profiling results
    memory_info = {}
    if profile_memory and device.startswith("cuda"):
        logger.info("Saving memory snapshot...")
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        
        # Get memory stats
        memory_info = {
            "max_memory_allocated": torch.cuda.max_memory_allocated() / 1e9,  # GB
            "max_memory_reserved": torch.cuda.max_memory_reserved() / 1e9,    # GB
            "current_memory_allocated": torch.cuda.memory_allocated() / 1e9,   # GB
            "current_memory_reserved": torch.cuda.memory_reserved() / 1e9      # GB
        }
        
        logger.info(f"Max GPU memory allocated: {memory_info['max_memory_allocated']:.2f} GB")
        logger.info(f"Max GPU memory reserved: {memory_info['max_memory_reserved']:.2f} GB")
        logger.info("Memory snapshot saved to: memory_snapshot.pickle")
        logger.info("Open https://pytorch.org/memory_viz to analyze the snapshot")
    
    results = {
        "forward_time_total": forward_time,
        "forward_time_per_step": forward_time_per_step,
        "forward_backward_time_total": forward_backward_time,
        "forward_backward_time_per_step": forward_backward_time_per_step,
        "backward_time_per_step": backward_time_per_step,
        "steps": num_steps,
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "device": device,
        "model_params": model.get_num_params(),
        "memory_info": memory_info
    }
    
    logger.info("=== Benchmarking Results ===")
    logger.info(f"Forward pass: {forward_time_per_step*1000:.3f} ms/step")
    logger.info(f"Backward pass: {backward_time_per_step*1000:.3f} ms/step")
    logger.info(f"Forward + Backward: {forward_backward_time_per_step*1000:.3f} ms/step")
    logger.info(f"Tokens/sec (forward): {(batch_size * context_length) / forward_time_per_step:.0f}")
    logger.info(f"Tokens/sec (forward+backward): {(batch_size * context_length) / forward_backward_time_per_step:.0f}")
    
    return results


def main():
    """Main function to run benchmarking with command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark transformer model forward and backward passes")
    
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--context_length", type=int, default=1024, help="Maximum context length")
    parser.add_argument("--vocab_size", type=int, default=50000, help="Vocabulary size")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")
    
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for benchmarking")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to benchmark")
    parser.add_argument("--profile_memory", action="store_true", help="Enable memory profiling")
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for model optimization")
    
    args = parser.parse_args()
    
    results = benchmarking_script(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        device=args.device,
        warmup_steps=args.warmup_steps,
        num_steps=args.num_steps,
        rope_theta=args.rope_theta,
        profile_memory=args.profile_memory,
        torch_compile=args.torch_compile
    )
    
    return results


if __name__ == "__main__":
    main()
    