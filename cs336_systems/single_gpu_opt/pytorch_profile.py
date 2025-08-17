#!/usr/bin/env python3

import torch
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from cs336_systems.single_gpu_opt.benchmarking import benchmarking_script
import argparse


def pytorch_profile_warmup():
    print("=== PyTorch Профилирование с warmup (5 шагов) ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        schedule=schedule(wait=1, warmup=2, active=3, repeat=1)
    ) as prof:
        with record_function("warmup_benchmark"):
            results = benchmarking_script(
                num_layers=6,
                d_model=512,
                num_heads=8,
                d_ff=2048,
                context_length=1024,
                vocab_size=32000,
                batch_size=4,
                seq_len=256,
                device="cuda" if torch.cuda.is_available() else "cpu",
                warmup_steps=5,
                num_steps=6,  
                rope_theta=10000.0
            )
            prof.step()
    
    prof.export_chrome_trace("warmup_pytorch_trace.json")
    prof.export_stacks("warmup_pytorch_stacks.txt", "self_cuda_time_total")
    
    print(f"Forward time per step: {results['forward_time_per_step']*1000:.3f} ms")
    
    print("\n=== Топ CUDA операций ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    return results


def pytorch_profile_no_warmup(profile_memory=False):
    print("=== PyTorch Профилирование без warmup ===")
    
    from cs336_basics.cs336_basics.model import BasicsTransformerLM
    
    model = BasicsTransformerLM(
        vocab_size=32000,
        context_length=1024,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        rope_theta=10000.0
    ).to("cuda")
    
    model.train()
    input_ids = torch.randint(0, 32000, (4, 256), device="cuda")
    
    # Memory profiling setup
    if profile_memory:
        print("Starting memory profiling...")
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,  
        profile_memory=True,
        with_stack=False,     
        with_flops=False      
    ) as prof:
        with record_function("model_forward"):
            outputs = model(input_ids)
            loss = outputs.sum()
            
        with record_function("model_backward"):
            loss.backward()
            
        torch.cuda.synchronize()
    
    # Memory profiling results
    if profile_memory:
        print("Saving memory snapshot...")
        torch.cuda.memory._dump_snapshot("pytorch_memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        
        memory_info = {
            "max_memory_allocated": torch.cuda.max_memory_allocated() / 1e9,
            "max_memory_reserved": torch.cuda.max_memory_reserved() / 1e9,
            "current_memory_allocated": torch.cuda.memory_allocated() / 1e9,
            "current_memory_reserved": torch.cuda.memory_reserved() / 1e9
        }
        
        print(f"\n=== Memory Usage ===")
        print(f"Max GPU memory allocated: {memory_info['max_memory_allocated']:.2f} GB")
        print(f"Max GPU memory reserved: {memory_info['max_memory_reserved']:.2f} GB")
        print(f"Current GPU memory allocated: {memory_info['current_memory_allocated']:.2f} GB")
        print(f"Current GPU memory reserved: {memory_info['current_memory_reserved']:.2f} GB")
        print("Memory snapshot saved to: pytorch_memory_snapshot.pickle")
        print("Open https://pytorch.org/memory_viz to analyze the snapshot")
    
    try:
        prof.export_chrome_trace("no_warmup_pytorch_trace_compact.json")
        print("Compact Chrome trace saved successfully")
    except Exception as e:
        print(f"Error saving Chrome trace: {e}")
        
    try:
        prof.export_stacks("no_warmup_pytorch_stacks.txt", "self_cuda_time_total")
        print("Stack traces saved successfully")
    except Exception as e:
        print(f"Error saving stacks: {e}")
    
    print("\n=== Топ CUDA операций ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    print("\n=== Топ операций по CPU времени ===") 
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    print("\n=== Kernel Summary ===")
    cuda_ops = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    print("Chrome trace saved to: no_warmup_pytorch_trace.json")
    print("Stack traces saved to: no_warmup_pytorch_stacks.txt")
    
    return {"cuda_ops": cuda_ops}


def main():
    parser = argparse.ArgumentParser(description="Profile with PyTorch Profiler")
    parser.add_argument("--mode", choices=["warmup", "no-warmup", "both"], default="both",
                        help="Режим профилирования")
    parser.add_argument("--profile_memory", action="store_true", help="Enable memory profiling")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA недоступна, профилирование не имеет смысла")
        return
    
    torch.cuda.synchronize()
    
    if args.mode in ["warmup", "both"]:
        pytorch_profile_warmup()
    
    if args.mode in ["no-warmup", "both"]:
        pytorch_profile_no_warmup(profile_memory=args.profile_memory)


if __name__ == "__main__":
    main()
