import sys

# This is not used directly here, but removing this results in a build error.
# Something about the import order of cuda.cccl.parallel.experimental and torch
# affects the build.
import cuda.cccl.parallel.experimental as parallel

import cuda.bench as bench
import torch
import numpy as np


def cumulative_ops_benchmark(state: bench.State):
    """Benchmark cumsum and cumprod operations with different data types"""
    n_elems = state.get_int64("numElems")
    dtype_str = state.get_string("dtype")
    operator = state.get_string("operator")
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64
    }
    dtype = dtype_map[dtype_str]
    
    state.add_summary("dtype", dtype_str)
    state.add_summary("operator", operator)

    # Create input tensor on CUDA
    device = torch.device(f"cuda:{state.get_device()}")
    
    # Initialize with small positive values for multiplication to avoid overflow/underflow
    if operator == "cumprod":
        # Use values close to 1.0 to avoid numerical issues
        input_tensor = torch.ones(n_elems, dtype=dtype, device=device) + torch.randn(n_elems, dtype=dtype, device=device) * 0.1
    else:
        # For cumsum, use random values
        input_tensor = torch.randn(n_elems, dtype=dtype, device=device)
    
    torch_op = torch.cumsum if operator == "cumsum" else torch.cumprod
    # Make sure tensor is contiguous for optimal performance
    input_tensor = input_tensor.contiguous()
    
    def launcher(launch: bench.Launch):
        # Perform the cumulative operation
        result = torch_op(input_tensor, dim=0)
        # The result computation is already synchronous
    
    # Use sync=True since cumulative ops synchronize internally
    state.exec(launcher, sync=True)


if __name__ == "__main__":
    # Register the benchmark
    b = bench.register(cumulative_ops_benchmark)
    
    # Add axes for different array sizes (powers of 2 from 2^12 to 2^28, step 4)
    b.add_int64_power_of_two_axis("numElems", range(12, 29, 4))  # 2^12, 2^16, 2^20, 2^24, 2^28
    
    # Add axes for different data types
    b.add_string_axis("dtype", ["float32", "float64"])
    
    # Add axes for different operators
    b.add_string_axis("operator", ["cumsum", "cumprod"])
    
    # Run all benchmarks
    bench.run_all_benchmarks(sys.argv)
