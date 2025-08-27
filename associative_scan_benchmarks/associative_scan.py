import sys
import os

# This is not used directly here, but removing this results in a build error.
# Something about the import order of cuda.cccl.parallel.experimental and torch
# affects the build.
import cuda.cccl.parallel.experimental as parallel


import cuda.bench as bench
import torch
import numpy as np

# Import associative_scan (will use macro internally to handle implementation)
from torch._higher_order_ops import associative_scan

# Increase recompile limit to avoid warnings when benchmarking different tensor sizes
torch._dynamo.config.recompile_limit = 100


def as_torch_cuda_Stream(
    cs: bench.CudaStream, dev: int | None
) -> torch.cuda.ExternalStream:
    return torch.cuda.ExternalStream(
        stream_ptr=cs.addressof(), device=torch.cuda.device(dev)
    )


def associative_scan_benchmark(state: bench.State):
    """Benchmark associative_scan with different operators and data types"""
    n_elems = state.get_int64("numElems")
    dtype_str = state.get_string("dtype")
    operator = state.get_string("operator")
    compile_mode = state.get_string("compile")
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64
    }
    dtype = dtype_map[dtype_str]
    
    # Set up operator functions
    def add_op(x, y):
        return x + y
    
    def mul_op(x, y):
        return x * y

    if "PYTORCH_ASSOCIATIVE_SCAN_CUDA_PARALLEL" in os.environ:
        op_map = {
            "add": parallel.OpKind.PLUS,
            "mul": parallel.OpKind.MULTIPLIES
        }
    else:
        op_map = {
            "add": add_op,
            "mul": mul_op
        }

    combine_fn = op_map[operator]

    state.add_summary("dtype", dtype_str)
    state.add_summary("operator", operator)
    state.add_summary("compile", compile_mode)

    dev_id = state.get_device()
    tc_s = as_torch_cuda_Stream(state.get_stream(), dev_id)
    device = torch.device(f"cuda:{dev_id}")

    with torch.cuda.device(device), torch.cuda.stream(tc_s):
            # Initialize with small positive values for multiplication to avoid overflow/underflow
            if operator == "mul":
                # Use values close to 1.0 to avoid numerical issues
                input_tensor = torch.ones(n_elems, dtype=dtype, device=device) + torch.randn(n_elems, dtype=dtype, device=device) * 0.1
            else:
                # For addition, use random values
                input_tensor = torch.randn(n_elems, dtype=dtype, device=device)
    
    # Make sure tensor is contiguous for optimal performance
    input_tensor = input_tensor.contiguous()
    
    # Create the scan function
    def scan_fn(tensor):
        return associative_scan(
            combine_fn=combine_fn,
            xs=tensor,
            dim=0,
            reverse=False,
            combine_mode="pointwise"
        )
    
    # Optionally compile the function
    if compile_mode in ["compiled", "dynamic"]:
        dynamic_shapes = compile_mode == "dynamic"
        scan_fn = torch.compile(scan_fn, dynamic=dynamic_shapes)

    _ = scan_fn(input_tensor)
    torch.cuda.synchronize()
    
    def launcher(launch: bench.Launch):
        tc_s = as_torch_cuda_Stream(launch.get_stream(), dev_id)
        with torch.cuda.device(device), torch.cuda.stream(tc_s):
            scan_fn(input_tensor)

    state.exec(launcher, sync=True)


if __name__ == "__main__":
    b = bench.register(associative_scan_benchmark)
    b.add_int64_power_of_two_axis("numElems", range(12, 29, 4))  # 2^12, 2^16, 2^20, 2^24, 2^28
    b.add_string_axis("dtype", ["float16", "float32", "float64"])
    b.add_string_axis("operator", ["add", "mul"])
    b.add_string_axis("compile", ["eager", "compiled", "dynamic"])
    bench.run_all_benchmarks(sys.argv)
