# mypy: allow-untyped-defs
from typing import Callable

import cuda.cccl.parallel.experimental as parallel

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.associative_scan import associative_scan, _fake_associative_scan

# We can't pass functions directly to a custom op so we workaround with this instead
function_registry = {}
temp_storage_registry = {}

# Check for optimized allocation at module load time
try:
    import torch._C._dynamo.guards as guards
    _HAS_OPTIMIZED_ALLOC = hasattr(guards, '_empty_strided_cuda')
except ImportError:
    _HAS_OPTIMIZED_ALLOC = False

def initialize_scan(
    combine_fn_name: str,
    size: int,
    d_input: torch.Tensor,
    dim: int,
    reverse: bool,
) -> torch.Tensor:
    """
    This function builds cuda.cccl.parallel scan and allocates temp storage,
    d_output, and h_init for it.
    """

    dtype = d_input.dtype
    d_output = torch.empty_like(d_input)
    if reverse:
        d_input = parallel.ReverseInputIterator(d_input)
        d_output_it = parallel.ReverseOutputIterator(d_output[:-1])
    else:
        d_output_it = d_output

    storage_cache_key = (size, dtype, combine_fn_name, reverse, torch.cuda.current_stream())
    if storage_cache_key in temp_storage_registry:
        scanner, d_temp_storage, h_init = temp_storage_registry[storage_cache_key]
        return scanner, d_temp_storage, h_init, d_output, d_output_it

    h_init = torch.empty(1, dtype=dtype).numpy()

    combine_fn = function_registry[combine_fn_name]

    scanner = parallel.make_inclusive_scan(d_input, d_output_it, combine_fn, h_init)
    temp_storage_size = scanner(None, d_input, d_output_it, size, h_init)

    # Use optimized allocation (checked at module load)
    if _HAS_OPTIMIZED_ALLOC:
        d_temp_storage = guards._empty_strided_cuda((temp_storage_size,), (1,), torch.uint8)
    else:
        # Fallback to standard allocation
        d_temp_storage = torch.empty(temp_storage_size, dtype=torch.uint8, device="cuda")

    temp_storage_registry[storage_cache_key] = (scanner, d_temp_storage, h_init)
    
    return scanner, d_temp_storage, h_init, d_output, d_output_it


@torch.library.custom_op("cccl::associative_scan", mutates_args=())
def associative_scan_impl(
    combine_fn_name: str,
    d_input: torch.Tensor,
    dim: int,
    reverse: bool,
) -> torch.Tensor:

    size = d_input.shape[0]
    scanner, d_temp_storage, h_init, d_output, d_output_it = initialize_scan(combine_fn_name, size, d_input, dim, reverse)

    if reverse:
        first_elem = d_input[-1]
        current_input_it = parallel.ReverseInputIterator(d_input[:-1])
        current_output_it = d_output_it # already a reverse iterator
    else:
        first_elem = d_input[0]
        current_input_it = d_input[1:]
        current_output_it = d_output[1:]

    h_init[0] = first_elem
    scanner(d_temp_storage, current_input_it, current_output_it, size - 1, h_init)

    if reverse:
        d_output.data[-1] = first_elem
    else:
        d_output.data[0] = first_elem

    return d_output

@associative_scan_impl.register_fake
def _(combine_fn_name, xs, dim, reverse):
    return torch.empty_like(xs)


def cuda_parallel_associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    xs: pytree.PyTree,
    dim: int,
    reverse: bool = False,
    combine_mode: str = "pointwise",
) -> torch.Tensor:
    r"""
    A CUDA-optimized parallel implementation of associative scan that preserves
    the same semantics as the standard associative_scan.

    This implementation currently falls back to the standard associative_scan
    to ensure semantic compatibility. Future versions will implement a
    specialized CUDA parallel algorithm while maintaining the same behavior.

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, i.e., no lifted arguments are supported at the moment,
            satisfy the associative property and have no side-effects.
        xs (torch.Tensor): The input tensor, or nested pytree of tensors.
            All inputs are expected to have the same shape.
        dim (int): the dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.
        combine_mode (str): A string indicating whether the ``combine_fn`` is ``pointwise`` or ``generic``, default ``pointwise``.
            If ``combine_mode=pointwise``, ``combine_fn`` must be pure, may only contain pointwise operations
            and ``xs`` must be CUDA tensors.
            In all other cases ``combine_mode=generic`` should be used.
            Note: ``combine_mode=pointwise`` is more efficient than ``combine_mode=generic``.

    Returns:
        torch.Tensor: The result of the associative scan operation.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        cumsum = cuda_parallel_associative_scan(add, x, dim)

    .. note::
        This implementation currently falls back to the standard associative_scan
        implementation to ensure semantic compatibility during development.
    """

    if combine_mode == "pointwise" and isinstance(xs, torch.Tensor) and xs.device.type == "cuda" and xs.is_contiguous() and xs.ndim == 1:
        # TODO: instead of using the name, is there a better unique identifier? What
        # if there is a naming clash?
        combine_fn_name = combine_fn.__name__
        function_registry[combine_fn_name] = combine_fn # type: ignore
        
        # Force a graph break to ensure the registry update happens in eager mode
        # This prevents the compilation system from capturing an empty registry
        torch._dynamo.graph_break()

        # print("CUDA_PARALLEL_SCAN_USED: Using CUDA parallel associative scan")
        return associative_scan_impl(combine_fn_name, xs, dim, reverse)

    # For now, fall back to the standard associative_scan implementation
    # This ensures we preserve the exact same semantics and behavior
    return associative_scan(
        combine_fn=combine_fn,
        xs=xs,
        dim=dim,
        reverse=reverse,
        combine_mode=combine_mode,
    )


# Re-export _fake_associative_scan for compatibility with tests that import it directly
__all__ = ["cuda_parallel_associative_scan", "_fake_associative_scan"]
