#!/usr/bin/env python3
"""
Script to plot benchmark results comparing CCCL vs Base performance.
Creates two plots:
1. CCCL (untuned & tuned) eager vs Base eager speedup
2. CCCL (untuned & tuned) eager vs Base compiled speedup
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict

def load_benchmark_data(json_file):
    """Load and parse benchmark data from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = defaultdict(lambda: defaultdict(dict))
    
    # Parse benchmark results
    for benchmark in data['benchmarks']:
        for state in benchmark['states']:
            # Extract axis values
            axis_values = {}
            for axis in state['axis_values']:
                if axis['name'] == 'numElems':
                    axis_values['size'] = int(axis['value'])
                elif axis['name'] == 'dtype':
                    axis_values['dtype'] = axis['value']
                elif axis['name'] == 'operator':
                    axis_values['operator'] = axis['value']
                elif axis['name'] == 'compile':
                    axis_values['compile'] = axis['value']
            
            # Only process 'add' operator
            if axis_values.get('operator') != 'add':
                continue
            
            # Extract GPU timing data
            gpu_time = None
            for summary in state['summaries']:
                if summary['tag'] == 'nv/cold/time/gpu/mean':
                    gpu_time = float(summary['data'][0]['value'])
                    break
            
            if gpu_time is not None:
                key = (axis_values['size'], axis_values['dtype'], axis_values['compile'])
                results[key] = gpu_time
    
    return results

def create_speedup_plot(base_data, cccl_data, base_mode, title, filename):
    """Create a speedup comparison plot with CCCL data."""
    # Collect data for plotting
    sizes = [4096, 65536, 1048576, 16777216, 268435456]  # 2^12, 2^16, 2^20, 2^24, 2^28
    dtypes = ['float16', 'float32', 'float64']
    
    # Prepare data arrays
    x_positions = np.arange(len(sizes))
    width = 0.25  # Wider bars for 3 bars per group
    
    # Arrays to store speedup data
    speedups_f16 = []
    speedups_f32 = []
    speedups_f64 = []
    
    for size in sizes:
        # Get base time for this size and mode
        base_time_f16 = base_data.get((size, 'float16', base_mode))
        base_time_f32 = base_data.get((size, 'float32', base_mode))
        base_time_f64 = base_data.get((size, 'float64', base_mode))
        
        # Get cccl eager times for this size
        cccl_time_f16 = cccl_data.get((size, 'float16', 'eager'))
        cccl_time_f32 = cccl_data.get((size, 'float32', 'eager'))
        cccl_time_f64 = cccl_data.get((size, 'float64', 'eager'))
        
        # Calculate speedups (base_time / cccl_time) - PyTorch time / CCCL time
        speedup_f16 = base_time_f16 / cccl_time_f16 if base_time_f16 and cccl_time_f16 else None
        speedup_f32 = base_time_f32 / cccl_time_f32 if base_time_f32 and cccl_time_f32 else None
        speedup_f64 = base_time_f64 / cccl_time_f64 if base_time_f64 and cccl_time_f64 else None
            
        speedups_f16.append(speedup_f16)
        speedups_f32.append(speedup_f32)
        speedups_f64.append(speedup_f64)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Define colors for the 3 bars per group
    colors = {
        'f16': '#32CD32',    # limegreen
        'f32': '#4169E1',    # royalblue
        'f64': '#DC143C'     # crimson
    }
    
    # Plot bars for each category
    for i, size in enumerate(sizes):
        x_base = i
        
        # float16
        if speedups_f16[i] is not None:
            ax.bar(x_base - width, speedups_f16[i], width, 
                   color=colors['f16'], alpha=0.8, 
                   label='float16' if i == 0 else "")
        
        # float32
        if speedups_f32[i] is not None:
            ax.bar(x_base, speedups_f32[i], width, 
                   color=colors['f32'], alpha=0.8, 
                   label='float32' if i == 0 else "")
        
        # float64
        if speedups_f64[i] is not None:
            ax.bar(x_base + width, speedups_f64[i], width,
                   color=colors['f64'], alpha=0.8,
                   label='float64' if i == 0 else "")
    
    # Add horizontal line at y=1 (no speedup/slowdown)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Input Size (Number of Elements)', fontsize=12)
    ax.set_ylabel('CCCL Speedup (PyTorch time / CCCL time)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'2^{int(np.log2(size))}' for size in sizes], rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels_on_bars():
        for i, size in enumerate(sizes):
            x_base = i
            
            # Add labels for each bar
            if speedups_f16[i] is not None:
                ax.text(x_base - width, speedups_f16[i],
                       f'{speedups_f16[i]:.2f}', ha='center', va='bottom', fontsize=9, rotation=90)
            
            if speedups_f32[i] is not None:
                ax.text(x_base, speedups_f32[i],
                       f'{speedups_f32[i]:.2f}', ha='center', va='bottom', fontsize=9, rotation=90)
            
            if speedups_f64[i] is not None:
                ax.text(x_base + width, speedups_f64[i],
                       f'{speedups_f64[i]:.2f}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    add_value_labels_on_bars()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory
    
    return fig


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot benchmark results comparing CCCL vs Base performance')
    parser.add_argument('base_json', help='Path to base benchmark JSON file (e.g., base-h200.json)')
    parser.add_argument('untuned_json', help='Path to untuned CCCL benchmark JSON file (e.g., cccl-h200-untuned.json)')
    parser.add_argument('tuned_json', help='Path to tuned CCCL benchmark JSON file (e.g., cccl-h200-tuned.json)')
    
    args = parser.parse_args()
    
    # Load data from the three JSON files
    print(f"Loading {args.base_json}...")
    base_data = load_benchmark_data(args.base_json)
    
    print(f"Loading {args.untuned_json}...")
    cccl_untuned_data = load_benchmark_data(args.untuned_json)
    
    print(f"Loading {args.tuned_json}...")
    cccl_tuned_data = load_benchmark_data(args.tuned_json)
    
    print(f"Base data points: {len(base_data)}")
    print(f"CCCL untuned data points: {len(cccl_untuned_data)}")
    print(f"CCCL tuned data points: {len(cccl_tuned_data)}")
    
    # Create untuned plots
    print("\nCreating plot 1: CCCL speedup vs PyTorch eager...")
    fig1 = create_speedup_plot(
        base_data, cccl_untuned_data, 'eager',
        'CCCL Speedup vs PyTorch Eager Performance\n(Add Operator - Higher is Better)',
        'cccl_vs_pytorch_eager.png'
    )
    
    print("Creating plot 2: CCCL speedup vs PyTorch compiled...")
    fig2 = create_speedup_plot(
        base_data, cccl_untuned_data, 'compiled',
        'CCCL Speedup vs PyTorch Compiled Performance\n(Add Operator - Higher is Better)',
        'cccl_vs_pytorch_compiled.png'
    )
    
    # Create tuned plots
    print("Creating plot 3: CCCL speedup vs PyTorch eager (after tunings)...")
    fig3 = create_speedup_plot(
        base_data, cccl_tuned_data, 'eager',
        'CCCL Speedup vs PyTorch Eager Performance (after tunings)\n(Add Operator - Higher is Better)',
        'cccl_tuned_vs_pytorch_eager.png'
    )
    
    print("Creating plot 4: CCCL speedup vs PyTorch compiled (after tunings)...")
    fig4 = create_speedup_plot(
        base_data, cccl_tuned_data, 'compiled',
        'CCCL Speedup vs PyTorch Compiled Performance (after tunings)\n(Add Operator - Higher is Better)',
        'cccl_tuned_vs_pytorch_compiled.png'
    )
    
    print("\nPlots saved as:")
    print("  - 'cccl_vs_pytorch_eager.png'")
    print("  - 'cccl_vs_pytorch_compiled.png'")
    print("  - 'cccl_tuned_vs_pytorch_eager.png'")
    print("  - 'cccl_tuned_vs_pytorch_compiled.png'")
    
    # Print some sample data for verification
    print("\nSample data verification:")
    print("PyTorch eager times (float32):")
    for size in [4096, 65536, 1048576]:
        time_val = base_data.get((size, 'float32', 'eager'))
        if time_val:
            print(f"  Size {size}: {time_val:.6f}s")
    
    print("CCCL untuned eager times (float32):")
    for size in [4096, 65536, 1048576]:
        time_val = cccl_untuned_data.get((size, 'float32', 'eager'))
        if time_val:
            print(f"  Size {size}: {time_val:.6f}s")
    
    print("CCCL tuned eager times (float32):")
    for size in [4096, 65536, 1048576]:
        time_val = cccl_tuned_data.get((size, 'float32', 'eager'))
        if time_val:
            print(f"  Size {size}: {time_val:.6f}s")
    
    print("Sample speedups (PyTorch / CCCL) for float32:")
    for size in [4096, 65536, 1048576]:
        pytorch_time = base_data.get((size, 'float32', 'eager'))
        untuned_time = cccl_untuned_data.get((size, 'float32', 'eager'))
        tuned_time = cccl_tuned_data.get((size, 'float32', 'eager'))
        if pytorch_time and untuned_time and tuned_time:
            untuned_speedup = pytorch_time / untuned_time
            tuned_speedup = pytorch_time / tuned_time
            print(f"  Size {size}: Untuned={untuned_speedup:.2f}x, Tuned={tuned_speedup:.2f}x")

if __name__ == "__main__":
    main()
