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

def create_speedup_plot(base_data, cccl_untuned_data, cccl_tuned_data, base_mode, title, filename):
    """Create a speedup comparison plot with untuned and tuned CCCL data."""
    # Collect data for plotting
    sizes = [4096, 65536, 1048576, 16777216, 268435456]  # 2^12, 2^16, 2^20, 2^24, 2^28
    dtypes = ['float32', 'float64']
    
    # Prepare data arrays
    x_positions = np.arange(len(sizes))
    width = 0.12  # Narrower bars to fit 6 bars per group
    
    # Arrays to store speedup data
    untuned_speedups_f32 = []
    untuned_speedups_f64 = []
    tuned_speedups_f32 = []
    tuned_speedups_f64 = []
    
    for size in sizes:
        # Get base time for this size and mode
        base_time_f32 = base_data.get((size, 'float32', base_mode))
        base_time_f64 = base_data.get((size, 'float64', base_mode))
        
        # Get cccl eager times for this size
        untuned_time_f32 = cccl_untuned_data.get((size, 'float32', 'eager'))
        untuned_time_f64 = cccl_untuned_data.get((size, 'float64', 'eager'))
        tuned_time_f32 = cccl_tuned_data.get((size, 'float32', 'eager'))
        tuned_time_f64 = cccl_tuned_data.get((size, 'float64', 'eager'))
        
        # Calculate speedups (base_time / cccl_time) - PyTorch time / CCCL time
        untuned_speedup_f32 = base_time_f32 / untuned_time_f32 if base_time_f32 and untuned_time_f32 else None
        untuned_speedup_f64 = base_time_f64 / untuned_time_f64 if base_time_f64 and untuned_time_f64 else None
        tuned_speedup_f32 = base_time_f32 / tuned_time_f32 if base_time_f32 and tuned_time_f32 else None
        tuned_speedup_f64 = base_time_f64 / tuned_time_f64 if base_time_f64 and tuned_time_f64 else None
            
        untuned_speedups_f32.append(untuned_speedup_f32)
        untuned_speedups_f64.append(untuned_speedup_f64)
        tuned_speedups_f32.append(tuned_speedup_f32)
        tuned_speedups_f64.append(tuned_speedup_f64)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Define colors and positions for the 4 bars per group
    colors = {
        'untuned_f32': '#87CEEB',  # skyblue
        'untuned_f64': '#F08080',  # lightcoral
        'tuned_f32': '#4169E1',    # royalblue (darker blue)
        'tuned_f64': '#DC143C'     # crimson (darker red)
    }
    
    # Plot bars for each category
    for i, size in enumerate(sizes):
        x_base = i
        
        # Untuned float32
        if untuned_speedups_f32[i] is not None:
            ax.bar(x_base - 1.5*width, untuned_speedups_f32[i], width, 
                   color=colors['untuned_f32'], alpha=0.8, 
                   label='CCCL Untuned float32' if i == 0 else "")
        
        # Untuned float64
        if untuned_speedups_f64[i] is not None:
            ax.bar(x_base - 0.5*width, untuned_speedups_f64[i], width,
                   color=colors['untuned_f64'], alpha=0.8,
                   label='CCCL Untuned float64' if i == 0 else "")
        
        # Tuned float32
        if tuned_speedups_f32[i] is not None:
            ax.bar(x_base + 0.5*width, tuned_speedups_f32[i], width,
                   color=colors['tuned_f32'], alpha=0.8,
                   label='CCCL Tuned float32' if i == 0 else "")
        
        # Tuned float64
        if tuned_speedups_f64[i] is not None:
            ax.bar(x_base + 1.5*width, tuned_speedups_f64[i], width,
                   color=colors['tuned_f64'], alpha=0.8,
                   label='CCCL Tuned float64' if i == 0 else "")
    
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
            if untuned_speedups_f32[i] is not None:
                ax.text(x_base - 1.5*width, untuned_speedups_f32[i],
                       f'{untuned_speedups_f32[i]:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
            
            if untuned_speedups_f64[i] is not None:
                ax.text(x_base - 0.5*width, untuned_speedups_f64[i],
                       f'{untuned_speedups_f64[i]:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
            
            if tuned_speedups_f32[i] is not None:
                ax.text(x_base + 0.5*width, tuned_speedups_f32[i],
                       f'{tuned_speedups_f32[i]:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
            
            if tuned_speedups_f64[i] is not None:
                ax.text(x_base + 1.5*width, tuned_speedups_f64[i],
                       f'{tuned_speedups_f64[i]:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
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
    
    # Create first plot: CCCL (untuned & tuned) eager vs Base eager
    print("\nCreating plot 1: CCCL speedup vs PyTorch eager...")
    fig1 = create_speedup_plot(
        base_data, cccl_untuned_data, cccl_tuned_data, 'eager',
        'CCCL Speedup vs PyTorch Eager Performance\n(Add Operator - Higher is Better)',
        'cccl_vs_pytorch_eager.png'
    )
    
    # Create second plot: CCCL (untuned & tuned) eager vs Base compiled
    print("Creating plot 2: CCCL speedup vs PyTorch compiled...")
    fig2 = create_speedup_plot(
        base_data, cccl_untuned_data, cccl_tuned_data, 'compiled',
        'CCCL Speedup vs PyTorch Compiled Performance\n(Add Operator - Higher is Better)',
        'cccl_vs_pytorch_compiled.png'
    )
    
    print("\nPlots saved as:")
    print("  - 'cccl_vs_pytorch_eager.png'")
    print("  - 'cccl_vs_pytorch_compiled.png'")
    
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
