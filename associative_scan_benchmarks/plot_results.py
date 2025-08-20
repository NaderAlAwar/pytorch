#!/usr/bin/env python3
"""
Script to plot benchmark results comparing CCCL vs Base performance.
Creates two plots:
1. CCCL eager vs Base eager speedup
2. CCCL eager vs Base compiled speedup
"""

import json
import matplotlib.pyplot as plt
import numpy as np
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
    """Create a speedup comparison plot."""
    # Collect data for plotting
    sizes = [4096, 65536, 1048576, 16777216, 268435456]  # 2^12, 2^16, 2^20, 2^24, 2^28
    dtypes = ['float32', 'float64']
    
    # Prepare data arrays
    x_positions = np.arange(len(sizes))
    width = 0.35
    
    speedups_f32 = []
    speedups_f64 = []
    
    for size in sizes:
        # Get base time for this size and mode
        base_time_f32 = base_data.get((size, 'float32', base_mode))
        base_time_f64 = base_data.get((size, 'float64', base_mode))
        
        # Get cccl eager time for this size
        cccl_time_f32 = cccl_data.get((size, 'float32', 'eager'))
        cccl_time_f64 = cccl_data.get((size, 'float64', 'eager'))
        
        # Calculate speedup (cccl_time / base_time)
        if base_time_f32 and cccl_time_f32:
            speedup_f32 = cccl_time_f32 / base_time_f32
        else:
            speedup_f32 = None
            
        if base_time_f64 and cccl_time_f64:
            speedup_f64 = cccl_time_f64 / base_time_f64
        else:
            speedup_f64 = None
            
        speedups_f32.append(speedup_f32)
        speedups_f64.append(speedup_f64)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter out None values for plotting
    valid_f32 = [(i, s) for i, s in enumerate(speedups_f32) if s is not None]
    valid_f64 = [(i, s) for i, s in enumerate(speedups_f64) if s is not None]
    
    if valid_f32:
        indices_f32, values_f32 = zip(*valid_f32)
        bars1 = ax.bar([x - width/2 for x in indices_f32], values_f32, width, 
                      label='float32', alpha=0.8, color='skyblue')
    
    if valid_f64:
        indices_f64, values_f64 = zip(*valid_f64)
        bars2 = ax.bar([x + width/2 for x in indices_f64], values_f64, width,
                      label='float64', alpha=0.8, color='lightcoral')
    
    # Add horizontal line at y=1 (no speedup/slowdown)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Input Size (Number of Elements)', fontsize=12)
    ax.set_ylabel('Speedup (CCCL time / Base time)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'2^{int(np.log2(size))}' for size in sizes], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    if valid_f32:
        add_value_labels(bars1, values_f32)
    if valid_f64:
        add_value_labels(bars2, values_f64)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    # Load data from both JSON files
    print("Loading base.json...")
    base_data = load_benchmark_data('base.json')
    
    print("Loading cccl.json...")
    cccl_data = load_benchmark_data('cccl.json')
    
    print(f"Base data points: {len(base_data)}")
    print(f"CCCL data points: {len(cccl_data)}")
    
    # Create first plot: CCCL eager vs Base eager
    print("\nCreating plot 1: CCCL eager vs Base eager...")
    fig1 = create_speedup_plot(
        base_data, cccl_data, 'eager',
        'CCCL vs Base Eager Performance Comparison\n(Add Operator - Lower is Better)',
        'cccl_vs_base_eager.png'
    )
    
    # Create second plot: CCCL eager vs Base compiled
    print("Creating plot 2: CCCL eager vs Base compiled...")
    fig2 = create_speedup_plot(
        base_data, cccl_data, 'compiled',
        'CCCL vs Base Compiled Performance Comparison\n(Add Operator - Lower is Better)',
        'cccl_vs_base_compiled.png'
    )
    
    print("\nPlots saved as:")
    print("  - 'cccl_vs_base_eager.png'")
    print("  - 'cccl_vs_base_compiled.png'")
    
    # Print some sample data for verification
    print("\nSample data verification:")
    print("Base eager times (float32):")
    for size in [4096, 65536, 1048576]:
        time_val = base_data.get((size, 'float32', 'eager'))
        if time_val:
            print(f"  Size {size}: {time_val:.6f}s")
    
    print("CCCL eager times (float32):")
    for size in [4096, 65536, 1048576]:
        time_val = cccl_data.get((size, 'float32', 'eager'))
        if time_val:
            print(f"  Size {size}: {time_val:.6f}s")

if __name__ == "__main__":
    main()
