# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# @author Yi-Chen Lu
# @file visualization/plotting.py
# @brief plotting utilities

import os
import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Union, Any

# Configure plot style
sns.set(style='whitegrid')


def plot_correlation(
    true_slacks: torch.Tensor,
    pred_slacks: torch.Tensor,
    ep_levels: torch.Tensor,
    true_levels: torch.Tensor,
    prefix: str,
    num_pins: int,
    design_name: str,
    file_postfix: str = '',
    cmap: str = 'RdYlBu_r',
    tight_layout: bool = True,
    plot_format: str = 'png',
    **scatter_kwargs
) -> None:
    """
    Plot correlation between predicted and actual timing values

    Args:
        true_slacks: True slack values tensor
        pred_slacks: Predicted slack values tensor
        ep_levels: Endpoint levels tensor
        true_levels: True levels tensor
        prefix: Directory prefix for output
        num_pins: Number of pins in design
        design_name: Name of the design
        file_postfix: Optional postfix for output filename
        cmap: Colormap for plotting
        tight_layout: Whether to use tight layout
        plot_format: Output file format
        **scatter_kwargs: Additional arguments for scatter plot
    """
    # Create output directory
    os.makedirs(os.path.join(prefix, 'plots'), exist_ok=True)

    pred_slacks = pred_slacks.cpu()
    true_slacks = true_slacks.cpu()

    # Calculate correlation coefficient
    slack_corr = np.corrcoef(
        pred_slacks.to(torch.float32).numpy(),
        true_slacks.to(torch.float32).numpy()
    )[0][1]

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 12))

    # Create scatter plot with colorbar
    scatter = ax.scatter(
        pred_slacks.to(torch.float32),
        true_slacks.to(torch.float32),
        c=ep_levels,
        cmap=cmap,
        **scatter_kwargs
    )

    # Add labels and title
    ax.set_xlabel(f"ep slack from INSTA (ns)", fontsize=20)
    ax.set_ylabel(f"ep slack from ref tool (ns)", fontsize=20)
    ax.set_title(f"{design_name}, # pins: {num_pins}, corr: {slack_corr:.8f}", fontsize=22)
    ax.tick_params(axis='both', labelsize=18)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('max logic depth', size=20)
    cbar.ax.tick_params(labelsize=18)

    # Add diagonal line
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    min_val = min(xlim[0], ylim[0])
    max_val = max(xlim[1], ylim[1])
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Layout and save
    if tight_layout:
        fig.tight_layout()

    # Construct filename
    filename = f"slack_corr{('_' + file_postfix) if file_postfix else ''}.{plot_format}"
    fig.savefig(os.path.join(prefix, 'plots', filename))
    plt.close(fig)

    print(f"Correlation plot saved to {os.path.join(prefix, 'plots', filename)}")


def plot_histogram(
    true_values: torch.Tensor,
    pred_values: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    prefix: str,
    name: str = 'histogram',
    bin_width: float = 0.03,
    plot_format: str = 'png',
    log_scale: bool = False
) -> None:
    """
    Plot histogram comparison between true and predicted values

    Args:
        true_values: True values tensor
        pred_values: Predicted values tensor
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        prefix: Directory prefix for output
        name: Base filename
        bin_width: Width of histogram bins
        plot_format: Output file format
        log_scale: Whether to use log scale
    """
    # Create output directory
    os.makedirs(os.path.join(prefix, 'plots'), exist_ok=True)

    # Filter out inf values
    inf_mask = ~(torch.isinf(true_values) | torch.isinf(pred_values))
    true_values = true_values[inf_mask]
    pred_values = pred_values[inf_mask]

    # Convert to numpy
    true_np = true_values.cpu().numpy()
    pred_np = pred_values.cpu().numpy()

    # Determine bin range
    min_val = min(np.min(true_np), np.min(pred_np))
    max_val = max(np.max(true_np), np.max(pred_np))
    bin_count = max(10, int((max_val - min_val) / bin_width))
    bins = np.linspace(min_val, max_val, bin_count)

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot histograms
    sns.histplot(true_np, bins=bins, ax=ax, color='blue', alpha=0.4, label='ref tool')
    sns.histplot(pred_np, bins=bins, ax=ax, color='red', alpha=0.4, label='INSTA')

    # Add labels and title
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    # Set log scale if requested
    if log_scale:
        ax.set_yscale('log')

    # Add legend
    ax.legend(fontsize=14)

    # Layout and save
    fig.tight_layout()
    filename = f"{name}.{plot_format}"
    fig.savefig(os.path.join(prefix, 'plots', filename))
    plt.close(fig)

    print(f"Histogram saved to {os.path.join(prefix, 'plots', filename)}")


def write_analysis_csv(
    epNames: List[str],
    true_slacks: torch.Tensor,
    pred_slacks: torch.Tensor,
    true_arrivals: torch.Tensor,
    pred_arrivals: torch.Tensor,
    true_requireds: torch.Tensor,
    spNames: List[str],
    riseFalls: List[str],
    prefix: str,
    filename: str = 'analysis.csv'
) -> None:
    """
    Write detailed timing analysis to CSV file

    Args:
        epNames: List of endpoint pin names
        true_slacks: True slack values tensor
        pred_slacks: Predicted slack values tensor
        true_arrivals: True arrival times tensor
        pred_arrivals: Predicted arrival times tensor
        true_requireds: Required times tensor
        spNames: List of startpoint pin names
        riseFalls: List of rise/fall edge types
        prefix: Directory prefix for output
        filename: Output filename
    """
    # Create output directory
    os.makedirs(os.path.join(prefix, 'plots'), exist_ok=True)

    # Calculate differences
    diff_slacks = true_slacks - pred_slacks
    diff_arrs = true_arrivals - pred_arrivals

    # Sort by absolute slack difference
    indices = torch.argsort(torch.abs(diff_slacks), descending=True)

    # Write CSV file
    with open(os.path.join(prefix, 'plots', filename), 'w') as f:
        # Write header
        f.write('epName,true_slack,pred_slack,diff_slack,true_arr,pred_arr,diff_arr,required,spName,riseFall\n')

        # Write data rows
        for idx in indices.tolist():
            f.write(f'{epNames[idx]},{true_slacks[idx]},{pred_slacks[idx]},{diff_slacks[idx]},'
                    f'{true_arrivals[idx]},{pred_arrivals[idx]},{diff_arrs[idx]},'
                    f'{true_requireds[idx]},{spNames[idx]},{riseFalls[idx]}\n')

    print(f"Analysis CSV saved to {os.path.join(prefix, 'plots', filename)}")


def plot_ep_correlation(
    file_postfix: str,
    true_slacks: torch.Tensor,
    pred_slacks: torch.Tensor,
    ep_levels: torch.Tensor,
    true_levels: torch.Tensor,
    prefix: str,
    num_pins: int,
    design_name: str,
    cmap: str = 'RdYlBu_r',
    tight_layout: bool = True,
    plot_format: str = 'png',
    **scatter_kwargs
) -> None:
    """
    Plot correlation of endpoint timing data and write analysis CSV

    Args:
        file_postfix: Postfix for output filename
        true_slacks: True slack values tensor
        pred_slacks: Predicted slack values tensor
        ep_levels: Endpoint levels tensor
        true_levels: True levels tensor
        prefix: Directory prefix for output
        num_pins: Number of pins in design
        design_name: Name of the design
        cmap: Colormap for plotting
        tight_layout: Whether to use tight layout
        plot_format: Output file format
        **scatter_kwargs: Additional arguments for scatter plot
    """
    # Create output directory
    os.makedirs(os.path.join(prefix, 'plots'), exist_ok=True)

    # Filter out inf values
    import ipdb
    try:
        inf_mask = ~(torch.isinf(true_slacks) | torch.isinf(pred_slacks))
        filtered_true_slacks = true_slacks[inf_mask]
        filtered_pred_slacks = pred_slacks[inf_mask]
        inf_mask = inf_mask.cpu()
        filtered_ep_levels = ep_levels[inf_mask].tolist()
        filtered_true_levels = true_levels[inf_mask]
    except:
        ipdb.set_trace()

    # Create slack correlation plot
    plot_correlation(
        filtered_true_slacks,
        filtered_pred_slacks,
        filtered_ep_levels,
        filtered_true_levels,
        prefix,
        num_pins,
        design_name,
        file_postfix,
        cmap,
        tight_layout,
        plot_format,
        **scatter_kwargs
    )


