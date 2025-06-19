# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# @author Yi-Chen Lu
# @file timing/propagation.py
# @brief timing prop related functions

import os
import time
import torch
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from copy import deepcopy

from .pocv import calculate_slack
from .cuda_ops import cuda_arrival_propagate_pocv


def clear_timing_cache(
    max_gid: int,
    topk: int,
    device: torch.device,
    dtype: torch.dtype,
    sp_mean_tensor: torch.Tensor,
    sp_std_tensor: torch.Tensor,
    ep_rise_required_truth: torch.Tensor,
    ep_fall_required_truth: torch.Tensor,
    ep_to_sp_map: Dict,
    pin_to_id_map: Dict[str, int],
    source_nodes: Set[int],
    dest_nodes: Set[int],
    existing_tensors: Optional[Dict[str, torch.Tensor]] = None,
    is_diff_prop: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Initialize or clear timing propagation cache

    Args:
        max_gid: Maximum node ID in the graph
        topk: Number of paths to track per endpoint
        device: Computation device
        dtype: Floating point precision
        sp_mean_tensor: Startpoint mean arrival times
        ep_rise_required_truth: Rise required times for endpoints
        ep_fall_required_truth: Fall required times for endpoints
        ep_to_sp_map: Mapping from endpoints to their startpoints
        pin_to_id_map: Mapping from pin names to their IDs
        source_nodes: Set of source nodes
        dest_nodes: Set of destination nodes
        existing_tensors: Optional dict of existing tensors to reuse

    Returns:
        Dictionary of initialized tensors
    """
    print('Clearing timing cache...')
    start_time = time.time()

    # Use existing tensors if provided and topK matches
    if existing_tensors and len(existing_tensors) > 0:
        first_tensor = next(iter(existing_tensors.values()))
        if len(first_tensor.shape) > 1 and first_tensor.shape[1] == topk:
            # Clear existing tensors
            for name, tensor in existing_tensors.items():
                if name.startswith('Gid_2_rise') or name.startswith('Gid_2_fall') or name.startswith('Gid_2_max'):
                    if 'slack' in name:
                        tensor.fill_(float('-inf'))
                    elif 'startpoints' in name:
                        tensor.fill_(-1)
                    else:
                        tensor.fill_(float('-inf'))
            # Ensure sp_std_tensor exists in the cached tensors (new requirement)
            assert 'sp_std_tensor' in existing_tensors, (
                "Cached timing tensors missing 'sp_std_tensor'; please reinitialize")
            print(f'Cache clearing takes {time.time() - start_time:.2f} seconds')
            return existing_tensors

    print('Initializing timing tensors for the first time')
    tensors = {}

    # Initialize 2D tensors for topK values
    tensors['Gid_2_rise_arrival'] = torch.full((max_gid, topk), float('-inf'), dtype=dtype).to(device)
    tensors['Gid_2_rise_arrival_mean'] = torch.full((max_gid, topk), float('-inf'), dtype=dtype).to(device)
    tensors['Gid_2_rise_arrival_std'] = torch.full((max_gid, topk), float('-inf'), dtype=dtype).to(device)
    tensors['Gid_2_rise_startpoints'] = torch.full((max_gid, topk), -1, dtype=torch.int32).to(device)
    tensors['Gid_2_fall_arrival'] = torch.full((max_gid, topk), float('-inf'), dtype=dtype).to(device)
    tensors['Gid_2_fall_arrival_mean'] = torch.full((max_gid, topk), float('-inf'), dtype=dtype).to(device)
    tensors['Gid_2_fall_arrival_std'] = torch.full((max_gid, topk), float('-inf'), dtype=dtype).to(device)
    tensors['Gid_2_fall_startpoints'] = torch.full((max_gid, topk), -1, dtype=torch.int32).to(device)
    tensors['Gid_2_max_arrival'] = torch.full((max_gid, topk), float('-inf'), dtype=dtype).to(device)
    tensors['Gid_2_startpoints'] = torch.full((max_gid, topk), -1, dtype=torch.int32).to(device)

    # Reshape if topK == 1
    if topk == 1 or is_diff_prop:
        for name, tensor in list(tensors.items()):
            tensors[name] = tensor.reshape(-1)

    # Move required times to device
    tensors['ep_rise_required_truth'] = ep_rise_required_truth.to(device)
    tensors['ep_fall_required_truth'] = ep_fall_required_truth.to(device)

    # Create valid startpoint tensor
    valid_sps = torch.zeros_like(ep_rise_required_truth).cpu()
    sps = [pin_to_id_map[spName] for spName in ep_to_sp_map.values()
           if spName in pin_to_id_map]
    valid_sps[sps] = 1
    tensors['valid_sps'] = valid_sps.to(torch.bool).to(device)

    # Create node tensors
    tensors['dest_node_tensor'] = torch.tensor(list(dest_nodes), dtype=torch.int32).to(device)
    tensors['sp_node_tensor'] = torch.tensor(list(source_nodes), dtype=torch.long).to(device)
    tensors['sp_mean_tensor'] = sp_mean_tensor.to(device)
    tensors['sp_std_tensor'] = sp_std_tensor.to(device)

    # Initialize slack tensors
    tensors['Gid_2_slack'] = torch.tensor([float('-inf')] * max_gid, dtype=dtype).to(device)
    tensors['Gid_2_rise_slack'] = torch.tensor([float('-inf')] * max_gid, dtype=dtype).to(device)
    tensors['Gid_2_fall_slack'] = torch.tensor([float('-inf')] * max_gid, dtype=dtype).to(device)

    print(f'Cache initialization takes {time.time() - start_time:.2f} seconds')
    return tensors


def propagate_arrival_times(
    timing_tensors: Dict[str, torch.Tensor],
    level_2_collaterals: Dict[int, Any],
    inPin_parent_tensor: torch.Tensor,
    device: torch.device,
    max_gid: int,
    float_dtype: torch.dtype,
    sigma: float = 3.0,
    topk: int = 256,
    temperature: float = 1.0,
    is_diff_prop: bool = False,
    debug: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Perform timing propagation to calculate arrival times

    Args:
        timing_tensors: Dictionary of timing tensors
        level_2_collaterals: Precomputed timing collaterals by level
        inPin_parent_tensor: Tensor mapping input pins to their drivers
        device: Computation device
        max_gid: Maximum node ID in the graph
        float_dtype: Floating point precision
        sigma: Sigma multiplier for statistical timing
        topk: Number of paths to track per endpoint
        temperature: Temperature for softmax operations

    Returns:
        Updated dictionary of timing tensors
    """
    # Ensure required tensors exist
    assert 'sp_std_tensor' in timing_tensors, 'timing_tensors missing sp_std_tensor'

    # Start propagation
    start_time = time.time()
    if is_diff_prop:
        temperature_tensor = torch.tensor([temperature], dtype=float_dtype).to(device)
    else:
        temperature_tensor = None

    # Execute arrival time propagation
    (
        timing_tensors['Gid_2_rise_arrival'],
        timing_tensors['Gid_2_rise_arrival_mean'],
        timing_tensors['Gid_2_rise_arrival_std'],
        timing_tensors['Gid_2_rise_startpoints'],
        timing_tensors['Gid_2_fall_arrival'],
        timing_tensors['Gid_2_fall_arrival_mean'],
        timing_tensors['Gid_2_fall_arrival_std'],
        timing_tensors['Gid_2_fall_startpoints']
    ) = cuda_arrival_propagate_pocv(
        timing_tensors['sp_mean_tensor'],
        timing_tensors['sp_std_tensor'],
        level_2_collaterals,
        inPin_parent_tensor,
        device,
        max_gid,
        timing_tensors['Gid_2_rise_arrival'],
        timing_tensors['Gid_2_rise_arrival_mean'],
        timing_tensors['Gid_2_rise_arrival_std'],
        timing_tensors['Gid_2_rise_startpoints'],
        timing_tensors['Gid_2_fall_arrival'],
        timing_tensors['Gid_2_fall_arrival_mean'],
        timing_tensors['Gid_2_fall_arrival_std'],
        timing_tensors['Gid_2_fall_startpoints'],
        float_dtype,
        timing_tensors['valid_sps'],
        topK=topk,
        is_diff_prop=is_diff_prop,
        temperature_tensor=temperature_tensor
    )

    print(f"[timing propagation] completed in {time.time() - start_time:.2f} seconds")

    # Calculate slack values
    wns, tns = calculate_slack(
        timing_tensors['Gid_2_rise_slack'],
        timing_tensors['Gid_2_fall_slack'],
        timing_tensors['Gid_2_slack'],
        timing_tensors['Gid_2_rise_arrival'],
        timing_tensors['Gid_2_fall_arrival'],
        timing_tensors['ep_rise_required_truth'],
        timing_tensors['ep_fall_required_truth'],
        timing_tensors['dest_node_tensor'],
        topk
    )

    timing_tensors['Gid_2_max_arrival_mean'] = torch.max(timing_tensors['Gid_2_rise_arrival_mean'], timing_tensors['Gid_2_fall_arrival_mean'])
    timing_tensors['Gid_2_max_arrival_std'] = torch.max(timing_tensors['Gid_2_rise_arrival_std'], timing_tensors['Gid_2_fall_arrival_std'])

    if debug:
        # Debug: Check how many nodes have valid arrival times
        valid_rise = (~torch.isinf(timing_tensors['Gid_2_rise_arrival_mean'])).sum().item()
        valid_fall = (~torch.isinf(timing_tensors['Gid_2_fall_arrival_mean'])).sum().item()
        valid_max = (~torch.isinf(timing_tensors['Gid_2_max_arrival_mean'])).sum().item()
        print(f"[propagate_arrival_times] Valid arrivals - rise: {valid_rise}, fall: {valid_fall}, max: {valid_max}")
        print(f"                          (tensor sized for max_gid={max_gid}")

    return timing_tensors, wns, tns


def save_arrival_tensors(
    timing_tensors: Dict[str, torch.Tensor],
    save_dir: str
) -> None:
    """
    Save arrival time tensors to disk

    Args:
        timing_tensors: Dictionary of timing tensors
        save_dir: Directory to save tensors
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save key tensor results
    torch.save(timing_tensors['Gid_2_rise_startpoints'], os.path.join(save_dir, 'Gid_2_rise_startpoints.pt'))
    torch.save(timing_tensors['Gid_2_fall_startpoints'], os.path.join(save_dir, 'Gid_2_fall_startpoints.pt'))
    torch.save(timing_tensors['Gid_2_rise_arrival_mean'], os.path.join(save_dir, 'Gid_2_rise_arrival_mean.pt'))
    torch.save(timing_tensors['Gid_2_fall_arrival_mean'], os.path.join(save_dir, 'Gid_2_fall_arrival_mean.pt'))
    torch.save(timing_tensors['Gid_2_rise_arrival_std'], os.path.join(save_dir, 'Gid_2_rise_arrival_std.pt'))
    torch.save(timing_tensors['Gid_2_fall_arrival_std'], os.path.join(save_dir, 'Gid_2_fall_arrival_std.pt'))
    torch.save(timing_tensors['Gid_2_rise_arrival'], os.path.join(save_dir, 'Gid_2_rise_arrival.pt'))
    torch.save(timing_tensors['Gid_2_fall_arrival'], os.path.join(save_dir, 'Gid_2_fall_arrival.pt'))
    torch.save(timing_tensors['Gid_2_slack'], os.path.join(save_dir, 'Gid_2_slack.pt'))
    torch.save(timing_tensors['Gid_2_rise_slack'], os.path.join(save_dir, 'Gid_2_rise_slack.pt'))
    torch.save(timing_tensors['Gid_2_fall_slack'], os.path.join(save_dir, 'Gid_2_fall_slack.pt'))


def get_critical_paths(
    timing_tensors: Dict[str, torch.Tensor],
    dest_nodes: Set[int],
    gid_to_pin_map: Dict[int, str],
    topk: int = 1,
    max_paths: int = 10
) -> List[Dict]:
    """
    Extract critical paths from timing results

    Args:
        timing_tensors: Dictionary of timing tensors
        dest_nodes: Set of destination nodes
        gid_to_pin_map: Mapping from graph IDs to pin names
        topk: Number of paths to consider per endpoint
        max_paths: Maximum number of paths to return

    Returns:
        List of path dictionaries with slack, pins, etc.
    """
    # Get slack values for destination nodes
    dest_node_tensor = timing_tensors['dest_node_tensor']
    if topk > 1:
        dest_slacks = timing_tensors['Gid_2_slack'][dest_node_tensor]
    else:
        dest_slacks = timing_tensors['Gid_2_slack'][dest_node_tensor].unsqueeze(1)

    # Filter out invalid slacks
    valid_mask = ~torch.isinf(dest_slacks)

    # Sort by slack (most negative first)
    paths = []
    if valid_mask.any():
        values, indices = torch.sort(dest_slacks[valid_mask])

        # Limit to max_paths
        values = values[:max_paths]
        indices = indices[:max_paths]

        # Build path information
        for i, slack in enumerate(values):
            idx = indices[i]
            dest_idx = dest_node_tensor[idx]

            # Determine if rise or fall is more critical
            rise_slack = timing_tensors['Gid_2_rise_slack'][dest_idx]
            fall_slack = timing_tensors['Gid_2_fall_slack'][dest_idx]

            is_rise = rise_slack <= fall_slack

            # Get startpoint
            if is_rise:
                sp_idx = timing_tensors['Gid_2_rise_startpoints'][dest_idx]
                arrival = timing_tensors['Gid_2_rise_arrival'][dest_idx]
                required = timing_tensors['ep_rise_required_truth'][dest_idx]
            else:
                sp_idx = timing_tensors['Gid_2_fall_startpoints'][dest_idx]
                arrival = timing_tensors['Gid_2_fall_arrival'][dest_idx]
                required = timing_tensors['ep_fall_required_truth'][dest_idx]

            # Create path dictionary
            path = {
                'slack': slack.item(),
                'arrival': arrival.item(),
                'required': required.item(),
                'startpoint': gid_to_pin_map[sp_idx.item()],
                'endpoint': gid_to_pin_map[dest_idx.item()],
                'edge': 'rise' if is_rise else 'fall'
            }

            paths.append(path)

    return paths


def write_analysis_csv(
    output_dir: str,
    true_slacks: torch.Tensor,
    pred_slacks: torch.Tensor,
    dest_nodes: torch.Tensor,
    Gid_2_pinName: Dict[int, str]
) -> None:
    """
    Write analysis results to CSV file

    Args:
        output_dir: Directory to save the CSV file
        true_slacks: Ground truth slack values
        pred_slacks: Predicted slack values
        dest_nodes: Destination node IDs
        Gid_2_pinName: Mapping from graph IDs to pin names
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate statistics
    error = pred_slacks - true_slacks
    abs_error = torch.abs(error)
    rel_error = abs_error / torch.max(torch.abs(true_slacks), torch.tensor(1e-6))

    # Create CSV file
    csv_path = os.path.join(output_dir, "timing_analysis.csv")
    with open(csv_path, 'w') as f:
        # Write header
        f.write("pin_name,true_slack,pred_slack,abs_error,rel_error\n")

        # Write data
        for i in range(len(dest_nodes)):
            node_id = dest_nodes[i].item()
            pin_name = Gid_2_pinName[node_id]
            f.write(f"{pin_name},{true_slacks[i].item():.6f},{pred_slacks[i].item():.6f},"
                   f"{abs_error[i].item():.6f},{rel_error[i].item():.6f}\n")

    # Calculate and print summary statistics
    mean_abs_error = torch.mean(abs_error).item()
    max_abs_error = torch.max(abs_error).item()
    rmse = torch.sqrt(torch.mean(torch.square(error))).item()
    corr = torch.corrcoef(torch.stack([pred_slacks, true_slacks]))[0, 1].item()

    # Write summary file
    summary_path = os.path.join(output_dir, "timing_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Number of endpoints: {len(dest_nodes)}\n")
        f.write(f"Mean absolute error: {mean_abs_error:.6f}\n")
        f.write(f"Max absolute error: {max_abs_error:.6f}\n")
        f.write(f"Root mean square error: {rmse:.6f}\n")
        f.write(f"Correlation coefficient: {corr:.6f}\n")

    print(f"Analysis results written to {csv_path}")
    print(f"Summary statistics written to {summary_path}")


def propagate_subgraph_collateral(
    level_2_collaterals: Dict[int, Any],
    cone_gid_list: torch.Tensor,
    gid2local_map: Dict[int, int],
    sp_mean_tensor: torch.Tensor,
    sp_std_tensor: torch.Tensor,
    sigma: float = 3.0,
    inPinMod: int = 1,
    endpoint_globals: List[int] = None
):
    """Compute rise/fall arrivals for a cone using CUDA kernels only.

    Args:
        level_2_collaterals : per-cone collateral dict (from precompute_collaterals subgraph mode)
        cone_gid_list       : global-gid list (maps local index→global)
        sp_mean/std_tensors : full-chip tensors (read only)
        sigma               : sigma multiplier (default 3)
        inPinMod            : parity for input-pin levels (default 1)
        endpoint_globals    : list of global gids for endpoint calculation

    Returns (rise_mean, fall_mean, endpoint_dict) where rise/fall_mean are **local** tensors (size |cone|)
    and endpoint_dict maps *global gid* → arrival (mean+σ·std).
    """
    import src.installed_ops.sta_compute_arrival.compute_arrival as compute_arrival

    device = sp_mean_tensor.device
    float_dtype = sp_mean_tensor.dtype
    sigma_tensor = torch.tensor([sigma], dtype=float_dtype, device=device)

    N = cone_gid_list.size(0)
    rise_mean = torch.zeros(N, dtype=float_dtype, device=device)
    rise_std  = torch.zeros(N, dtype=float_dtype, device=device)
    fall_mean = torch.zeros(N, dtype=float_dtype, device=device)
    fall_std  = torch.zeros(N, dtype=float_dtype, device=device)

    # CRITICAL: Validate all node IDs in the collateral are within bounds
    max_allowed_idx = sp_mean_tensor.size(0) - 1
    all_nodes = set()
    
    # Collect all node IDs from all levels
    for lvl, content in level_2_collaterals.items():
        if lvl == 1:
            all_nodes.update(content.tolist())
        elif isinstance(content, list) and len(content) > 0:
            # Input pin levels
            if isinstance(content[0], torch.Tensor):
                all_nodes.update(content[0].tolist())
            # Output pin levels - check multiple positions
            if len(content) > 10 and isinstance(content[10], (list, torch.Tensor)):
                if isinstance(content[10], torch.Tensor):
                    all_nodes.update(content[10].tolist())
                else:
                    all_nodes.update(content[10])
    
    # Check for out of bounds nodes
    invalid_nodes = [n for n in all_nodes if n > max_allowed_idx or n < 0]
    if invalid_nodes:
        print(f"\n[CRITICAL ERROR] Found {len(invalid_nodes)} invalid node IDs in collateral!")
        print(f"  Max allowed index: {max_allowed_idx}")
        print(f"  Invalid nodes sample: {sorted(invalid_nodes)[:10]}")
        print(f"  This indicates a problem with subgraph merging!")
        raise RuntimeError(f"Invalid node IDs found: min={min(invalid_nodes)}, max={max(invalid_nodes)}, allowed_max={max_allowed_idx}")

    # --- level-1 initialisation ----------------------------------
    lvl1_nodes: torch.Tensor = level_2_collaterals[1]
    
    # DEBUG: Check bounds before accessing
    print(f"\n[SUBGRAPH INIT DEBUG]")
    print(f"  Level-1 nodes: {len(lvl1_nodes)} nodes")
    print(f"  sp_mean_tensor size: {sp_mean_tensor.size()}")
    print(f"  Max node ID in lvl1: {lvl1_nodes.max().item() if lvl1_nodes.numel() > 0 else 'empty'}")
    print(f"  Min node ID in lvl1: {lvl1_nodes.min().item() if lvl1_nodes.numel() > 0 else 'empty'}")
    
    # Check if any indices are out of bounds
    if lvl1_nodes.numel() > 0:
        max_allowed_idx = sp_mean_tensor.size(0) - 1
        out_of_bounds = lvl1_nodes[lvl1_nodes > max_allowed_idx]
        if out_of_bounds.numel() > 0:
            print(f"  ERROR: {out_of_bounds.numel()} nodes have indices > {max_allowed_idx}")
            print(f"  Out of bounds indices: {out_of_bounds[:5].tolist()}")
            print(f"  This likely means the merging created invalid node IDs!")
            # Don't access the tensor with invalid indices
            valid_mask = lvl1_nodes <= max_allowed_idx
            lvl1_nodes = lvl1_nodes[valid_mask]
            print(f"  Filtered to {len(lvl1_nodes)} valid nodes")
    
    lvl1_local_idx = torch.tensor([gid2local_map[int(g)] for g in lvl1_nodes.tolist()],
                                  dtype=torch.int64, device=device)
    lvl1_means = sp_mean_tensor[lvl1_nodes].to(device)
    lvl1_stds  = sp_std_tensor[lvl1_nodes].to(device)
    nan_mask = torch.isinf(lvl1_means)
    lvl1_means = torch.where(nan_mask, torch.zeros_like(lvl1_means), lvl1_means)
    lvl1_stds  = torch.where(nan_mask, torch.zeros_like(lvl1_stds), lvl1_stds)

    print(f"  First few level-1 node GIDs: {lvl1_nodes[:5].tolist() if lvl1_nodes.numel() >= 5 else lvl1_nodes.tolist()}")
    print(f"  Initialized to 0: {nan_mask.sum().item()} nodes")
    print(f"  Non-zero initialization: {(~nan_mask).sum().item()} nodes")

    rise_mean.index_copy_(0, lvl1_local_idx, lvl1_means)
    fall_mean.index_copy_(0, lvl1_local_idx, lvl1_means)
    rise_std .index_copy_(0, lvl1_local_idx, lvl1_stds)
    fall_std .index_copy_(0, lvl1_local_idx, lvl1_stds)

    # --- forward sweep over subsequent levels -------------------
    sorted_lvls = sorted(k for k in level_2_collaterals.keys() if k != 1)
    for lvl in sorted_lvls:
        coll = level_2_collaterals[lvl]

        if lvl % 2 == inPinMod:
            cur_nodes, parents = coll[0], coll[1]
            r_m, r_s, f_m, f_s = coll[2], coll[3], coll[5], coll[6]

            cur_local = coll[-2]
            parent_local = coll[-1]

            cur_local_idx = cur_local.to(torch.int64)
            parent_idx    = parent_local.to(torch.int64)

            new_rise_mean = rise_mean[parent_idx] + r_m
            new_fall_mean = fall_mean[parent_idx] + f_m
            new_rise_std  = torch.sqrt(rise_std[parent_idx]**2 + r_s**2)
            new_fall_std  = torch.sqrt(fall_std[parent_idx]**2 + f_s**2)

            rise_mean.index_copy_(0, cur_local_idx, new_rise_mean)
            fall_mean.index_copy_(0, cur_local_idx, new_fall_mean)
            rise_std .index_copy_(0, cur_local_idx, new_rise_std)
            fall_std .index_copy_(0, cur_local_idx, new_fall_std)

        else:  # output-pin level (child pins)
            # Collateral layout (same as precompute_collaterals):
            # 0 dup_nodes(list[int]), 1-6 rise/fall tensors, 7 senses, 8 p_indices(int32),
            # 9 node_start_end(int32), 10 c_unique_indices(list[int]),
            # 11 p_idx_unique(int32), 12 p_mapping(int32), 13 c_unique_indices(int64),
            # 14 cellArc_ids(int32), 15 p_local_unique(int64), 16 c_unique_local(int64)

            (dup_nodes,
             c_rise_means, c_rise_stds, c_rise_sigmas,
             c_fall_means, c_fall_stds, c_fall_sigmas,
             senses, p_indices, node_se,
             c_unique_indices, p_idx_unique, p_mapping,
             _, _, p_local_unique, c_unique_local) = (
                 coll[0], coll[1], coll[2], coll[3],
                 coll[4], coll[5], coll[6],
                 coll[7], coll[8], coll[9],
                 coll[10], coll[11], coll[12],
                 coll[13], coll[14], coll[15], coll[16])

            # Parent tensors (advanced indexing expects int64)
            p_local_unique = p_local_unique.to(torch.int64)
            p_rise_means = rise_mean[p_local_unique]
            p_rise_stds  = rise_std [p_local_unique]
            p_fall_means = fall_mean[p_local_unique]
            p_fall_stds  = fall_std [p_local_unique]
            p_rise_start = torch.zeros_like(p_local_unique, dtype=torch.int32, device=device)
            p_fall_start = p_rise_start

            # Compute arrivals for child uniques via CUDA op
            try:
                (cur_rise_means, cur_rise_stds, _, _,
                 cur_fall_means, cur_fall_stds, _, _) = \
                    compute_arrival.ComputeArrivalPOCVWithGrad.apply(
                        p_rise_means, p_rise_stds, p_rise_start,
                        p_fall_means, p_fall_stds, p_fall_start,
                        c_rise_means, c_rise_stds, c_rise_sigmas,
                        c_fall_means, c_fall_stds, c_fall_sigmas,
                        senses, node_se, sigma_tensor,
                        p_indices, p_mapping,
                        torch.tensor([1.0], dtype=float_dtype, device=device))
            except Exception as e:
                print(f"[propagate_subgraph_collateral] CUDA compute error: {e}")
                import ipdb; ipdb.set_trace()

            # Update local tensors at child unique positions
            c_unique_idx = c_unique_local.to(torch.int64)
            rise_mean.index_copy_(0, c_unique_idx, cur_rise_means)
            rise_std .index_copy_(0, c_unique_idx, cur_rise_stds)
            fall_mean.index_copy_(0, c_unique_idx, cur_fall_means)
            fall_std .index_copy_(0, c_unique_idx, cur_fall_stds)

    # ---- endpoints ------------------------------------------------
    if endpoint_globals is None:
        max_lvl = max(level_2_collaterals.keys())
        end_globals = level_2_collaterals[max_lvl][0].tolist() if max_lvl % 2 == inPinMod else level_2_collaterals[max_lvl][10]
    else:
        end_globals = endpoint_globals

    # Vectorised endpoint calculation
    end_idx_tensor = torch.tensor([gid2local_map[int(g)] for g in end_globals], dtype=torch.int64, device=device)
    rise_end = rise_mean[end_idx_tensor] + sigma*rise_std[end_idx_tensor]
    fall_end = fall_mean[end_idx_tensor] + sigma*fall_std[end_idx_tensor]
    combined  = torch.minimum(rise_end, fall_end)

    endpoint_arr = {int(g): float(arr) for g, arr in zip(end_globals, combined.tolist())}

    return rise_mean, fall_mean, endpoint_arr



# ===  NEW UTILITIES: multi-cone merging & propagation  =======================
# The goal is to accelerate local what-if analyses by merging several mutually
# disjoint sub-graphs (cones) into a single collateral dict so that the CUDA
# kernels are launched once instead of once-per-cone.  The input for every
# cone is exactly the tuple returned by `precompute_collaterals(..., subgraph=True)`
# i.e. (level_2_collaterals, net_arc_2_collateral_loc, cell_arc_2_collateral_loc,
#       cellArcId_2_cellName, cellArcKey_2_cellArcId, cellArcId_2_cellArcKey,
#       netArcId_2_inCellName, netArcId_2_outCellName, netArcKey_2_netArcId,
#       netArcId_2_netArcKey, cone_gid_list, gid2local_map)
#
# These helpers assume the cones are node-disjoint – this is the normal case
# after using `select_nonoverlapping_khop`. If overlap occurs we keep the first
# occurrence and silently ignore subsequent duplicates.

def _merge_subgraph_collaterals(
    subgraph_tuples: List[Tuple[Any, ...]],
    device: torch.device,
    inPinMod: int = 1,
):
    """High-throughput merge of many sub-graph collateral tuples.

    The previous implementation called `torch.cat` inside the per-cone loop,
    leading to O(N_cones) kernel launches *per tensor field*.  Here we buffer
    every piece first and concatenate once, reducing launches to O(#levels).
    In addition we vectorise gid-mapping with dense lookup tensors instead of
    Python comprehensions.
    """

    assert subgraph_tuples, "No subgraph tuples supplied"

    # ------------------------------------------------------------------
    # 1.  Synthetic Gid assignment – ***STRICT per-cone uniqueness***
    # ------------------------------------------------------------------
    # Requirement:
    #   !!!  A node appearing in *different* cones with the same global Gid
    #   !!!  MUST be treated as TWO *distinct* timing variables in the
    #   !!!  merged collateral.
    #   Therefore we generate an *individual* dense map for every cone and
    #   never reuse synthetic IDs across cones.

    new2orig: List[int] = []          # synthetic_gid -> original_gid
    new2cone: List[int] = []          # synthetic_gid -> cone_idx
    cone_old2new: List[torch.Tensor] = []  # per-cone dense mapping

    max_orig_gid = max(int(tpl[-2].max()) for tpl in subgraph_tuples)

    next_gid = 0
    for cone_idx, tpl in enumerate(subgraph_tuples):
        cone_gids: torch.Tensor = tpl[-2].to(torch.int64)  # global gids (unique within cone)
        n_nodes = cone_gids.numel()

        # Build dense mapping for this cone -------------------------------
        dense_map = torch.full((max_orig_gid + 1,), -1, dtype=torch.int32, device=device)
        dense_map[cone_gids] = torch.arange(next_gid, next_gid + n_nodes, dtype=torch.int32, device=device)
        cone_old2new.append(dense_map)

        # Record synthetic ↔ metadata relations
        new2orig.extend(cone_gids.tolist())
        new2cone.extend([cone_idx] * n_nodes)

        next_gid += n_nodes

    merged_cone_gid_list = torch.arange(next_gid, dtype=torch.int32, device=device)
    gid2local = {int(i): int(i) for i in merged_cone_gid_list.tolist()}  # identity (synthetic idx = local idx)
    new2orig_tensor = torch.tensor(new2orig, dtype=torch.int32, device=device)
    new2cone_tensor = torch.tensor(new2cone, dtype=torch.int32, device=device)

    # ------------------------------------------------------------------
    # 2.  Buffer per-level fields & collect per-cone endpoints
    # ------------------------------------------------------------------
    from collections import defaultdict

    buf = defaultdict(lambda: defaultdict(list))  # level -> field -> list

    def push(level, key, tensor_or_list):
        buf[level][key].append(tensor_or_list)

    # Accumulate original-gid endpoints for each cone
    endpoints_global: List[int] = []

    for cone_idx, tpl in enumerate(subgraph_tuples):
        lvl2 = tpl[0]
        # Per-cone dense mapping – ensures duplicate global Gids across
        # cones receive *different* synthetic IDs.
        mp_dense = cone_old2new[cone_idx]

        # Identify endpoints of this cone (its own last level) ------------
        max_lvl_cone = max(lvl2.keys())

        def _collect_endpoints(lvl: int):
            """Return *original* endpoint Gids for level *lvl* (as 1-D tensor).

            For output-pin levels the canonical set is `coll[10]` (unique
            child pins).  However, in degenerate cones this list can be
            empty (e.g., when every child pin was pruned).  In that case we
            fall back to `coll[0]` which holds the duplicated child pins.
            Likewise, for input-pin levels we use `coll[0]` directly.
            """
            if lvl % 2 == inPinMod:
                # Input-pin level – endpoints stored as first entry
                ep = lvl2[lvl][0]
                return ep if isinstance(ep, torch.Tensor) else torch.tensor(ep, dtype=torch.int64, device=device)
            # Output-pin level
            coll = lvl2[lvl]
            preferred = coll[10]
            assert len(preferred) > 0, f"Cone {cone_idx} has no endpoints at level {lvl}"
            return preferred if isinstance(preferred, torch.Tensor) else torch.tensor(preferred, dtype=torch.int64, device=device)

        cone_endpoints_orig = _collect_endpoints(max_lvl_cone)

        # Map original Gids to synthetic ones
        cone_endpoints_syn = mp_dense[cone_endpoints_orig.to(torch.int64)].to(torch.int32)

        # ------------------------------------------------------------------
        # Robustness: ensure each cone contributes *at least* one endpoint.
        # It can happen (rarely) that the primary extraction is empty, e.g.,
        # when the last level carries no timing-visible pins after pruning.
        # In that case we walk back towards level-1 until we hit a non-empty
        # candidate set.
        # ------------------------------------------------------------------
        # Filter out unmapped (-1) ids before emptiness check
        cone_endpoints_syn = cone_endpoints_syn[cone_endpoints_syn >= 0]

        if cone_endpoints_syn.numel() == 0:
            for lvl_fb in sorted((l for l in lvl2.keys() if l < max_lvl_cone), reverse=True):
                cand_orig = _collect_endpoints(lvl_fb)
                cand_syn = mp_dense[cand_orig.to(torch.int64)].to(torch.int32)
                cand_syn = cand_syn[cand_syn >= 0]
                if cand_syn.numel() > 0:
                    cone_endpoints_syn = cand_syn
                    break

        # If still empty, emit warning and continue (cone will be ignored later)
        if cone_endpoints_syn.numel() == 0:
            print(f"[merge_subgraph_collaterals] warning: cone {cone_idx} has no endpoints – it will be skipped")
        else:
            endpoints_global.extend(cone_endpoints_syn.tolist())

        for lvl, coll in lvl2.items():
            if lvl == 1:
                push(lvl, 'clk', mp_dense[coll.to(torch.int64)].to(torch.int64))
                continue

            if lvl % 2 == inPinMod:  # input pin level
                cur_nodes  = mp_dense[coll[0].to(torch.int64)].to(torch.int64)
                parents    = mp_dense[coll[1].to(torch.int64)].to(torch.int64)

                push(lvl, 'cur', cur_nodes)
                push(lvl, 'par', parents)
                for idx, tag in enumerate(('rm','rs','rsig','fm','fs','fsig')):
                    push(lvl, tag, coll[2+idx])
                push(lvl, 'net_ids', coll[8])   # python list, keep as list
            else:                   # output-pin (child) level
                dup_nodes      = mp_dense[torch.tensor(coll[0], dtype=torch.int64, device=device)].tolist()
                p_indices_m    = mp_dense[coll[8].to(torch.int64)].to(torch.int32)
                c_unique_m     = mp_dense[torch.tensor(coll[10], dtype=torch.int64, device=device)].tolist()

                # shift node_start_end later – store original + cone_idx for now
                push(lvl, 'dup_nodes', dup_nodes)
                for idx, tag in enumerate(('rm','rs','rsig','fm','fs','fsig')):
                    push(lvl, tag, coll[1+idx])
                push(lvl, 'senses', coll[7])
                push(lvl, 'p_indices', p_indices_m)
                push(lvl, 'node_se', (coll[9], cone_idx))
                push(lvl, 'c_unique_list', c_unique_m)
                push(lvl, 'cellArc_ids', coll[14])

    # ------------------------------------------------------------------
    # 3.  Finalise each level (single concat per tensor field)
    # ------------------------------------------------------------------
    merged = {}

    for lvl, fields in buf.items():
        if lvl == 1:
            merged[lvl] = torch.cat(fields['clk'])
            continue

        if lvl % 2 == inPinMod:
            # Input-pin level -----------------------------------------------------------------
            cur = torch.cat(fields['cur'])
            par = torch.cat(fields['par'])
            cur_local = cur.to(torch.int64)
            par_local = par.to(torch.int64)

            tensors_cat = [torch.cat(fields[tag]) for tag in ('rm','rs','rsig','fm','fs','fsig')]
            net_ids = [nid for sub in fields['net_ids'] for nid in sub]

            merged[lvl] = [cur, par, *tensors_cat, net_ids, cur_local, par_local]
        else:
            # Output-pin level ---------------------------------------------------------------
            dup_nodes = [d for sub in fields['dup_nodes'] for d in sub]

            c_rm, c_rs, c_rsig, c_fm, c_fs, c_fsig = [torch.cat(fields[tag]) for tag in
                                                      ('rm','rs','rsig','fm','fs','fsig')]
            senses = torch.cat(fields['senses'])
            p_indices = torch.cat(fields['p_indices'])

            # Reconstruct node_start_end index array across cones.
            # Each individual cone's `se` tensor has length (#unique_child + 1) and
            # starts with 0.  For the merged cone we must preserve a single leading 0
            # and then append the shifted slice[1:] of each subsequent cone.
            # This guarantees:
            #   len(node_se_shift) == len(c_unique_all) + 1
            # which is required by the CUDA arrival kernel.
            node_se_shift: List[int] = [0]
            offset: int = 0
            for se, _ in fields['node_se']:
                # `se` is a 1-D tensor of shape (n_child_unique + 1,).  The final
                # value equals n_child_unique for that cone.
                se_int = se.to(torch.int32)
                node_se_shift.extend((se_int[1:] + offset).tolist())
                offset += int(se_int[-1].item())
            node_se_shift = torch.tensor(node_se_shift, dtype=torch.int32, device=device)

            c_unique_all = [u for sub in fields['c_unique_list'] for u in sub]
            c_unique_tensor = torch.tensor(c_unique_all, dtype=torch.int64, device=device)

            # parent uniqueness mapping
            uniq_par = torch.unique(p_indices, sorted=True)
            p_map = torch.full((int(uniq_par[-1])+1,), -1, dtype=torch.int32, device=device)
            p_map[uniq_par] = torch.arange(uniq_par.size(0), dtype=torch.int32, device=device)

            p_local_unique = uniq_par.to(torch.int64)
            c_local_unique = c_unique_tensor.to(torch.int64)

            cellArc_ids = torch.cat(fields['cellArc_ids'])

            merged[lvl] = [dup_nodes,
                           c_rm, c_rs, c_rsig,
                           c_fm, c_fs, c_fsig,
                           senses,
                           p_indices, node_se_shift,
                           c_unique_all,
                           uniq_par, p_map,
                           c_unique_tensor,
                           cellArc_ids,
                           p_local_unique,
                           c_local_unique]

    endpoints_tensor = torch.tensor(endpoints_global, dtype=torch.int32, device=device)
    return merged, merged_cone_gid_list, gid2local, new2orig_tensor, new2cone_tensor, endpoints_tensor


def merge_subgraph_collaterals(
    subgraph_tuples: List[Tuple[Any, ...]],
    device: torch.device,
    inPinMod: int = 1,
):
    """External API: merge a list of subgraph collateral tuples into one.
    Simply forwards to the internal helper and returns the same tuple."""
    return _merge_subgraph_collaterals(subgraph_tuples, device, inPinMod)

# ---------------------------------------------------------------------------
