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
# @file timing/collaterals.py
# @brief CUDA accelerated timing operations

import time
import torch
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import src.installed_ops.sta_compute_arrival.compute_arrival as compute_arrival
import ipdb


def cuda_arrival_propagate_pocv(
        sp_mean_tensor: torch.Tensor,
        sp_std_tensor: torch.Tensor,
        level_2_collaterals: Dict[int, Any],
        inPin_parent_tensor: torch.Tensor,
        device: torch.device,
        num_nodes: int,
        Gid_2_rise_arrival: torch.Tensor,
        Gid_2_rise_arrival_mean: torch.Tensor,
        Gid_2_rise_arrival_std: torch.Tensor,
        Gid_2_rise_startpoints: torch.Tensor,
        Gid_2_fall_arrival: torch.Tensor,
        Gid_2_fall_arrival_mean: torch.Tensor,
        Gid_2_fall_arrival_std: torch.Tensor,
        Gid_2_fall_startpoints: torch.Tensor,
        float_dtype: torch.dtype,
        valid_sps: torch.Tensor,
        temperature_tensor: Optional[torch.Tensor] = None,
        cellId_2_probs: Optional[torch.Tensor] = None,
        cellId_2_deltas: Optional[torch.Tensor] = None,
        Gid_2_cellId: Optional[torch.Tensor] = None,
        sigma: float = 3.0,
        log: bool = True,
        to_assert: bool = False,
        topK: int = 256,
        inPinMod: int = 1,
        is_diff_prop = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Propagate timing information through the circuit using CUDA acceleration.

    Args:
        sp_mean_tensor: Mean arrival times for startpoints
        sp_std_tensor: Standard deviations for startpoints
        level_2_collaterals: Precomputed timing collaterals by level
        inPin_parent_tensor: Tensor mapping input pins to their drivers
        device: Computation device
        num_nodes: Maximum node ID in the graph
        Gid_2_rise_arrival: Tensor to store rise arrival times
        Gid_2_rise_arrival_mean: Tensor to store rise arrival means
        Gid_2_rise_arrival_std: Tensor to store rise arrival standard deviations
        Gid_2_rise_startpoints: Tensor to store rise startpoints
        Gid_2_fall_arrival: Tensor to store fall arrival times
        Gid_2_fall_arrival_mean: Tensor to store fall arrival means
        Gid_2_fall_arrival_std: Tensor to store fall arrival standard deviations
        Gid_2_fall_startpoints: Tensor to store fall startpoints
        float_dtype: Floating point precision
        valid_sps: Tensor indicating valid startpoints
        temperature_tensor: Temperature for softmax operations
        cellId_2_probs: Cell probability tensor for optimization
        cellId_2_deltas: Cell delta tensor for optimization
        Gid_2_cellId: Mapping from graph IDs to cell IDs
        sigma: Sigma multiplier for statistical timing
        log: Whether to log progress
        to_assert: Whether to perform assertions
        topK: Number of paths to track per endpoint
        inPinMod: Input pin modulo for levelization

    Returns:
        Tuple of tensors containing propagated timing information
    """
    sigma_tensor = torch.tensor([sigma], dtype=float_dtype).to(device)

    for level in level_2_collaterals:
        if level == 1:
            start_time = time.time()
            sp_nodes = level_2_collaterals[level]
            sp_means = sp_mean_tensor[sp_nodes]
            sp_stds = sp_std_tensor[sp_nodes]
            if to_assert:
                assert not torch.isinf(sp_means).any() and not torch.isnan(sp_means).any()
                assert not torch.isinf(sp_stds).any() and not torch.isnan(sp_stds).any()
            sp_arrivals = sp_means + sigma * sp_stds
            if topK > 1:
                sp_arrivals = sp_arrivals.unsqueeze(1).expand(-1, topK)
                sp_means = sp_means.unsqueeze(1).expand(-1, topK)
                sp_stds = sp_stds.unsqueeze(1).expand(-1, topK)
                sp_nodes_expanded = sp_nodes.unsqueeze(1).expand(-1, topK).to(torch.int32)
            else:
                sp_nodes_expanded = sp_nodes.to(torch.int32)
            try:
                Gid_2_rise_arrival_mean.index_copy_(0, sp_nodes, sp_means)
            except Exception as e:
                print(f'error: {e}')
                ipdb.set_trace()
            Gid_2_rise_arrival_std.index_copy_(0, sp_nodes, sp_stds)
            Gid_2_rise_startpoints.index_copy_(0, sp_nodes, sp_nodes_expanded)
            Gid_2_rise_arrival.index_copy_(0, sp_nodes, sp_arrivals)
            Gid_2_fall_arrival_mean.index_copy_(0, sp_nodes, sp_means)
            Gid_2_fall_arrival_std.index_copy_(0, sp_nodes, sp_stds)
            Gid_2_fall_startpoints.index_copy_(0, sp_nodes, sp_nodes_expanded)
            Gid_2_fall_arrival.index_copy_(0, sp_nodes, sp_arrivals)
            if log:
                print(f"[fw-sta] level: {level}, # nodes: {len(sp_nodes)}, time: {time.time() - start_time:.2f}s")
        elif level % 2 == inPinMod:  # input pins
            global_start_time = time.time()
            (c_unique_indices, p_indices,
             rise_means, rise_stds, rise_sigmas,
             fall_means, fall_stds, fall_sigmas,
             net_arc_ids) = level_2_collaterals[level]

            if is_diff_prop:
                rise_means.requires_grad_()
                fall_means.requires_grad_()

            if to_assert:
                assert not torch.isinf(rise_means).any() and not torch.isnan(rise_means).any()
                assert not torch.isinf(rise_stds).any() and not torch.isnan(rise_stds).any()
                assert not torch.isinf(rise_sigmas).any() and not torch.isnan(rise_sigmas).any()
                assert not torch.isinf(fall_means).any() and not torch.isnan(fall_means).any()
                assert not torch.isinf(fall_stds).any() and not torch.isnan(fall_stds).any()
                assert not torch.isinf(fall_sigmas).any() and not torch.isnan(fall_sigmas).any()

            # Calculate rise values
            if topK > 1:
                cur_rise_means = rise_means.unsqueeze(1).expand(-1, Gid_2_rise_arrival_mean.size(1)) + Gid_2_rise_arrival_mean[p_indices]
                cur_rise_stds = torch.sqrt(torch.pow(Gid_2_rise_arrival_std[p_indices], 2) +
                                           torch.pow(rise_stds.unsqueeze(1).expand(-1, Gid_2_rise_arrival_std.size(1)), 2))
            else:
                cur_rise_means = rise_means + Gid_2_rise_arrival_mean[p_indices]
                cur_rise_stds = torch.sqrt( torch.pow(Gid_2_rise_arrival_std[p_indices], 2) + torch.pow(rise_stds, 2))

            cur_rise_arrivals = cur_rise_means + sigma * cur_rise_stds
            cur_rise_startpoints = Gid_2_rise_startpoints[p_indices]

            # Update rise tensors
            Gid_2_rise_arrival_mean.index_copy_(0, c_unique_indices, cur_rise_means)
            Gid_2_rise_arrival_std.index_copy_(0, c_unique_indices, cur_rise_stds)
            Gid_2_rise_arrival.index_copy_(0, c_unique_indices, cur_rise_arrivals)
            Gid_2_rise_startpoints.index_copy_(0, c_unique_indices, cur_rise_startpoints)

            # Calculate fall values
            if topK > 1:
                cur_fall_means = fall_means.unsqueeze(1).expand(-1, Gid_2_fall_arrival_mean.size(1)) + Gid_2_fall_arrival_mean[p_indices]
                cur_fall_stds = torch.sqrt(torch.pow(Gid_2_fall_arrival_std[p_indices], 2) +
                                          torch.pow(fall_stds.unsqueeze(1).expand(-1, Gid_2_fall_arrival_std.size(1)), 2))
            else:
                cur_fall_means = fall_means + Gid_2_fall_arrival_mean[p_indices]
                cur_fall_stds = torch.sqrt( torch.pow(Gid_2_fall_arrival_std[p_indices], 2) + torch.pow(fall_stds, 2) )

            cur_fall_arrivals = cur_fall_means + sigma * cur_fall_stds
            cur_fall_startpoints = Gid_2_fall_startpoints[p_indices]

            # Update fall tensors
            Gid_2_fall_arrival_mean.index_copy_(0, c_unique_indices, cur_fall_means)
            Gid_2_fall_arrival_std.index_copy_(0, c_unique_indices, cur_fall_stds)
            Gid_2_fall_arrival.index_copy_(0, c_unique_indices, cur_fall_arrivals)
            Gid_2_fall_startpoints.index_copy_(0, c_unique_indices, cur_fall_startpoints)

            if log:
                print(f'[fw-sta] level: {level}, # nodes: {len(c_unique_indices)}, time: {time.time() - global_start_time:.2f}s')
        else:  # output pins
            (c_duplicated_indices,
             c_rise_means, c_rise_stds, c_rise_sigmas,
             c_fall_means, c_fall_stds, c_fall_sigmas,
             senses, p_indices, node_start_end_idx, c_unique_indices,
             p_idx_unique, p_mapping, c_unique_idx_tensor,
             cellArc_ids) = level_2_collaterals[level]

            if is_diff_prop:
                c_rise_means.requires_grad_()
                c_fall_means.requires_grad_()

            if to_assert:
                assert not torch.isinf(c_rise_means).any() and not torch.isnan(c_rise_means).any()
                assert not torch.isinf(c_rise_stds).any() and not torch.isnan(c_rise_stds).any()
                assert not torch.isinf(c_fall_means).any() and not torch.isnan(c_fall_means).any()
                assert not torch.isinf(c_fall_stds).any() and not torch.isnan(c_fall_stds).any()

            # Get parent values
            p_rise_means = Gid_2_rise_arrival_mean[p_idx_unique]
            p_rise_stds = Gid_2_rise_arrival_std[p_idx_unique]
            p_rise_startpoints = Gid_2_rise_startpoints[p_idx_unique]
            p_fall_means = Gid_2_fall_arrival_mean[p_idx_unique]
            p_fall_stds = Gid_2_fall_arrival_std[p_idx_unique]
            p_fall_startpoints = Gid_2_fall_startpoints[p_idx_unique]

            start_time = time.time()

            if not is_diff_prop:
                (cur_unique_rise_means, cur_unique_rise_stds, cur_unique_rise_startpoints,
                 cur_unique_fall_means, cur_unique_fall_stds, cur_unique_fall_startpoints) = \
                    compute_arrival.ComputeArrivalPOCV.forward(
                        p_rise_means,
                        p_rise_stds,
                        p_rise_startpoints,
                        p_fall_means,
                        p_fall_stds,
                        p_fall_startpoints,
                        c_rise_means,
                        c_rise_stds,
                        c_rise_sigmas,
                        c_fall_means,
                        c_fall_stds,
                        c_fall_sigmas,
                        senses,
                        node_start_end_idx,
                        sigma_tensor,
                        p_indices,
                        p_mapping,
                        valid_sps,
                        topK,
                        float_dtype
                    )
            else:
                (cur_unique_rise_means, cur_unique_rise_stds, cur_unique_rise_startpoints, cur_unique_rise_sigmas,
                 cur_unique_fall_means, cur_unique_fall_stds, cur_unique_fall_startpoints, cur_unique_fall_sigmas) = \
                         compute_arrival.ComputeArrivalPOCVWithGrad.apply(
                             p_rise_means,
                             p_rise_stds,
                             p_rise_startpoints,
                             p_fall_means,
                             p_fall_stds,
                             p_fall_startpoints,
                             c_rise_means, # + cell arc rise delta * probs
                             c_rise_stds,
                             c_rise_sigmas,
                             c_fall_means, # + cell arc fall delta * probs
                             c_fall_stds,
                             c_fall_sigmas,
                             senses,
                             node_start_end_idx,
                             sigma_tensor,
                             p_indices,
                             p_mapping,
                             temperature_tensor
                         )

            if to_assert:
                assert cur_unique_rise_means.size(0) == len(c_unique_indices)
                assert cur_unique_rise_stds.size(0) == len(c_unique_indices)
                assert cur_unique_rise_startpoints.size(0) == len(c_unique_indices)
                assert cur_unique_fall_means.size(0) == len(c_unique_indices)
                assert cur_unique_fall_stds.size(0) == len(c_unique_indices)
                assert cur_unique_fall_startpoints.size(0) == len(c_unique_indices)
                assert not torch.isinf(cur_unique_rise_means).any() and not torch.isnan(cur_unique_rise_means).any()
                assert not torch.isinf(cur_unique_rise_stds).any() and not torch.isnan(cur_unique_rise_stds).any()
                assert not torch.isinf(cur_unique_rise_startpoints).any() and not torch.isnan(cur_unique_rise_startpoints).any()
                assert not torch.isinf(cur_unique_fall_means).any() and not torch.isnan(cur_unique_fall_means).any()
                assert not torch.isinf(cur_unique_fall_stds).any() and not torch.isnan(cur_unique_fall_stds).any()
                assert not torch.isinf(cur_unique_fall_startpoints).any() and not torch.isnan(cur_unique_fall_startpoints).any()

            # Update arrival time tensors
            try:
                Gid_2_rise_arrival_mean.index_copy_(0, c_unique_idx_tensor, cur_unique_rise_means)  # [num_nodes, topK]
                Gid_2_rise_arrival_std.index_copy_(0, c_unique_idx_tensor, cur_unique_rise_stds)    # [num_nodes, topK]
                Gid_2_rise_startpoints.index_copy_(0, c_unique_idx_tensor, cur_unique_rise_startpoints)  # [num_nodes, topK]
                Gid_2_fall_arrival_mean.index_copy_(0, c_unique_idx_tensor, cur_unique_fall_means)  # [num_nodes, topK]
                Gid_2_fall_arrival_std.index_copy_(0, c_unique_idx_tensor, cur_unique_fall_stds)    # [num_nodes, topK]
                Gid_2_fall_startpoints.index_copy_(0, c_unique_idx_tensor, cur_unique_fall_startpoints)  # [num_nodes, topK]
            except:
                ipdb.set_trace()

            # Update total arrival times
            cur_rise_arrivals = cur_unique_rise_means + 3.0 * cur_unique_rise_stds  # [num_nodes, topK]
            cur_fall_arrivals = cur_unique_fall_means + 3.0 * cur_unique_fall_stds  # [num_nodes, topK]
            Gid_2_rise_arrival.index_copy_(0, c_unique_idx_tensor, cur_rise_arrivals)  # [num_nodes, topK]
            Gid_2_fall_arrival.index_copy_(0, c_unique_idx_tensor, cur_fall_arrivals)  # [num_nodes, topK]

            if log:
                print(f'[fw-sta] level: {level}, # valid nodes: {len(c_unique_indices)}')

    return (Gid_2_rise_arrival, Gid_2_rise_arrival_mean, Gid_2_rise_arrival_std, Gid_2_rise_startpoints,
            Gid_2_fall_arrival, Gid_2_fall_arrival_mean, Gid_2_fall_arrival_std, Gid_2_fall_startpoints)

