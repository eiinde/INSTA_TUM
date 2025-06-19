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
# @file timing/pocv.py
# @brief pocv-related timing handling and grad extraction

import os
import torch
from typing import Dict, List, Set, Tuple, Optional, Union, Any

from ..io.serialization import save_pickle, load_pickle, save_torch_tensor, load_torch_tensor


def initialize_timing_tensors(
    sp_attributes: Dict[str, Tuple[float, float, float]],
    ep_attributes: Dict[Tuple[str, str], Tuple],
    pin_to_id_map: Dict[str, int],
    max_gid: int,
    float_dtype: torch.dtype,
    save_dir: str,
    force: bool = False
) -> Tuple[
    torch.Tensor,  # ep_rise_arrival_truth
    torch.Tensor,  # ep_rise_required_truth
    torch.Tensor,  # ep_rise_slack_truth
    torch.Tensor,  # ep_rise_depth_truth
    torch.Tensor,  # ep_fall_arrival_truth
    torch.Tensor,  # ep_fall_required_truth
    torch.Tensor,  # ep_fall_slack_truth
    torch.Tensor,  # ep_fall_depth_truth
    torch.Tensor,  # sp_arrival_truth
    torch.Tensor,  # sp_mean_tensor
    torch.Tensor,  # sp_std_tensor
    Set[int],      # source_nodes
    Set[int]       # dest_nodes
]:
    """
    Initialize timing ground truth tensors from endpoint and startpoint data

    Args:
        sp_attributes: Dictionary mapping startpoint names to timing attributes
        ep_attributes: Dictionary mapping (endpoint, edge) to timing attributes
        pin_to_id_map: Dictionary mapping pin names to graph IDs
        max_gid: Maximum node ID in the graph
        float_dtype: Floating point precision type
        save_dir: Directory to save/load tensors
        force: Whether to force recomputation even if cached results exist

    Returns:
        Tuple of timing tensors and node sets
    """
    # Check for cached results
    cache_path = os.path.join(save_dir, 'ep_rise_arrival_truth.pt')
    if not force and os.path.exists(cache_path):
        # Load cached tensors and node sets
        source_nodes = load_pickle(os.path.join(save_dir, 'source_nodes.pkl'), set())
        dest_nodes = load_pickle(os.path.join(save_dir, 'dest_nodes.pkl'), set())

        ep_rise_arrival_truth = load_torch_tensor(os.path.join(save_dir, 'ep_rise_arrival_truth.pt'))
        ep_rise_required_truth = load_torch_tensor(os.path.join(save_dir, 'ep_rise_required_truth.pt'))
        ep_rise_slack_truth = load_torch_tensor(os.path.join(save_dir, 'ep_rise_slack_truth.pt'))
        ep_rise_depth_truth = load_torch_tensor(os.path.join(save_dir, 'ep_rise_depth_truth.pt'))

        ep_fall_arrival_truth = load_torch_tensor(os.path.join(save_dir, 'ep_fall_arrival_truth.pt'))
        ep_fall_required_truth = load_torch_tensor(os.path.join(save_dir, 'ep_fall_required_truth.pt'))
        ep_fall_slack_truth = load_torch_tensor(os.path.join(save_dir, 'ep_fall_slack_truth.pt'))
        ep_fall_depth_truth = load_torch_tensor(os.path.join(save_dir, 'ep_fall_depth_truth.pt'))

        sp_arrival_truth = load_torch_tensor(os.path.join(save_dir, 'sp_arrival_truth.pt'))
        sp_mean_tensor = load_torch_tensor(os.path.join(save_dir, 'sp_mean_tensor.pt'))
        sp_std_tensor = load_torch_tensor(os.path.join(save_dir, 'sp_std_tensor.pt'))

        return (
            ep_rise_arrival_truth, ep_rise_required_truth, ep_rise_slack_truth, ep_rise_depth_truth,
            ep_fall_arrival_truth, ep_fall_required_truth, ep_fall_slack_truth, ep_fall_depth_truth,
            sp_arrival_truth, sp_mean_tensor, sp_std_tensor,
            source_nodes, dest_nodes
        )

    # Initialize sets for source and destination nodes
    source_nodes = set()
    dest_nodes = set()

    # Initialize timing arrays with default values
    ep_rise_arrival_truth = [float('-inf')] * max_gid
    ep_rise_required_truth = [float('-inf')] * max_gid
    ep_rise_slack_truth = [float('-inf')] * max_gid
    ep_rise_depth_truth = [float('-inf')] * max_gid

    ep_fall_arrival_truth = [float('-inf')] * max_gid
    ep_fall_required_truth = [float('-inf')] * max_gid
    ep_fall_slack_truth = [float('-inf')] * max_gid
    ep_fall_depth_truth = [float('-inf')] * max_gid

    sp_arrival_truth = [float('-inf')] * max_gid
    sp_mean_tensor = [float('-inf')] * max_gid
    sp_std_tensor = [float('-inf')] * max_gid

    # Process endpoint data
    for (epName, rise_fall), ep_attrs in ep_attributes.items():
        if epName not in pin_to_id_map:
            continue

        epId = pin_to_id_map[epName]

        # Store timing values based on edge type
        if rise_fall == 'rise':
            ep_rise_arrival_truth[epId] = ep_attrs[0]   # arrival
            ep_rise_required_truth[epId] = ep_attrs[1]  # required
            ep_rise_slack_truth[epId] = ep_attrs[2]     # slack
            ep_rise_depth_truth[epId] = ep_attrs[3]     # path depth
        else:  # fall
            ep_fall_arrival_truth[epId] = ep_attrs[0]   # arrival
            ep_fall_required_truth[epId] = ep_attrs[1]  # required
            ep_fall_slack_truth[epId] = ep_attrs[2]     # slack
            ep_fall_depth_truth[epId] = ep_attrs[3]     # path depth

        # Track destination nodes
        dest_nodes.add(epId)

    # Process startpoint data
    for spName, sp_attrs in sp_attributes.items():
        if spName not in pin_to_id_map:
            continue

        spId = pin_to_id_map[spName]

        # Store timing values
        sp_arrival_truth[spId] = sp_attrs[0]  # arrival
        sp_mean_tensor[spId] = sp_attrs[1]    # mean
        sp_std_tensor[spId] = sp_attrs[2]     # std

        # Track source nodes
        source_nodes.add(spId)

    # Convert lists to tensors
    ep_rise_arrival_truth = torch.tensor(ep_rise_arrival_truth, dtype=float_dtype)
    ep_rise_required_truth = torch.tensor(ep_rise_required_truth, dtype=float_dtype)
    ep_rise_slack_truth = torch.tensor(ep_rise_slack_truth, dtype=float_dtype)
    ep_rise_depth_truth = torch.tensor(ep_rise_depth_truth, dtype=float_dtype)

    ep_fall_arrival_truth = torch.tensor(ep_fall_arrival_truth, dtype=float_dtype)
    ep_fall_required_truth = torch.tensor(ep_fall_required_truth, dtype=float_dtype)
    ep_fall_slack_truth = torch.tensor(ep_fall_slack_truth, dtype=float_dtype)
    ep_fall_depth_truth = torch.tensor(ep_fall_depth_truth, dtype=float_dtype)

    sp_arrival_truth = torch.tensor(sp_arrival_truth, dtype=float_dtype)
    sp_mean_tensor = torch.tensor(sp_mean_tensor, dtype=float_dtype)
    sp_std_tensor = torch.tensor(sp_std_tensor, dtype=float_dtype)

    # Save results
    save_pickle(source_nodes, os.path.join(save_dir, 'source_nodes.pkl'))
    save_pickle(dest_nodes, os.path.join(save_dir, 'dest_nodes.pkl'))

    save_torch_tensor(ep_rise_arrival_truth, os.path.join(save_dir, 'ep_rise_arrival_truth.pt'))
    save_torch_tensor(ep_rise_required_truth, os.path.join(save_dir, 'ep_rise_required_truth.pt'))
    save_torch_tensor(ep_rise_slack_truth, os.path.join(save_dir, 'ep_rise_slack_truth.pt'))
    save_torch_tensor(ep_rise_depth_truth, os.path.join(save_dir, 'ep_rise_depth_truth.pt'))

    save_torch_tensor(ep_fall_arrival_truth, os.path.join(save_dir, 'ep_fall_arrival_truth.pt'))
    save_torch_tensor(ep_fall_required_truth, os.path.join(save_dir, 'ep_fall_required_truth.pt'))
    save_torch_tensor(ep_fall_slack_truth, os.path.join(save_dir, 'ep_fall_slack_truth.pt'))
    save_torch_tensor(ep_fall_depth_truth, os.path.join(save_dir, 'ep_fall_depth_truth.pt'))

    save_torch_tensor(sp_arrival_truth, os.path.join(save_dir, 'sp_arrival_truth.pt'))
    save_torch_tensor(sp_mean_tensor, os.path.join(save_dir, 'sp_mean_tensor.pt'))
    save_torch_tensor(sp_std_tensor, os.path.join(save_dir, 'sp_std_tensor.pt'))

    return (
        ep_rise_arrival_truth, ep_rise_required_truth, ep_rise_slack_truth, ep_rise_depth_truth,
        ep_fall_arrival_truth, ep_fall_required_truth, ep_fall_slack_truth, ep_fall_depth_truth,
        sp_arrival_truth, sp_mean_tensor, sp_std_tensor,
        source_nodes, dest_nodes
    )


def calculate_slack(
    rise_slack: torch.Tensor,
    fall_slack: torch.Tensor,
    overall_slack: torch.Tensor,
    rise_arrival: torch.Tensor,
    fall_arrival: torch.Tensor,
    rise_required: torch.Tensor,
    fall_required: torch.Tensor,
    dest_nodes: torch.Tensor,
    topk: int = 1
) -> None:
    """
    Calculate slack values based on arrival and required times

    Args:
        rise_slack: Tensor to store rise slack values
        fall_slack: Tensor to store fall slack values
        overall_slack: Tensor to store overall slack values
        rise_arrival: Rise arrival times
        fall_arrival: Fall arrival times
        rise_required: Rise required times for endpoints
        fall_required: Fall required times for endpoints
        dest_nodes: Tensor of destination node IDs
        topk: Number of paths to track per endpoint
    """
    # Handle tensors with or without topK dimension
    if topk > 1:
        # For tensors with topK dimension, we need to handle indexing carefully

        # Calculate rise slack
        if rise_arrival.ndim > 1:
            # Use first path for slack calculation
            rise_slack[dest_nodes] = rise_required[dest_nodes] - rise_arrival[dest_nodes][:, 0]
        else:
            # Handle case where rise_arrival has been reshaped to 1D
            rise_slack[dest_nodes] = rise_required[dest_nodes] - rise_arrival[dest_nodes]

        # Calculate fall slack
        if fall_arrival.ndim > 1:
            # Use first path for slack calculation
            fall_slack[dest_nodes] = fall_required[dest_nodes] - fall_arrival[dest_nodes][:, 0]
        else:
            # Handle case where fall_arrival has been reshaped to 1D
            fall_slack[dest_nodes] = fall_required[dest_nodes] - fall_arrival[dest_nodes]
    else:
        # For 1D tensors, indexing is straightforward
        rise_slack[dest_nodes] = rise_required[dest_nodes] - rise_arrival[dest_nodes]
        fall_slack[dest_nodes] = fall_required[dest_nodes] - fall_arrival[dest_nodes]

    # Fix inf values
    rise_mask = torch.isinf(rise_slack[dest_nodes])
    if rise_mask.any():
        rise_slack[dest_nodes][rise_mask] = float('-inf')

    fall_mask = torch.isinf(fall_slack[dest_nodes])
    if fall_mask.any():
        fall_slack[dest_nodes][fall_mask] = float('-inf')

    # Calculate overall slack (minimum of rise and fall)
    overall_slack[dest_nodes] = torch.minimum(rise_slack[dest_nodes], fall_slack[dest_nodes])

    # Calculate Total Negative Slack (TNS)
    tns_mask = (overall_slack[dest_nodes] < 0)
    inf_mask = torch.isinf(overall_slack[dest_nodes][tns_mask])
    valid_slacks = overall_slack[dest_nodes][tns_mask][~inf_mask]
    tns = valid_slacks.to(torch.float32).sum()

    # Calculate Worst Negative Slack (WNS)
    if valid_slacks.numel() > 0:
        wns = valid_slacks.min()
        print(f"TNS: {tns.item():.4f}, WNS: {wns.item():.4f}")
    else:
        print(f"TNS: {tns.item():.4f}, No negative slack paths")

    return wns, tns


def process_crpr_data(
    crpr_file: str,
    pin_to_id_map: Dict[str, int],
    sigma: float = 3.0
) -> Dict[Tuple[int, int], float]:
    """
    Process Clock Reconvergence Pessimism Removal (CRPR) data

    Args:
        crpr_file: Path to CRPR data file
        pin_to_id_map: Mapping from pin names to graph IDs
        sigma: Sigma multiplier for statistical timing

    Returns:
        Dictionary mapping (from_sp, to_sp) pairs to CRPR values
    """
    # Check if file exists
    if not os.path.exists(crpr_file):
        return {}

    # Read CRPR data
    import polars as pl

    df = pl.read_csv(crpr_file, infer_schema_length=0)
    df = df.with_columns(
        pl.col('mean_value').cast(pl.Float32),
        pl.col('stddev_value').cast(pl.Float32)
    ).drop_nulls()

    # Process into dictionary
    crpr_dict = {}
    for row in df.to_dicts():
        from_sp = row['from_sp_name']
        to_sp = row['to_sp_name']

        if from_sp not in pin_to_id_map or to_sp not in pin_to_id_map:
            continue

        from_id = pin_to_id_map[from_sp]
        to_id = pin_to_id_map[to_sp]

        # Apply CRPR as mean + sigma*std
        crpr_dict[(from_id, to_id)] = row['mean_value'] + sigma * row['stddev_value']

    return crpr_dict


def apply_cppr_correction(
    timing_tensors,
    dest_nodes,
    Gid_2_pinName,
    pinName_2_Gid,
    epName_riseFall_2_spName,
    topK,
    ep_rise_depth_truth,
    ep_rise_slack_truth,
    ep_rise_arrival_truth,
    ep_fall_depth_truth,
    ep_fall_slack_truth,
    ep_fall_arrival_truth,
    to_filter=False
) -> None:
    """
    Apply Clock Reconvergence Pessimism Removal (CRPR) correction to slack values
    """

    temporal_metadata = {
        'd_pins': [],             # Destination pin identifiers
        'true_levels': [],        # Ground truth logical depth metrics
        'true_slacks': [],        # Ground truth slack metrics
        'true_requireds': [],     # Ground truth required time metrics
        'true_arrivals': [],      # Ground truth arrival time metrics
        'pred_arrivals': [],      # Predicted arrival time metrics
        'epNames': [],            # Endpoint identifier strings
        'spNames': [],            # Startpoint identifier strings
        'riseFalls': [],          # Edge transition polarity indicators
    }

    # Initialize statistical reconciliation counters
    temporal_alignment_metrics = {
        'cnt_mismatch': 0,        # Path mismatch count
        'cnt_found_crpr': 0,      # Clock reconvergent pessimism removed count
    }

    for node_index, destination_node_id in enumerate(dest_nodes):
        endpoint_identifier = Gid_2_pinName[destination_node_id]

        if ((endpoint_identifier, 'rise') not in epName_riseFall_2_spName or
            (endpoint_identifier, 'fall') not in epName_riseFall_2_spName):
            continue

        rise_required_time = timing_tensors['rise_required'][destination_node_id].item()
        fall_required_time = timing_tensors['fall_required'][destination_node_id].item()

        if topK > 1:
            rise_startpoint = timing_tensors['rise_startpoints'][destination_node_id][0].item()
            rise_arrival_time = timing_tensors['rise_arrival'][destination_node_id][0]
            fall_startpoint = timing_tensors['fall_startpoints'][destination_node_id][0].item()
            fall_arrival_time = timing_tensors['fall_arrival'][destination_node_id][0]
        else:
            rise_startpoint = timing_tensors['rise_startpoints'][destination_node_id].item()
            rise_arrival_time = timing_tensors['rise_arrival'][destination_node_id]
            fall_startpoint = timing_tensors['fall_startpoints'][destination_node_id].item()
            fall_arrival_time = timing_tensors['fall_arrival'][destination_node_id]

        rise_slack = rise_required_time - rise_arrival_time
        fall_slack = fall_required_time - fall_arrival_time

        transition_polarity = 'rise' if rise_slack < fall_slack else 'fall'
        if transition_polarity == 'rise':  # Handle rising transition paths
            if epName_riseFall_2_spName[(endpoint_identifier, 'rise')] != Gid_2_pinName[rise_startpoint]:
                golden_startpoint_id = pinName_2_Gid[epName_riseFall_2_spName[(endpoint_identifier, 'rise')]]
                path_found = False
                temporal_alignment_metrics['cnt_mismatch'] += 1

                if topK > 1:
                    for path_index in range(topK):
                        current_startpoint = timing_tensors['rise_startpoints'][destination_node_id][path_index].item()
                        if golden_startpoint_id == current_startpoint:
                            rise_startpoint = golden_startpoint_id
                            rise_arrival_time = timing_tensors['rise_arrival'][destination_node_id][path_index]
                            path_found = True
                            temporal_alignment_metrics['cnt_found_crpr'] += 1
                            break
                    if to_filter and not path_found:
                        continue
        else:
            if epName_riseFall_2_spName[(endpoint_identifier, 'fall')] != Gid_2_pinName[fall_startpoint]:
                golden_startpoint_id = pinName_2_Gid[epName_riseFall_2_spName[(endpoint_identifier, 'fall')]]
                path_found = False
                temporal_alignment_metrics['cnt_mismatch'] += 1
                if topK > 1:
                    for path_index in range(topK):
                        current_startpoint = timing_tensors['fall_startpoints'][destination_node_id][path_index].item()
                        if golden_startpoint_id == current_startpoint:
                            fall_startpoint = golden_startpoint_id
                            fall_arrival_time = timing_tensors['fall_arrival'][destination_node_id][path_index]
                            path_found = True
                            temporal_alignment_metrics['cnt_found_crpr'] += 1
                            break
                    if to_filter and not path_found:
                        continue

        # transition polarity
        if transition_polarity == 'rise':
            temporal_metadata['true_levels'].append(ep_rise_depth_truth[destination_node_id].item())
            temporal_metadata['true_slacks'].append(ep_rise_slack_truth[destination_node_id].item())
            temporal_metadata['true_requireds'].append(rise_required_time)
            temporal_metadata['true_arrivals'].append(ep_rise_arrival_truth[destination_node_id].item())
            temporal_metadata['pred_arrivals'].append(rise_arrival_time)
            temporal_metadata['spNames'].append(Gid_2_pinName[rise_startpoint])
            temporal_metadata['riseFalls'].append('rise')
        else:
            temporal_metadata['true_levels'].append(ep_fall_depth_truth[destination_node_id].item())
            temporal_metadata['true_slacks'].append(ep_fall_slack_truth[destination_node_id].item())
            temporal_metadata['true_requireds'].append(fall_required_time)
            temporal_metadata['true_arrivals'].append(ep_fall_arrival_truth[destination_node_id].item())
            temporal_metadata['pred_arrivals'].append(fall_arrival_time)
            temporal_metadata['spNames'].append(Gid_2_pinName[fall_startpoint])
            temporal_metadata['riseFalls'].append('fall')

        temporal_metadata['d_pins'].append(destination_node_id)
        temporal_metadata['epNames'].append(endpoint_identifier)

    print(f'# of valid eps: {len(temporal_metadata["d_pins"])}')
    if temporal_alignment_metrics['cnt_mismatch']:
        # Report path reconciliation statistics
        print(f'# ep mismatch: {temporal_alignment_metrics["cnt_mismatch"]}, '
              f'# corrected by topK crpr: {temporal_alignment_metrics["cnt_found_crpr"]}, '
              f'rate: {temporal_alignment_metrics["cnt_found_crpr"] / temporal_alignment_metrics["cnt_mismatch"]:.2f}')
    else:
        print("no initial ep mismatch")

    return temporal_metadata


def extract_cellArc_grad(
    level_2_collaterals,
    cellArcId_2_cellArcKey,
    inPinMod=1
) -> Dict[Tuple[str, str], List]:
    cellArc_2_riseFallGrads = {}
    for level in level_2_collaterals:
        if level != 1 and level % 2 != inPinMod:
            rise_means    = level_2_collaterals[level][1]
            assert rise_means.grad is not None, "objective not diff yet"
            fall_means    = level_2_collaterals[level][4]
            cell_arc_ids  = level_2_collaterals[level][-1]
            non_zero_mask = (rise_means.grad != 0) | (fall_means.grad != 0)
            for arc_id, rise_grad, fall_grad in zip(
                    cell_arc_ids[non_zero_mask], rise_means.grad[non_zero_mask], fall_means.grad[non_zero_mask]):
                fromPinName, toPinName, sense = cellArcId_2_cellArcKey[arc_id.item()]
                cellArc_2_riseFallGrads[fromPinName, toPinName] = [ rise_grad.item(), fall_grad.item() ]
    return cellArc_2_riseFallGrads


def extract_netArc_grad(
    level_2_collaterals,
    netArcId_2_netArcKey,
    inPinMod=1
) -> Dict[Tuple[str, str], List]:
    netArc_2_riseFallGrads = {}
    for level in level_2_collaterals:
        if level != 1 and level % 2 == inPinMod:
            rise_means  = level_2_collaterals[level][2]
            fall_means  = level_2_collaterals[level][5]
            net_arc_ids = level_2_collaterals[level][-1].to(torch.int32).to(rise_means.device)
            non_zero_mask = (rise_means.grad != 0) | (fall_means.grad != 0)
            for arc_id, rise_grad, fall_grad in zip(
                net_arc_ids[non_zero_mask], rise_means.grad[non_zero_mask], fall_means.grad[non_zero_mask]):
                netArc_2_riseFallGrads[netArcId_2_netArcKey[arc_id.item()]] = [rise_grad.item(), fall_grad.item()]
    return netArc_2_riseFallGrads

def extract_stage_grad(
    outPin_set,
    Gid_2_pinName,
    Gid_2_children,
    Gid_2_parents,
    netArc_2_riseFallGrads,
    cellArc_2_riseFallGrads
):
    outPinName_2_stageGrad = {}
    for curPin in outPin_set:
        curPinName = Gid_2_pinName[curPin]
        grad = 0
        for child in Gid_2_children[curPin]:
            childName = Gid_2_pinName[child]
            if (curPinName, childName) not in netArc_2_riseFallGrads: continue
            grad += sum(netArc_2_riseFallGrads[curPinName, childName])
        for inPin in Gid_2_parents[curPin]:
            inPinName = Gid_2_pinName[inPin]
            if (inPinName, curPinName) not in cellArc_2_riseFallGrads: continue
            grad += sum(cellArc_2_riseFallGrads[inPinName, curPinName])
        assert curPinName not in outPinName_2_stageGrad, f"repeating output pin: {curPinName}"
        outPinName_2_stageGrad[curPinName] = grad
    return outPinName_2_stageGrad

