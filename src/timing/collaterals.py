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
# @brief initialization

import os
import torch
import collections
from typing import Dict, List, Set, Tuple, Optional, Any
import ipdb

from ..io.serialization import save_pickle, load_pickle


def precompute_collaterals(
    net_arc_2_variation: Dict,
    cell_arc_2_variation: Dict,
    sp_mean_tensor: torch.Tensor,
    sp_std_tensor: torch.Tensor,
    level_2_nodes: Dict[int, List[int]],
    Gid_2_pinName: Dict[int, str],
    inPin_parent_tensor: torch.Tensor,
    Gid_2_parents: Dict[int, Set[int]],
    device: torch.device,
    num_nodes: int,
    cell_2_libCell: Dict[str, str] = None,
    libCell_2_riseFallguardband: Dict[str, List[float]] = None,
    libCell_2_riseFallStd: Dict[str, List[float]] = None,
    net_2_pocvScaling: Dict[str, List[float]] = None,
    float_dtype: torch.dtype = torch.float32,
    save_dir: str = None,
    inPinMod: int = 1,
    use_cache: bool = True,
    save: bool = True,
    subgraph: bool = False,
    debug: bool = False,
    promote_dangling: bool = True
) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict, Optional[torch.Tensor], Optional[Dict[int, int]]]:
    """
    Precompute timing collaterals for efficient propagation

    Args:
        net_arc_2_variation: Net arc timing variations
        cell_arc_2_variation: Cell arc timing variations
        sp_mean_tensor: Startpoint mean arrival times
        sp_std_tensor: Startpoint standard deviations
        level_2_nodes: Nodes by level dictionary
        Gid_2_pinName: Mapping from graph IDs to pin names
        inPin_parent_tensor: Tensor mapping input pins to their drivers
        Gid_2_parents: Mapping from node IDs to sets of parent node IDs
        device: Computation device
        num_nodes: Maximum node ID in the graph
        cell_2_libCell: Mapping from cell names to library cells
        libCell_2_riseFallguardband: Library cell guardbands for POCV
        libCell_2_riseFallStd: Library cell standard deviations for POCV
        net_2_pocvScaling: Net scaling factors for POCV
        float_dtype: Floating point precision
        save_dir: Directory to save/load cache
        inPinMod: Input pin levelization modulo
        use_cache: Whether to use cached results if available

    Returns:
        Tuple of timing collaterals and mappings
    """
    # Check for cached results
    cache_files = [
        os.path.join(save_dir, "level_2_collaterals.pkl"),
        os.path.join(save_dir, "net_arc_2_collateral_loc.pkl"),
        os.path.join(save_dir, "cell_arc_2_collateral_loc.pkl"),
        os.path.join(save_dir, "cellArcId_2_cellName.pkl"),
        os.path.join(save_dir, "cellArcKey_2_cellArcId.pkl"),
        os.path.join(save_dir, "cellArcId_2_cellArcKey.pkl"),
        os.path.join(save_dir, "netArcId_2_inCellName.pkl"),
        os.path.join(save_dir, "netArcId_2_outCellName.pkl"),
        os.path.join(save_dir, "netArcKey_2_netArcId.pkl"),
        os.path.join(save_dir, "netArcId_2_netArcKey.pkl")
    ]

    if use_cache:
        if all(os.path.exists(f) for f in cache_files):
            level_2_collaterals = load_pickle(cache_files[0], {})
            net_arc_2_collateral_loc = load_pickle(cache_files[1], {})
            cell_arc_2_collateral_loc = load_pickle(cache_files[2], {})
            cellArcId_2_cellName = load_pickle(cache_files[3], {})
            cellArcKey_2_cellArcId = load_pickle(cache_files[4], {})
            cellArcId_2_cellArcKey = load_pickle(cache_files[5], {})
            netArcId_2_inCellName = load_pickle(cache_files[6], {})
            netArcId_2_outCellName = load_pickle(cache_files[7], {})
            netArcKey_2_netArcId = load_pickle(cache_files[8], {})
            netArcId_2_netArcKey = load_pickle(cache_files[9], {})

            return (
                    level_2_collaterals,
                    net_arc_2_collateral_loc, cell_arc_2_collateral_loc,
                    cellArcId_2_cellName, cellArcKey_2_cellArcId, cellArcId_2_cellArcKey,
                    netArcId_2_inCellName, netArcId_2_outCellName, netArcKey_2_netArcId, netArcId_2_netArcKey
           )

    # Setup device for computations

    device = torch.device('cpu')

    is_pocv = True if len(list(net_arc_2_variation.values())[0]) == 4 else False
    if debug:
        print(f"using pocv: {is_pocv}")

    Gid_2_arrival = torch.tensor([float('-inf')] * num_nodes, dtype=float_dtype)
    Gid_2_mean = torch.tensor([float('-inf')] * num_nodes, dtype=float_dtype)
    Gid_2_std = torch.tensor([float('-inf')] * num_nodes, dtype=float_dtype)

    seen_arcs = set()

    net_arc_2_collateral_loc = {}
    cell_arc_2_collateral_loc = {}
    level_2_collaterals = {}

    '''for differentiability'''
    cellArcId = 0
    cellArcId_2_cellName = {}
    cellArcKey_2_cellArcId = {}
    cellArcId_2_cellArcKey = {}

    netArcId = 0
    netArcId_2_inCellName = {}
    netArcId_2_outCellName = {}
    netArcKey_2_netArcId = {}
    netArcId_2_netArcKey = {}

    design_rise_gb, design_fall_gb = libCell_2_riseFallguardband['design']
    net_rise_gb, net_fall_gb = net_2_pocvScaling['net']
    design_rise_std_coef, design_fall_std_coef = libCell_2_riseFallStd['design']

    # Build local index mapping for subgraph once (if subgraph)
    if subgraph:
        all_nodes_set = set()
        for v in level_2_nodes.values():
            all_nodes_set.update(v)
        cone_gid_list = torch.tensor(sorted(all_nodes_set), dtype=torch.int32)
        gid2local_map = {int(g): idx for idx, g in enumerate(cone_gid_list.tolist())}

        # ------------------------------------------------------------------
        # Promote dangling input pins (no valid driver) to level-1 startpoints
        # ------------------------------------------------------------------
        # These pins are discovered during fan-in traversal but their parent
        # driver output is missing (e.g. primary inputs).  They need to be
        # treated as startpoints so that their timing is properly initialized
        # in later propagation stages.
        extra_lvl1_nodes: List[int] = []
        levels_to_prune: List[int] = []
        if promote_dangling:
            for lvl, nodes in list(level_2_nodes.items()):
                if lvl == 1:
                    continue  # already in startpoint level
                if lvl % 2 != inPinMod:
                    continue  # only inspect *input-pin* levels

                retained_nodes: List[int] = []
                for n in nodes:
                    parent_gid = int(inPin_parent_tensor[n].item())
                    if parent_gid < 0 or parent_gid not in gid2local_map:
                        extra_lvl1_nodes.append(n)
                    else:
                        retained_nodes.append(n)
                if not retained_nodes:
                    levels_to_prune.append(lvl)
                else:
                    level_2_nodes[lvl] = retained_nodes  # update in place

        if promote_dangling:
            # Remove empty levels generated after pruning
            for lvl in levels_to_prune:
                level_2_nodes.pop(lvl, None)

        if promote_dangling and extra_lvl1_nodes:
            base_lvl1 = level_2_nodes.get(1, [])
            base_lvl1.extend(extra_lvl1_nodes)
            seen_set = set()
            dedup_lvl1 = []
            for gid in base_lvl1:
                if gid not in seen_set:
                    dedup_lvl1.append(gid)
                    seen_set.add(gid)
            level_2_nodes[1] = dedup_lvl1
            Gid_2_arrival[list(extra_lvl1_nodes)] = 1
            # ---------------- Debug / Sanity -----------------------------
            if debug:
                print(f"[promote_dangling] promoted {len(extra_lvl1_nodes)} input pins to level-1 startpoints")
            # Assert every promoted node truly lacks a valid driver within the subgraph
            for gid in extra_lvl1_nodes:
                parent_gid_chk = int(inPin_parent_tensor[gid].item())
                assert parent_gid_chk < 0 or parent_gid_chk not in gid2local_map, (
                    f"[promote_dangling] Pin {gid} has parent {parent_gid_chk} that exists inside the subgraph – should not be promoted")
    else:
        cone_gid_list = None
        gid2local_map = {}

    for level in level_2_nodes:
        if level == 1: # CLK pins
            cur_nodes = level_2_nodes[level]
            if not subgraph:
                means = sp_mean_tensor[cur_nodes]
                valid_mean_mask = ~torch.isinf(means)
                stds = sp_std_tensor[cur_nodes]
                valid_std_mask = ~torch.isinf(stds)
                valid_mask = valid_mean_mask & valid_std_mask
                indices = torch.where(valid_mask)[0].tolist()
                cur_nodes = [cur_nodes[idx] for idx in indices]
                Gid_2_mean[cur_nodes] = sp_mean_tensor[cur_nodes]
                Gid_2_std[cur_nodes] = sp_std_tensor[cur_nodes]
            else:
                Gid_2_mean[cur_nodes] = 0.0
                Gid_2_std[cur_nodes] = 0.0
            Gid_2_arrival[cur_nodes] = 1
            level_2_collaterals[level] = torch.tensor(cur_nodes, dtype=torch.int64).to(device)
            if debug:
                print("at level: {}, # valid nodes: {}".format(level, len(cur_nodes)))
        elif level % 2 == inPinMod: # input pins
            cur_nodes = level_2_nodes[level]
            if debug:
                print('# raw cur_nodes: {}'.format(len(cur_nodes)))
            parents = inPin_parent_tensor[cur_nodes]
            p_arrivals = Gid_2_arrival[parents]
            p_valid_mask = ~torch.isinf(p_arrivals)
            if debug:
                print('# valid nodes: {}'.format(p_valid_mask.sum()))
            indices = torch.where(p_valid_mask)[0].tolist()
            cur_nodes = [cur_nodes[idx] for idx in indices]
            parents = inPin_parent_tensor[cur_nodes]

            rise_means = []
            rise_stds = []
            rise_sigmas = []
            fall_means = []
            fall_stds = []
            fall_sigmas = []
            net_arc_ids = []


            for cur_node in cur_nodes:
                fromPinName = Gid_2_pinName[inPin_parent_tensor[cur_node].item()]
                toPinName = Gid_2_pinName[cur_node]
                assert (fromPinName, toPinName) not in seen_arcs
                seen_arcs.add((fromPinName, toPinName))
                if not is_pocv:
                    rise_mean, fall_mean = net_arc_2_variation[fromPinName, toPinName]
                    rise_stds.append(0.0)
                    fall_stds.append(0.0)
                    rise_sigmas.append(0.0)
                    fall_sigmas.append(0.0)
                else:
                    rise_mean, rise_std, fall_mean, fall_std = net_arc_2_variation[fromPinName, toPinName]
                    rise_stds.append(rise_std * net_rise_gb)
                    fall_stds.append(fall_std * net_fall_gb)
                    rise_sigmas.append(3.0)
                    fall_sigmas.append(3.0)

                net_arc_key = (fromPinName, toPinName)
                try:
                    assert net_arc_key not in net_arc_2_collateral_loc, f'duplicated net arc key: {net_arc_key}'
                except:
                    ipdb.set_trace()
                net_arc_2_collateral_loc[net_arc_key] = (level, len(rise_means))
                inCellName = '/'.join(fromPinName.split('/')[:-1])
                outCellName = '/'.join(toPinName.split('/')[:-1])

                netArcKey_2_netArcId[net_arc_key] = netArcId
                netArcId_2_netArcKey[netArcId] = net_arc_key
                netArcId_2_inCellName[netArcId] = inCellName
                netArcId_2_outCellName[netArcId] = outCellName
                net_arc_ids.append(netArcId)
                netArcId += 1
                rise_means.append(rise_mean * net_rise_gb)
                fall_means.append(fall_mean * net_fall_gb)

            level_2_collaterals[level] = [
                                             torch.tensor(cur_nodes, dtype=torch.int64).to(device),
                                             parents,
                                             torch.tensor(rise_means, dtype=float_dtype).to(device),
                                             torch.tensor(rise_stds, dtype=float_dtype).to(device),
                                             torch.tensor(rise_sigmas, dtype=float_dtype).to(device),
                                             torch.tensor(fall_means, dtype=float_dtype).to(device),
                                             torch.tensor(fall_stds, dtype=float_dtype).to(device),
                                             torch.tensor(fall_sigmas, dtype=float_dtype).to(device),
                                             net_arc_ids
                                         ]
            if subgraph:
                # append local index tensors for faster propagation
                cur_local = torch.tensor([gid2local_map[int(g)] for g in cur_nodes], dtype=torch.int32)
                parent_local = torch.tensor([gid2local_map[int(g)] for g in parents.tolist()], dtype=torch.int32)
                level_2_collaterals[level].extend([cur_local, parent_local])
            if not subgraph and debug:
                print("at level: {}, # valid nodes: {}".format(level, len(cur_nodes)))
            Gid_2_arrival[cur_nodes] = 1

        else: # output pins

            p_indices = []
            c_unique_indices = []
            node_start_end_idx = [] # 1D arr stores start/end idx of tables
            duplicated_nodes_in_level = [] # because an output can have multuple parents
            rise_means = []
            rise_stds = []
            rise_sigmas = []
            fall_means = []
            fall_stds = []
            fall_sigmas = []
            arc_senses = []
            cellArc_ids = []

            cur_nodes = level_2_nodes[level]
            seen_nodes = {}
            for node in cur_nodes: # out pins
                toPinName = Gid_2_pinName[node]
                cellName = '/'.join(toPinName.split('/')[:-1])

                if cell_2_libCell:
                    libCell = cell_2_libCell[cellName].split('/')[-1]
                    if libCell in libCell_2_riseFallguardband:
                        rise_gb, fall_gb = libCell_2_riseFallguardband[libCell]
                    else:
                        rise_gb, fall_gb = design_rise_gb, design_fall_gb
                    if libCell in libCell_2_riseFallStd:
                        rise_std_coef, fall_std_coef = libCell_2_riseFallStd[libCell]
                    else:
                        rise_std_coef, fall_std_coef = design_rise_std_coef, design_fall_std_coef
                else:
                    rise_gb, fall_gb = design_rise_gb, design_fall_gb
                    rise_std_coef, fall_std_coef = design_rise_std_coef, design_fall_std_coef

                for parent in Gid_2_parents[node]:
                    if Gid_2_arrival[parent] == float('-inf'):
                        if subgraph:
                            if debug:
                                p_name = Gid_2_pinName.get(parent, str(parent))
                                c_name = Gid_2_pinName.get(node, str(node))
                                in_cone = parent in gid2local_map
                                print(
                                    f"[subgraph][ASSERT] parent not arrived: parent {parent} ({p_name}) -> child {node} ({c_name}) "
                                    f"at level {level} | parent_in_cone={in_cone}")
                            import ipdb; ipdb.set_trace()
                            assert False, f"subgraph mode: parent not arrived: {parent}"
                        continue
                    if node not in seen_nodes: # first time encouter this node
                        node_start_end_idx.append(len(duplicated_nodes_in_level))
                        seen_nodes[node] = 1
                        c_unique_indices.append(node)
                    fromPinName = Gid_2_pinName[parent]

                    try:
                        assert cellName == '/'.join(fromPinName.split('/')[:-1]), \
                            f'fromPin: {fromPinName}, toPin: {toPinName} cell mismatch...'
                    except:
                        ipdb.set_trace()

                    is_cell_arc_valid_at_least_once = False
                    for sense in ['positive_unate', 'negative_unate', 'rising_edge', 'falling_edge']:
                        cell_arc_key = (fromPinName, toPinName, sense)
                        if cell_arc_key in cell_arc_2_variation:
                            is_cell_arc_valid_at_least_once = True
                            p_indices.append(parent)
                            duplicated_nodes_in_level.append(node)
                            if not is_pocv:
                                rise_mean, fall_mean = cell_arc_2_variation[cell_arc_key]
                                rise_sigmas.append(0.0)
                                fall_sigmas.append(0.0)
                                rise_stds.append(0.0)
                                fall_stds.append(0.0)
                            else:
                                rise_mean, rise_std, fall_mean, fall_std = cell_arc_2_variation[cell_arc_key]
                                rise_std *= rise_std_coef * rise_gb
                                fall_std *= fall_std_coef * fall_gb
                                rise_stds.append(rise_std)
                                fall_stds.append(fall_std)
                                rise_sigmas.append(3.0)
                                fall_sigmas.append(3.0)
                            rise_mean *= rise_gb
                            fall_mean *= fall_gb
                            if sense[:3] == 'neg':
                                arc_senses.append(1)
                            else:
                                arc_senses.append(0)

                            try:
                                assert cell_arc_key not in cell_arc_2_collateral_loc, f'duplicated cell arc key: {cell_arc_key}'
                            except:
                                ipdb.set_trace()

                            cell_arc_2_collateral_loc[cell_arc_key] = (level, len(rise_means))

                            '''for differentiability'''
                            cellArcKey_2_cellArcId[cell_arc_key] = cellArcId
                            cellArcId_2_cellArcKey[cellArcId] = cell_arc_key
                            cellArcId_2_cellName[cellArcId] = cellName
                            cellArc_ids.append(cellArcId)
                            cellArcId += 1

                            rise_means.append(rise_mean)
                            fall_means.append(fall_mean)


                    assert is_cell_arc_valid_at_least_once, "cell arc: {} is present but not valid".format(
                            cell_arc_key)

            assert duplicated_nodes_in_level, "at level: {}, no valid nodes".format(level)

            # this should be the common case unless the very last node is not valid
            if len(duplicated_nodes_in_level) != node_start_end_idx[-1]:
                node_start_end_idx.append(len(duplicated_nodes_in_level))

            assert len(duplicated_nodes_in_level) == node_start_end_idx[-1]
            assert len(c_unique_indices) == len(node_start_end_idx) - 1
            assert len(rise_means) == len(rise_stds)
            assert len(rise_means) == len(fall_means)
            assert len(rise_means) == len(fall_stds)
            assert len(rise_means) == len(rise_sigmas)
            assert len(rise_means) == len(fall_sigmas)
            assert len(rise_means) == len(arc_senses)

            # generate p_mapping as p_indices may duplicate and cause memory issue
            p_idx_unique = list(set(p_indices))
            p_mapping = [-1] * (max(p_idx_unique) + 1)
            for idx, ele in enumerate(p_idx_unique):
                p_mapping[ele] = idx

            level_2_collaterals[level] = [
                                             duplicated_nodes_in_level,
                                             torch.tensor(rise_means, dtype=float_dtype).to(device),
                                             torch.tensor(rise_stds, dtype=float_dtype).to(device),
                                             torch.tensor(rise_sigmas, dtype=float_dtype).to(device),
                                             torch.tensor(fall_means, dtype=float_dtype).to(device),
                                             torch.tensor(fall_stds, dtype=float_dtype).to(device),
                                             torch.tensor(fall_sigmas, dtype=float_dtype).to(device),
                                             torch.tensor(arc_senses, dtype=torch.int32).to(device),
                                             torch.tensor(p_indices, dtype=torch.int32).to(device),
                                             torch.tensor(node_start_end_idx, dtype=torch.int32).to(device),
                                             c_unique_indices,
                                             torch.tensor(p_idx_unique, dtype=torch.int32).to(device),
                                             torch.tensor(p_mapping, dtype=torch.int32).to(device),
                                             torch.tensor(c_unique_indices, dtype=torch.int64).to(device),
                                             torch.tensor(cellArc_ids, dtype=torch.int32).to(device)
                                         ]
            if subgraph:
                p_local_unique = torch.tensor([gid2local_map[int(g)] for g in p_idx_unique], dtype=torch.int32)
                c_unique_local = torch.tensor([gid2local_map[int(g)] for g in c_unique_indices], dtype=torch.int32)
                level_2_collaterals[level].extend([p_local_unique, c_unique_local])
            Gid_2_arrival[c_unique_indices] = 1
            if not subgraph and debug:
                print("at level: {}, # valid nodes: {}".format(level, len(c_unique_indices)))

    # === Ensure full node coverage for sub-graph mode =====================
    if subgraph:
        # Collect nodes already present in collaterals ----------------------
        present_nodes: set = set()
        for lvl, obj in level_2_collaterals.items():
            if lvl == 1:
                present_nodes.update(obj.tolist())
            elif lvl % 2 == inPinMod:
                # input pin level – first entry is tensor of current nodes
                present_nodes.update(obj[0].tolist())
            else:
                # output pin level – child unique indices list is at position 10
                present_nodes.update(obj[10])

        # Any nodes missing are appended to level-1 and treated as CLK pins.
        missing_nodes = set(cone_gid_list.tolist()) - present_nodes
        if missing_nodes:
            if debug:
                print(f"[subgraph] attaching {len(missing_nodes)} dangling nodes to level-1 startpoints")
            # Update level-1 tensor (ensure unique + deterministic order)
            lvl1_tensor: torch.Tensor = level_2_collaterals[1]
            merged_lvl1 = torch.cat([
                lvl1_tensor,
                torch.tensor(sorted(missing_nodes), dtype=torch.int64, device=device)
            ])
            # Remove duplicates (order not important for level-1)
            level_2_collaterals[1] = torch.unique(merged_lvl1)
            # Mark arrivals so that downstream levels can reference them safely
            Gid_2_arrival[list(missing_nodes)] = 1

        # Final coverage assertion – no pin from the cone should be missing
        final_present_nodes: set = set()
        for lvl_i, obj in level_2_collaterals.items():
            if lvl_i == 1:
                final_present_nodes.update(obj.tolist())
            elif lvl_i % 2 == inPinMod:
                final_present_nodes.update(obj[0].tolist())
            else:
                final_present_nodes.update(obj[10])

        still_missing = set(cone_gid_list.tolist()) - final_present_nodes
        assert not still_missing, (
            f"[subgraph] {len(still_missing)} nodes are still missing after coverage patch: {sorted(list(still_missing))[:10]} ...")
        if debug:
            print(f"[subgraph] node coverage complete – total nodes covered: {len(final_present_nodes)}")

    if save:
        print('saving precomputed collaterals...')
        save_pickle(level_2_collaterals, cache_files[0])
        save_pickle(net_arc_2_collateral_loc, cache_files[1])
        save_pickle(cell_arc_2_collateral_loc, cache_files[2])
        save_pickle(cellArcId_2_cellName, cache_files[3])
        save_pickle(cellArcKey_2_cellArcId, cache_files[4])
        save_pickle(cellArcId_2_cellArcKey, cache_files[5])
        save_pickle(netArcId_2_inCellName, cache_files[6])
        save_pickle(netArcId_2_outCellName, cache_files[7])
        save_pickle(netArcKey_2_netArcId, cache_files[8])
        save_pickle(netArcId_2_netArcKey, cache_files[9])

    if subgraph:
        return (
                level_2_collaterals,
                net_arc_2_collateral_loc, cell_arc_2_collateral_loc,
                cellArcId_2_cellName, cellArcKey_2_cellArcId, cellArcId_2_cellArcKey,
                netArcId_2_inCellName, netArcId_2_outCellName, netArcKey_2_netArcId, netArcId_2_netArcKey,
                cone_gid_list, gid2local_map
               )
    else:
        return (
                level_2_collaterals,
                net_arc_2_collateral_loc, cell_arc_2_collateral_loc,
                cellArcId_2_cellName, cellArcKey_2_cellArcId, cellArcId_2_cellArcKey,
                netArcId_2_inCellName, netArcId_2_outCellName, netArcKey_2_netArcId, netArcId_2_netArcKey
               )


def move_collaterals_to_device(level_2_collaterals: Dict, device: torch.device) -> Dict:
    """
    Move precomputed collaterals to a specified device

    Args:
        level_2_collaterals: Dictionary of collaterals by level
        device: Target device (CPU or CUDA)

    Returns:
        Dictionary of collaterals moved to device
    """
    new_level_2_collaterals = {}

    for level, objects in level_2_collaterals.items():
        if level == 1:
            # Level 1 collaterals are just a single tensor
            new_level_2_collaterals[level] = objects.to(device)
        else:
            # Other levels have a list of objects
            new_objects = []
            for obj in objects:
                if isinstance(obj, torch.Tensor):
                    new_objects.append(obj.to(device))
                elif isinstance(obj, list) and isinstance(obj[0], int):
                    # Convert lists of integers to tensors where appropriate
                    new_objects.append(torch.tensor(obj, device=device))
                else:
                    new_objects.append(obj)
            new_level_2_collaterals[level] = new_objects

    return new_level_2_collaterals


