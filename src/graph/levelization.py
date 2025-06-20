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
# @file core/levelization.py
# @brief graph levelization algorithms

import os
import collections
import torch
from typing import Dict, List, Set, Tuple, Optional, Union, Any

import graph_tool.all as gt

from ..io.serialization import save_pickle, load_pickle


def forward_levelization(
    gt_graph: gt.Graph,
    gt_2_nx: Dict[int, int],
    source_nodes: Set[int],
    dest_nodes: Set[int],
    id_2_parents: Dict[int, Set[int]],
    infer_sources: bool = False
) -> Tuple[Dict[int, List[int]], Dict[int, int], List[int], List[int]]:
    """
    Perform forward levelization using graph-tool for better performance

    Args:
        gt_graph: graph-tool graph
        gt_2_nx: Mapping from graph-tool vertex indices to NetworkX node IDs
        source_nodes: Set of starting nodes (e.g., clock pins)
        dest_nodes: Set of ending nodes (e.g., register data pins)
        id_2_parents: Mapping from node IDs to their parent node IDs
        infer_sources: Whether to infer source nodes from graph topology

    Returns:
        Tuple of (level_to_nodes, node_to_level, source_nodes, dest_nodes)
        where source_nodes and dest_nodes are only meaningful if infer_sources is True
    """
    # Initialize level mapping
    node_2_level = {}

    # Handle source nodes
    if infer_sources:
        # Find nodes with no incoming edges
        actual_source_nodes = [v for v in gt_graph.vertices() if v.in_degree() == 0]
        for s in actual_source_nodes:
            node_2_level[gt_2_nx[int(s)]] = 1

        print(f"[forward levelization] Found {len(actual_source_nodes)} source nodes from graph topology")
    else:
        # Use provided source nodes
        for s in source_nodes:
            node_2_level[s] = 1

    # Perform topological sort
    sorted_nodes = gt.topological_sort(gt_graph)

    # Assign levels based on longest path from source
    for gt_node in sorted_nodes:
        nx_node = gt_2_nx[int(gt_node)]

        # Skip if no parents (should only happen for source nodes)
        if nx_node not in id_2_parents or not id_2_parents[nx_node]:
            continue

        # Find maximum level of parents
        max_parent_level = 0
        for parent in id_2_parents[nx_node]:
            if parent in node_2_level:
                max_parent_level = max(max_parent_level, node_2_level[parent])

        # Set node level as 1 + max parent level
        if max_parent_level > 0:
            node_2_level[nx_node] = max_parent_level + 1

    # Group nodes by level
    level_2_nodes = collections.defaultdict(list)
    for node, level in node_2_level.items():
        level_2_nodes[level].append(node)

    # Sort levels
    sorted_level_2_nodes = {}
    for level in sorted(level_2_nodes.keys()):
        sorted_level_2_nodes[level] = level_2_nodes[level]

    # Print statistics for each level
    for level, nodes in sorted_level_2_nodes.items():
        endpoint_count = sum(1 for n in nodes if n in dest_nodes)
        print(f"[forward levelization] Level {level}: {len(nodes)} nodes, {endpoint_count} endpoints")

    # Return additional data if inferring sources
    if infer_sources:
        # Find nodes with no outgoing edges for destinations
        if not dest_nodes:
            actual_dest_nodes = [gt_2_nx[int(v)] for v in gt_graph.vertices() if v.out_degree() == 0]
            print(f"[forward levelization] Found {len(actual_dest_nodes)} destination nodes from graph topology")
        else:
            actual_dest_nodes = list(dest_nodes)

        return (
            sorted_level_2_nodes,
            node_2_level,
            [gt_2_nx[int(s)] for s in actual_source_nodes],
            actual_dest_nodes
        )

    return sorted_level_2_nodes, node_2_level, [], []


def backward_levelization(
    gt_graph: gt.Graph,
    gt_2_nx: Dict[int, int],
    source_nodes: Set[int],
    dest_nodes: Set[int],
    id_2_children: Dict[int, Set[int]]
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Perform backward levelization using graph-tool for better performance

    Args:
        gt_graph: graph-tool graph
        gt_2_nx: Mapping from graph-tool vertex indices to NetworkX node IDs
        source_nodes: Set of starting nodes (e.g., clock pins)
        dest_nodes: Set of ending nodes (e.g., register data pins)
        id_2_children: Mapping from node IDs to their child node IDs

    Returns:
        Tuple of (level_to_nodes, node_to_level)
    """
    # Create reversed view of graph for backward traversal
    reversed_gt_graph = gt.GraphView(gt_graph, directed=True)
    reversed_gt_graph.set_reversed(True)

    # Perform topological sort on reversed graph
    sorted_nodes = gt.topological_sort(reversed_gt_graph)

    # Initialize level mapping with destination nodes at level 1
    node_2_level = {}
    for d in dest_nodes:
        node_2_level[d] = 1

    # Assign levels based on longest path from destinations
    for gt_node in sorted_nodes:
        nx_node = gt_2_nx[int(gt_node)]

        # Skip if no children (should only happen for destination nodes)
        if nx_node not in id_2_children or not id_2_children[nx_node]:
            continue

        # Find maximum level of children
        max_child_level = 0
        for child in id_2_children[nx_node]:
            if child in node_2_level:
                max_child_level = max(max_child_level, node_2_level[child])

        # Set node level as 1 + max child level
        if max_child_level > 0:
            node_2_level[nx_node] = max_child_level + 1

    # Group nodes by level
    level_2_nodes = collections.defaultdict(list)
    for node, level in node_2_level.items():
        level_2_nodes[level].append(node)

    # Sort levels
    sorted_level_2_nodes = {}
    for level in sorted(level_2_nodes.keys()):
        sorted_level_2_nodes[level] = level_2_nodes[level]

    # Print statistics for each level
    for level, nodes in sorted_level_2_nodes.items():
        startpoint_count = sum(1 for n in nodes if n in source_nodes)
        print(f"[backward levelization] Level {level}: {len(nodes)} nodes, {startpoint_count} startpoints")

    return sorted_level_2_nodes, node_2_level


def levelize_graph(
    gt_graph: gt.Graph,
    gt_2_nx: Dict[int, int],
    source_nodes: Set[int],
    dest_nodes: Set[int],
    Gid_2_parents: Dict[int, Set[int]],
    Gid_2_children: Dict[int, Set[int]],
    inPin_parent_dict: Dict[int, int],
    save_dir: str,
    infer_sources: bool = False,
    do_backward: bool = True,
    use_cache: bool = True
) -> Tuple[
    Dict[int, List[int]],         # level_2_nodes
    Dict[int, int],               # node_2_level
    torch.Tensor,                 # inPin_parent_tensor
    Optional[Dict[int, List[int]]],  # level_2_nodes_bw
    Optional[Dict[int, int]]         # node_2_level_bw
]:
    """
    Perform graph levelization for timing analysis

    Args:
        gt_graph: graph-tool graph
        gt_2_nx: Mapping from graph-tool vertex indices to NetworkX node IDs
        source_nodes: Set of starting nodes (e.g., clock pins)
        dest_nodes: Set of ending nodes (e.g., register data pins)
        Gid_2_parents: Mapping from node IDs to their parent node IDs
        Gid_2_children: Mapping from node IDs to their child node IDs
        inPin_parent_dict: Mapping from input pin IDs to parent pin IDs
        save_dir: Directory to save/load cache
        infer_sources: Whether to infer source nodes from graph topology
        do_backward: Whether to perform backward levelization
        use_cache: Whether to use cached results if available

    Returns:
        Tuple of (level_2_nodes, node_2_level, inPin_parent_tensor, level_2_nodes_bw, node_2_level_bw)
    """
    # Check for source and destination nodes
    if not infer_sources and (not source_nodes or not dest_nodes):
        print('[levelization] error: missing source or destination nodes')
        return None, None, None, None, None

    # Check for cached results
    if use_cache:
        cache_path = os.path.join(save_dir, 'level_2_nodes.pkl')
        if os.path.exists(cache_path):
            # Load cached levelization data
            level_2_nodes = load_pickle(cache_path)
            node_2_level = load_pickle(os.path.join(save_dir, 'node_2_level.pkl'))
            inPin_parent_tensor = torch.load(os.path.join(save_dir, 'inPin_parent_tensor.pt'))

            # Load backward levelization if requested
            level_2_nodes_bw = None
            node_2_level_bw = None
            if do_backward:
                bw_cache = os.path.join(save_dir, 'level_2_nodes_bw.pkl')
                if os.path.exists(bw_cache):
                    level_2_nodes_bw = load_pickle(bw_cache)
                    node_2_level_bw = load_pickle(os.path.join(save_dir, 'node_2_level_bw.pkl'))

            # Check and load derived nodes if using inferrence
            if infer_sources:
                src_cache = os.path.join(save_dir, 'source_nodes.pkl')
                dst_cache = os.path.join(save_dir, 'dest_nodes.pkl')
                if os.path.exists(src_cache) and os.path.exists(dst_cache):
                    # These would normally be returned by forward_levelization with infer_sources=True
                    pass

            return level_2_nodes, node_2_level, inPin_parent_tensor, level_2_nodes_bw, node_2_level_bw

    # Start the levelization process
    import time
    start_time = time.time()

    # Perform forward levelization
    if infer_sources:
        # Infer source and destination nodes from graph topology
        level_2_nodes, node_2_level, src_nodes, dst_nodes = forward_levelization(
            gt_graph, gt_2_nx, source_nodes, dest_nodes, Gid_2_parents, True
        )

        # Update source and destination nodes if they were inferred
        if not source_nodes:
            source_nodes = set(src_nodes)
            save_pickle(source_nodes, os.path.join(save_dir, 'source_nodes.pkl'))

        if not dest_nodes:
            dest_nodes = set(dst_nodes)
            save_pickle(dest_nodes, os.path.join(save_dir, 'dest_nodes.pkl'))
    else:
        # Use provided source and destination nodes
        level_2_nodes, node_2_level, _, _ = forward_levelization(
            gt_graph, gt_2_nx, source_nodes, dest_nodes, Gid_2_parents, False
        )

    # Perform backward levelization if requested
    level_2_nodes_bw = None
    node_2_level_bw = None
    if do_backward:
        level_2_nodes_bw, node_2_level_bw = backward_levelization(
            gt_graph, gt_2_nx, source_nodes, dest_nodes, Gid_2_children
        )

    # Create input pin parent tensor
    max_id = max(Gid_2_parents.keys()) + 1 if Gid_2_parents else 0
    inPin_parent_tensor = torch.full((max_id,), -1, dtype=torch.int32)
    for toPin, fromPin in inPin_parent_dict.items():
        inPin_parent_tensor[toPin] = fromPin

    print(f'[levelization] completed in {time.time() - start_time:.2f}s')

    # Save results
    save_pickle(level_2_nodes, os.path.join(save_dir, 'level_2_nodes.pkl'))
    save_pickle(node_2_level, os.path.join(save_dir, 'node_2_level.pkl'))
    torch.save(inPin_parent_tensor, os.path.join(save_dir, 'inPin_parent_tensor.pt'))

    if do_backward:
        save_pickle(level_2_nodes_bw, os.path.join(save_dir, 'level_2_nodes_bw.pkl'))
        save_pickle(node_2_level_bw, os.path.join(save_dir, 'node_2_level_bw.pkl'))

    return level_2_nodes, node_2_level, inPin_parent_tensor, level_2_nodes_bw, node_2_level_bw
