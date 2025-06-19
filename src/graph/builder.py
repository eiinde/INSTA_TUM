"""
Graph construction functions for INSTA timing analysis

This module provides functions to build and manipulate the timing graph
used for static timing analysis. It supports both NetworkX and graph-tool
backends for different performance characteristics.
"""

import os
import time
import networkx as nx
import graph_tool.all as gt
import collections
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable

# Use absolute imports without src prefix
from ..io.serialization import save_pickle, load_pickle


def add_node(
    pinName: str,
    pinG: nx.DiGraph,
    gt_graph: gt.Graph,
    nx_2_gt: Dict[int, int],
    gt_2_nx: Dict[int, int],
    pinName_2_Gid: Dict[str, int],
    Gid_2_pinName: Dict[int, str],
    cellName_2_pinNames: Dict[str, Set[str]],
    max_Gid: int
) -> Tuple[int, int]:
    """
    Add a node to both NetworkX and graph-tool graphs

    Args:
        pinName: Name of the pin to add
        pinG: NetworkX digraph object
        gt_graph: graph-tool graph object
        nx_2_gt: Mapping from NetworkX node IDs to graph-tool vertex indices
        gt_2_nx: Mapping from graph-tool vertex indices to NetworkX node IDs
        pinName_2_Gid: Mapping from pin names to graph IDs
        Gid_2_pinName: Mapping from graph IDs to pin names
        cellName_2_pinNames: Mapping from cell names to sets of pin names
        max_Gid: Current maximum graph ID

    Returns:
        Tuple of (node_id, new_max_gid)
    """
    # Add to NetworkX graph
    pinG.add_node(max_Gid)

    # Add to graph-tool graph
    gt_node = int(gt_graph.add_vertex())

    # Create mappings
    nx_2_gt[max_Gid] = gt_node
    gt_2_nx[gt_node] = max_Gid

    # Track pin information
    pinName_2_Gid[pinName] = max_Gid
    Gid_2_pinName[max_Gid] = pinName

    # Track cell membership
    cellName = '/'.join(pinName.split('/')[:-1])
    cellName_2_pinNames[cellName].add(pinName)

    # Return node ID and incremented max_Gid
    return max_Gid, max_Gid + 1


def add_arc_to_graph(
    fromPinName: str,
    toPinName: str,
    pinG: nx.DiGraph,
    gt_graph: gt.Graph,
    nx_2_gt: Dict[int, int],
    gt_2_nx: Dict[int, int],
    pinName_2_Gid: Dict[str, int],
    Gid_2_pinName: Dict[int, str],
    cellName_2_pinNames: Dict[str, Set[str]],
    cellName_2_inPinNames: Dict[str, Set[str]],
    cellName_2_outPinNames: Dict[str, Set[str]],
    Gid_2_parents: Dict[int, Set[int]],
    Gid_2_children: Dict[int, Set[int]],
    inPin_parent_dict: Dict[int, int],
    outPin_set: Set[int],
    valid_pinNames_set: Optional[Set[str]] = None,
    noTiming_pinNames_set: Optional[Set[str]] = None,
    max_Gid: int = 0,
    mode: str = 'cell'
) -> Tuple[int, bool]:
    """
    Add a timing arc to the graph

    Args:
        fromPinName: Source pin name
        toPinName: Destination pin name
        pinG: NetworkX digraph object
        gt_graph: graph-tool graph object
        nx_2_gt: Mapping from NetworkX node IDs to graph-tool vertex indices
        pinName_2_Gid: Mapping from pin names to graph IDs
        Gid_2_pinName: Mapping from graph IDs to pin names
        cellName_2_pinNames: Mapping from cell names to sets of pin names
        cellName_2_inPinNames: Mapping from cell names to sets of input pin names
        cellName_2_outPinNames: Mapping from cell names to sets of output pin names
        Gid_2_parents: Mapping from node IDs to sets of parent node IDs
        Gid_2_children: Mapping from node IDs to sets of child node IDs
        inPin_parent_dict: Mapping from input pin IDs to parent pin IDs
        outPin_set: Set of output pin IDs
        valid_pinNames_set: Optional set of valid pin names to include
        noTiming_pinNames_set: Optional set of pin names to exclude
        max_Gid: Current maximum graph ID
        mode: 'cell' or 'net' to indicate type of arc

    Returns:
        Tuple of (new_max_gid, success)
    """
    # Skip self-loops
    if fromPinName == toPinName:
        print(f"Warning: cell arc self-loop at {fromPinName}")
        return max_Gid, False

    # Skip invalid pins
    if valid_pinNames_set and (fromPinName not in valid_pinNames_set or
                             toPinName not in valid_pinNames_set):
        print(f"Warning: cell arc {fromPinName} -> {toPinName} has no valid pin")
        return max_Gid, False

    ## Skip no-timing pins
    if noTiming_pinNames_set and (fromPinName in noTiming_pinNames_set or
                                toPinName in noTiming_pinNames_set):
        print(f"Warning: cell arc {fromPinName} -> {toPinName} has no timing pin")
        return max_Gid, False

    # Handle cell-specific operations
    if mode == 'cell':
        # Get cell name
        cellName = '/'.join(fromPinName.split('/')[:-1])

        # Ensure pins are from the same cell
        toCellName = '/'.join(toPinName.split('/')[:-1])
        if cellName != toCellName:
            print(f"Warning: Cell mismatch - {cellName} != {toCellName}")
            return max_Gid, False

        # Track input and output pins
        cellName_2_inPinNames[cellName].add(fromPinName)
        cellName_2_outPinNames[cellName].add(toPinName)

    # Add nodes if they don't exist
    fromGid = pinName_2_Gid.get(fromPinName)
    if fromGid is None:
        fromGid, max_Gid = add_node(
            fromPinName,
            pinG,
            gt_graph,
            nx_2_gt,
            gt_2_nx,
            pinName_2_Gid,
            Gid_2_pinName,
            cellName_2_pinNames,
            max_Gid
        )

    toGid = pinName_2_Gid.get(toPinName)
    if toGid is None:
        toGid, max_Gid = add_node(
            toPinName,
            pinG,
            gt_graph,
            nx_2_gt,
            gt_2_nx,
            pinName_2_Gid,
            Gid_2_pinName,
            cellName_2_pinNames,
            max_Gid
        )

    Gid_2_parents[toGid].add(fromGid)
    Gid_2_children[fromGid].add(toGid)

    # Handle net-specific operations
    if mode == 'net':
        # A sink pin should have only one driver
        if toGid in inPin_parent_dict:
            raise ValueError(f"sink pin {toPinName} has multiple drivers")

        inPin_parent_dict[toGid] = fromGid
        cellName = '/'.join(toPinName.split('/')[:-1])
        cellName_2_inPinNames[cellName].add(toPinName)
        outPin_set.add(fromGid)
    else:
        outPin_set.add(toGid)

    # Add edge to both graph representations
    pinG.add_edge(fromGid, toGid)
    gt_graph.add_edge(nx_2_gt[fromGid], nx_2_gt[toGid])

    return max_Gid, True


def build_timing_graph(
    cell_arcs: List[Tuple],
    net_arcs: List[Tuple],
    save_dir: str,
    valid_pins: Optional[Set[str]] = None,
    notiming_pins: Optional[Set[str]] = None,
    use_cache: bool = True
) -> Tuple[
    nx.DiGraph,                  # pinG
    gt.Graph,                    # gt_graph
    Dict[int, int],              # nx_2_gt
    Dict[int, int],              # gt_2_nx
    Dict[str, int],              # pinName_2_Gid
    Dict[int, str],              # Gid_2_pinName
    Dict[str, Set[str]],         # cellName_2_pinNames
    Dict[str, Set[str]],         # cellName_2_inPinNames
    Dict[str, Set[str]],         # cellName_2_outPinNames
    Dict[int, Set[int]],         # Gid_2_parents
    Dict[int, Set[int]],         # Gid_2_children
    Dict[int, int],              # inPin_parent_dict
    Set[int],                    # outPin_set
    int                          # max_Gid
]:
    """
    Build the timing graph from cell and net arcs

    Args:
        cell_arcs: List of cell arc tuples (from_pin, to_pin, sense)
        net_arcs: List of net arc tuples (from_pin, to_pin)
        save_dir: Directory to save/load cache
        valid_pins: Optional set of valid pin names to include
        notiming_pins: Optional set of pin names to exclude
        use_cache: Whether to use cached results if available

    Returns:
        Tuple of graph structures and mappings needed for timing analysis
    """
    # Check for cached results
    if use_cache:
        cache_files = [
            os.path.join(save_dir, "pinG.pkl"),
            os.path.join(save_dir, "nx_2_gt.pkl"),
            os.path.join(save_dir, "gt_2_nx.pkl"),
            os.path.join(save_dir, "pinName_2_Gid.pkl"),
            os.path.join(save_dir, "Gid_2_pinName.pkl"),
            os.path.join(save_dir, "cellName_2_pinNames.pkl"),
            os.path.join(save_dir, "cellName_2_inPinNames.pkl"),
            os.path.join(save_dir, "cellName_2_outPinNames.pkl"),
            os.path.join(save_dir, "Gid_2_parents.pkl"),
            os.path.join(save_dir, "Gid_2_children.pkl"),
            os.path.join(save_dir, "inPin_parent_dict.pkl"),
            os.path.join(save_dir, "outPin_set.pkl"),
            os.path.join(save_dir, "max_Gid.pkl"),
            os.path.join(save_dir, "gt_graph.xml.gz")
        ]

        # Check if all cache files exist
        if all(os.path.exists(f) for f in cache_files):
            # Load cached data
            pinG = load_pickle(cache_files[0], nx.DiGraph())
            nx_2_gt = load_pickle(cache_files[1], {})
            gt_2_nx = load_pickle(cache_files[2], {})
            pinName_2_Gid = load_pickle(cache_files[3], {})
            Gid_2_pinName = load_pickle(cache_files[4], {})
            cellName_2_pinNames = load_pickle(cache_files[5], collections.defaultdict(set))
            cellName_2_inPinNames = load_pickle(cache_files[6], collections.defaultdict(set))
            cellName_2_outPinNames = load_pickle(cache_files[7], collections.defaultdict(set))
            Gid_2_parents = load_pickle(cache_files[8], collections.defaultdict(set))
            Gid_2_children = load_pickle(cache_files[9], collections.defaultdict(set))
            inPin_parent_dict = load_pickle(cache_files[10], {})
            outPin_set = load_pickle(cache_files[11], set())
            max_Gid = load_pickle(cache_files[12], 0)

            # Load graph-tool graph
            gt_graph = gt.Graph()
            gt_graph.load(cache_files[13])

            return (
                pinG, gt_graph, nx_2_gt, gt_2_nx,
                pinName_2_Gid, Gid_2_pinName,
                cellName_2_pinNames, cellName_2_inPinNames, cellName_2_outPinNames,
                Gid_2_parents, Gid_2_children,
                inPin_parent_dict, outPin_set, max_Gid
            )

    # Initialize graph structures
    pinG = nx.DiGraph()
    gt_graph = gt.Graph()
    nx_2_gt = {}
    gt_2_nx = {}
    pinName_2_Gid = {}
    Gid_2_pinName = {}
    cellName_2_pinNames = collections.defaultdict(set)
    cellName_2_inPinNames = collections.defaultdict(set)
    cellName_2_outPinNames = collections.defaultdict(set)
    Gid_2_parents = collections.defaultdict(set)
    Gid_2_children = collections.defaultdict(set)
    inPin_parent_dict = {}
    outPin_set = set()
    max_Gid = 0

    start_time = time.time()

    # Add cell arcs to graph
    cell_arc_count = 0
    for from_pin, to_pin, sense in cell_arcs:
        max_Gid, success = add_arc_to_graph(
            from_pin, to_pin,
            pinG, gt_graph, nx_2_gt, gt_2_nx,
            pinName_2_Gid, Gid_2_pinName,
            cellName_2_pinNames, cellName_2_inPinNames, cellName_2_outPinNames,
            Gid_2_parents, Gid_2_children,
            inPin_parent_dict, outPin_set,
            valid_pins, notiming_pins,
            max_Gid, mode='cell'
        )
        if success:
            cell_arc_count += 1

    print(f"Added {cell_arc_count} cell arcs in {time.time() - start_time:.2f}s")
    start_time = time.time()

    # Add net arcs to graph
    net_arc_count = 0
    for from_pin, to_pin in net_arcs:
        max_Gid, success = add_arc_to_graph(
            from_pin, to_pin,
            pinG, gt_graph,
            nx_2_gt, gt_2_nx,
            pinName_2_Gid, Gid_2_pinName,
            cellName_2_pinNames, cellName_2_inPinNames, cellName_2_outPinNames,
            Gid_2_parents, Gid_2_children,
            inPin_parent_dict, outPin_set,
            valid_pins, notiming_pins,
            max_Gid, mode='net'
        )
        if success:
            net_arc_count += 1

    print(f"Added {net_arc_count} net arcs in {time.time() - start_time:.2f}s")

    # Print graph statistics
    print(f"Graph statistics:")
    print(f"  Nodes: {pinG.number_of_nodes()}")
    print(f"  Edges: {pinG.number_of_edges()}")
    print(f"  Cells: {len(cellName_2_pinNames)}")
    print(f"  Input pins: {len(inPin_parent_dict)}")
    print(f"  Output pins: {len(outPin_set)}")

    # Save graph structures to cache
    os.makedirs(save_dir, exist_ok=True)
    save_pickle(pinG, os.path.join(save_dir, "pinG.pkl"))
    save_pickle(nx_2_gt, os.path.join(save_dir, "nx_2_gt.pkl"))
    save_pickle(gt_2_nx, os.path.join(save_dir, "gt_2_nx.pkl"))
    save_pickle(pinName_2_Gid, os.path.join(save_dir, "pinName_2_Gid.pkl"))
    save_pickle(Gid_2_pinName, os.path.join(save_dir, "Gid_2_pinName.pkl"))
    save_pickle(cellName_2_pinNames, os.path.join(save_dir, "cellName_2_pinNames.pkl"))
    save_pickle(cellName_2_inPinNames, os.path.join(save_dir, "cellName_2_inPinNames.pkl"))
    save_pickle(cellName_2_outPinNames, os.path.join(save_dir, "cellName_2_outPinNames.pkl"))
    save_pickle(Gid_2_parents, os.path.join(save_dir, "Gid_2_parents.pkl"))
    save_pickle(Gid_2_children, os.path.join(save_dir, "Gid_2_children.pkl"))
    save_pickle(inPin_parent_dict, os.path.join(save_dir, "inPin_parent_dict.pkl"))
    save_pickle(outPin_set, os.path.join(save_dir, "outPin_set.pkl"))
    save_pickle(max_Gid, os.path.join(save_dir, "max_Gid.pkl"))
    gt_graph.save(os.path.join(save_dir, "gt_graph.xml.gz"))

    return (
        pinG, gt_graph, nx_2_gt, gt_2_nx,
        pinName_2_Gid, Gid_2_pinName,
        cellName_2_pinNames, cellName_2_inPinNames, cellName_2_outPinNames,
        Gid_2_parents, Gid_2_children,
        inPin_parent_dict, outPin_set, max_Gid
    )
