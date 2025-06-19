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
# @file core/insta.py
# @brief main INSTA class

import os
import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)
import time
import torch
import collections
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import copy
import time
import numpy as np
from ..io.parsers import (
    read_cell_libcell_file, read_no_timing_pin_file, read_valid_pin_file,
    read_cell_arc_file, read_net_arc_file, read_sp_file, read_ep_file,
    read_pocvm_file
)
from ..io.serialization import save_pickle, load_pickle, save_torch_tensor, load_torch_tensor
from ..graph.builder import build_timing_graph
from ..graph.levelization import levelize_graph
from ..timing.propagation import clear_timing_cache, propagate_arrival_times, save_arrival_tensors
from ..timing.collaterals import precompute_collaterals, move_collaterals_to_device
from ..timing.pocv   import initialize_timing_tensors, apply_cppr_correction
from ..timing.pocv   import extract_cellArc_grad, extract_netArc_grad, extract_stage_grad
from ..visualization.plotting import plot_ep_correlation, write_analysis_csv
import ipdb

from .constants import (
    DEFAULT_DEVICE, DEFAULT_FLOAT_DTYPE, DEFAULT_TOPK, DEFAULT_SCALING,
    DEFAULT_INPUT_FOLDER, DEFAULT_OUTPUT_FOLDER, NESTED_LIB_DICT_PATH,
    NESTED_PIN_DICT_PATH, DEFAULT_SIGMA, DEFAULT_TEMPERATURE
)

class INSTA:
    """Main infrastructure for Neural Static Timing Analysis"""

    def __init__(self):
        """Initialize the INSTA analyzer with default settings"""
        # Core configuration
        self.device = DEFAULT_DEVICE
        self.float_dtype = DEFAULT_FLOAT_DTYPE
        self.scale = DEFAULT_SCALING
        self.topK = DEFAULT_TOPK

        # Paths and design info
        self.prefix = './'
        self.save_dir = os.path.join(self.prefix, DEFAULT_OUTPUT_FOLDER)
        self.design_name = ""
        self.input_folderName = DEFAULT_INPUT_FOLDER

        self.pinG = None
        self.gt_graph = None
        self.nx_2_gt = {}
        self.gt_2_nx = {}
        self.max_Gid = 0

        self.pinName_2_Gid = {}
        self.Gid_2_pinName = {}
        self.cellName_2_pinNames = collections.defaultdict(set)
        self.cellName_2_inPinNames = collections.defaultdict(set)
        self.cellName_2_outPinNames = collections.defaultdict(set)
        self.cellName_2_orgLibCell = {}
        self.cellName_2_funcId = {}
        self.funcId_2_libCellNames = collections.defaultdict(set)

        self.Gid_2_parents = collections.defaultdict(set)
        self.Gid_2_children = collections.defaultdict(set)
        self.inPin_parent_dict = {}
        self.outPin_set = set()

        self.cell_arc_2_variation = {}
        self.net_arc_2_variation = {}
        self.is_pocv = False

        self.valid_pinNames_set = set()
        self.noTiming_pinNames_set = set()
        self.filter = False

        self.source_nodes = set()
        self.dest_nodes = set()
        self.spName_2_attributes = {}
        self.epName_riseFall_2_attributes = {}
        self.epName_riseFall_2_spName = {}
        self.epName_riseFall_2_ckPinName = {}
        self.epName_riseFall_2_launch_clock_latency = {}

        self.level_2_nodes = {}
        self.node_2_level = {}
        self.level_2_nodes_bw = {}
        self.node_2_level_bw = {}
        self.inPin_parent_tensor = None

        self.level_2_collaterals = None
        self.cell_arc_2_collateral_loc = {}
        self.net_arc_2_collateral_loc = {}
        self.cellArcId_2_cellName = {}
        self.cellArcKey_2_cellArcId = {}
        self.cellArcId_2_cellArcKey = {}
        self.netArcId_2_inCellName = {}
        self.netArcId_2_outCellName = {}
        self.netArcKey_2_netArcId = {}
        self.netArcId_2_netArcKey = {}
        self.max_cellArcId = 0
        self.max_netArcId = 0

        self.net_2_pocvScaling = {}
        self.libCell_2_riseFallguardband = {}
        self.libCell_2_riseFallStd = {}

        self.timing_tensors = {}



    def do_set_insta_path(self, path: str, design_name: str, input_folderName: Optional[str] = None):
        """
        Set the base path and design information

        Args:
            path: Base directory path for the design
            design_name: Name of the design
            input_folderName: Optional custom folder name for inputs
        """
        self.prefix = path
        self.design_name = design_name
        self.save_dir = os.path.join(self.prefix, DEFAULT_OUTPUT_FOLDER)
        os.makedirs(self.save_dir, exist_ok=True)

        if input_folderName:
            self.input_folderName = input_folderName

    def set_device(self, num: int):
        torch.cuda.set_device(num)
        self.device = torch.device(f'cuda:{num}')
        print(f'INSTA set device to {num}')

    def do_initialization(self, full_diff_sta: bool=False):
        """Perform the complete initialization sequence"""

        print('[reading noTiming file]')
        self._read_no_timing_pin_file()

        print('[reading valid pin file]')
        if not self._read_valid_pin_file():
            return False

        print('[reading cell arc file]')
        if not self._read_cell_arc_file():
            return False

        print('[reading net arc file]')
        if not self._read_net_arc_file():
            return False

        print('[building nx and gt graph]')
        if not self._build_graph():
            return False

        print('[reading sp file and launch clock rpt file]')
        if not self._read_sp_file():
            return False

        print('[reading ep file]')
        if not self._read_ep_file():
            return False

        print('[initializing timing groundtruths]')
        if not self._initialize_timing_groundtruths():
            return False

        print('[levelizing]')
        if not self._levelize():
            return False

        print('[reading pocvm guardband file]')
        if not self._read_pocvm_file():
            return False

        print('[precomputing collaterals]')
        if not self._precompute_collaterals():
            return False

        return True

    def do_eval_propagation(self, plot=False):
        if not self._propagate_arrival():
            return False

        if plot:
            print('[plotting]')
            if not self._plot_correlation():
                return False

        return True

    def do_diff_propagation(self, plot=False):
        if not self._diff_propagate_arrival():
            return False

        if plot:
            print('[plotting]')
            if not self._plot_correlation(topK=1):
                return False

        return True

    def do_extract_arc_grads(self, inPinMod=1):
        """
        Get timing gradients of each cell arc and net arc
        Combined then into stage-based gradient (per output pin)
        """
        assert self.level_2_collaterals is not None, "level_2_collaterals  not intiailized"
        assert self.cellArcId_2_cellArcKey is not None, "cell arc mapping not initialized"
        assert self.netArcId_2_netArcKey is not None, "net arc mapping not initialized"
        assert self.outPin_set, "output pin set not intiailized"
        assert self.Gid_2_pinName, "Gid name mapping not initialized"
        assert self.Gid_2_children, "Gid children mapping not initialized"
        assert self.Gid_2_parents, "Gid parent mapping not initialized"

        start_time = time.time()
        self.cellArc_2_riseFallGrads = extract_cellArc_grad(self.level_2_collaterals, self.cellArcId_2_cellArcKey)
        print(f"cell arc grad extraction time: {time.time() - start_time:.2}s")

        start_time = time.time()
        self.netArc_2_riseFallGrads = extract_netArc_grad(self.level_2_collaterals, self.netArcId_2_netArcKey)
        print(f"net arc grad extraction time: {time.time() - start_time:.2}s")

        start_time = time.time()
        self.outPinName_2_stageGrad = extract_stage_grad(
            self.outPin_set, self.Gid_2_pinName, self.Gid_2_children, self.Gid_2_parents,
            self.netArc_2_riseFallGrads, self.cellArc_2_riseFallGrads
        )
        print(f"stage grad extraction time: {time.time() - start_time:.2f}s")

    def _read_cell_libCell_file(self) -> bool:
        """Read cell to library cell mapping file"""
        file_path = os.path.join(self.prefix, self.input_folderName, 'cell_2_libCell.csv')
        self.cellName_2_orgLibCell, self.cellName_2_funcId, self.funcId_2_libCellNames, success = read_cell_libcell_file(
            file_path, self.save_dir
        )
        return success

    def _read_no_timing_pin_file(self) -> bool:
        """Read file containing pins to exclude from timing analysis"""
        file_path = os.path.join(self.prefix, self.input_folderName, 'no_timing_pins.csv')
        self.noTiming_pinNames_set, success = read_no_timing_pin_file(
            file_path, self.save_dir
        )
        return success

    def _read_valid_pin_file(self) -> bool:
        """Read file containing pins to include in timing analysis"""
        file_path = os.path.join(self.prefix, self.input_folderName, 'all_between_sp_ep_pins.csv')
        self.valid_pinNames_set, success = read_valid_pin_file(
            file_path, self.save_dir
        )
        return success

    def _read_cell_arc_file(self) -> bool:
        """Read timing arcs for cells"""
        file_path = os.path.join(self.prefix, self.input_folderName, 'cell_arcs.csv')
        self.cell_arc_2_variation, self.is_pocv, success = read_cell_arc_file(
            file_path, self.save_dir, self.scale
        )
        return success

    def _read_net_arc_file(self) -> bool:
        """Read timing arcs for nets"""
        file_path = os.path.join(self.prefix, self.input_folderName, 'net_arcs.csv')
        self.net_arc_2_variation, _, success = read_net_arc_file(
            file_path, self.save_dir, self.scale
        )
        return success

    def _build_graph(self) -> bool:
        """Build the timing graph from cell and net arcs"""
        # Create lists of arcs
        cell_arcs = list(self.cell_arc_2_variation.keys())
        net_arcs = list(self.net_arc_2_variation.keys())

        # Build the graph
        (
            self.pinG, self.gt_graph, self.nx_2_gt, self.gt_2_nx,
            self.pinName_2_Gid, self.Gid_2_pinName,
            self.cellName_2_pinNames, self.cellName_2_inPinNames, self.cellName_2_outPinNames,
            self.Gid_2_parents, self.Gid_2_children,
            self.inPin_parent_dict, self.outPin_set, self.max_Gid
        ) = build_timing_graph(
            cell_arcs, net_arcs, self.save_dir,
            self.valid_pinNames_set, self.noTiming_pinNames_set
        )

        return True


    def _read_sp_file(self) -> bool:
        """Read startpoint attributes file for timing propagation"""
        sp_file = os.path.join(self.prefix, self.input_folderName, 'sp_attributes.csv')
        rpt_file = os.path.join(self.prefix, self.input_folderName, 'clock_latency_timing_launch.rpt')
        self.spName_2_attributes, success = read_sp_file(
            sp_file, rpt_file, self.save_dir, self.pinName_2_Gid, self.scale
        )
        return success

    def _read_ep_file(self) -> bool:
        """Read endpoint attributes file for timing propagation"""
        ep_file = os.path.join(self.prefix, self.input_folderName, 'ep_attributes.csv')
        self.epName_riseFall_2_attributes, self.epName_riseFall_2_spName, self.epName_riseFall_2_ckPinName, \
        self.epName_riseFall_2_launch_clock_latency, success = read_ep_file(
            ep_file, self.save_dir, self.scale
        )
        return success

    def _initialize_timing_groundtruths(self) -> bool:
        """Initialize timing ground truth tensors from endpoint and startpoint data"""
        (
            self.ep_rise_arrival_truth, self.ep_rise_required_truth,
            self.ep_rise_slack_truth, self.ep_rise_depth_truth,
            self.ep_fall_arrival_truth, self.ep_fall_required_truth,
            self.ep_fall_slack_truth, self.ep_fall_depth_truth,
            self.sp_arrival_truth, self.sp_mean_tensor, self.sp_std_tensor,
            self.source_nodes, self.dest_nodes
        ) = initialize_timing_tensors(
            self.spName_2_attributes,
            self.epName_riseFall_2_attributes,
            self.pinName_2_Gid,
            self.max_Gid,
            self.float_dtype,
            self.save_dir
        )

        return True

    def _precompute_collaterals(self) -> bool:
        """Precompute timing collaterals for efficient propagation"""

        # Call the precompute_collaterals function from the timing module
        (
            self.level_2_collaterals,
            self.net_arc_2_collateral_loc,
            self.cell_arc_2_collateral_loc,
            self.cellArcId_2_cellName,
            self.cellArcKey_2_cellArcId,
            self.cellArcId_2_cellArcKey,
            self.netArcId_2_inCellName,
            self.netArcId_2_outCellName,
            self.netArcKey_2_netArcId,
            self.netArcId_2_netArcKey
        ) = precompute_collaterals(
            self.net_arc_2_variation,
            self.cell_arc_2_variation,
            self.sp_mean_tensor,
            self.sp_std_tensor,
            self.level_2_nodes,
            self.Gid_2_pinName,
            self.inPin_parent_tensor,
            self.Gid_2_parents,
            self.device,
            self.max_Gid,
            self.cellName_2_orgLibCell,
            self.libCell_2_riseFallguardband,
            self.libCell_2_riseFallStd,
            self.net_2_pocvScaling,
            self.float_dtype,
            self.save_dir
        )

        # Find max IDs for both cell and net arcs
        if self.cellArcId_2_cellName:
            self.max_cellArcId = max(self.cellArcId_2_cellName.keys()) + 1
        else:
            self.max_cellArcId = 0

        if self.netArcId_2_netArcKey:
            self.max_netArcId = max(self.netArcId_2_netArcKey.keys()) + 1
        else:
            self.max_netArcId = 0

        # Move collaterals to device if necessary
        if str(self.device) != 'cpu':
            self.level_2_collaterals = move_collaterals_to_device(
                self.level_2_collaterals,
                self.device
            )

        return True

    def _read_pocvm_file(self) -> bool:
        """Read Process, Operating conditions, and Voltage variation Model files"""
        (
            self.net_2_pocvScaling,
            self.libCell_2_riseFallguardband,
            self.libCell_2_riseFallStd,
            success
        ) = read_pocvm_file(
            os.path.join(self.prefix, self.input_folderName),
            self.save_dir
        )
        return success


    def _levelize(self) -> bool:
        """Levelize the timing graph for efficient propagation"""
        (
            self.level_2_nodes, self.node_2_level,
            self.inPin_parent_tensor,
            self.level_2_nodes_bw, self.node_2_level_bw
        ) = levelize_graph(
            self.gt_graph,
            self.gt_2_nx,
            self.source_nodes,
            self.dest_nodes,
            self.Gid_2_parents,
            self.Gid_2_children,
            self.inPin_parent_dict,
            self.save_dir
        )

        return True

    def _propagate_arrival(self) -> bool:
        """Propagate arrival times through the timing graph"""
        # Clear timing cache and initialize timing tensors
        self.timing_tensors = clear_timing_cache(
            self.max_Gid,
            self.topK,
            self.device,
            self.float_dtype,
            self.sp_mean_tensor,
            self.sp_std_tensor,
            self.ep_rise_required_truth,
            self.ep_fall_required_truth,
            self.epName_riseFall_2_spName,
            self.pinName_2_Gid,
            self.source_nodes,
            self.dest_nodes,
            self.timing_tensors
        )

        # Propagate arrival times
        self.timing_tensors, self.wns, self.tns = propagate_arrival_times(
            self.timing_tensors,
            self.level_2_collaterals,
            self.inPin_parent_tensor,
            self.device,
            self.max_Gid,
            self.float_dtype
        )

        # Save arrival tensors
        save_arrival_tensors(self.timing_tensors, self.save_dir)

        return True

    def _diff_propagate_arrival(self) -> bool:
        """Propagate arrival times through the timing graph"""
        # Clear timing cache and initialize timing tensors

        topk = 1
        self.timing_tensors = clear_timing_cache(
            self.max_Gid,
            topk,
            self.device,
            self.float_dtype,
            self.sp_mean_tensor,
            self.sp_std_tensor,
            self.ep_rise_required_truth,
            self.ep_fall_required_truth,
            self.epName_riseFall_2_spName,
            self.pinName_2_Gid,
            self.source_nodes,
            self.dest_nodes,
            self.timing_tensors,
            is_diff_prop=True
        )

        # Propagate arrival times
        self.timing_tensors, self.wns, self.tns = propagate_arrival_times(
            self.timing_tensors,
            self.level_2_collaterals,
            self.inPin_parent_tensor,
            self.device,
            self.max_Gid,
            self.float_dtype,
            is_diff_prop=True,
            topk=topk
        )

        # Save arrival tensors
        save_arrival_tensors(self.timing_tensors, self.save_dir)

        return True


    def _plot_correlation(self, topK=256) -> bool:
        """
        Generate correlation plots and analysis of timing results.
        """
        tensor_extraction_params = {
            'rise_arrival': self.timing_tensors['Gid_2_rise_arrival'],
            'rise_startpoints': self.timing_tensors['Gid_2_rise_startpoints'],
            'rise_required': self.timing_tensors['ep_rise_required_truth'],
            'fall_arrival': self.timing_tensors['Gid_2_fall_arrival'],
            'fall_startpoints': self.timing_tensors['Gid_2_fall_startpoints'],
            'fall_required': self.timing_tensors['ep_fall_required_truth'],
        }

        timing_tensors_cpu = {k: v.cpu() for k, v in tensor_extraction_params.items()}


        temporal_metadata = apply_cppr_correction(
            timing_tensors_cpu,
            self.dest_nodes,
            self.Gid_2_pinName,
            self.pinName_2_Gid,
            self.epName_riseFall_2_spName,
            topK,
            self.ep_rise_depth_truth,
            self.ep_rise_slack_truth,
            self.ep_rise_arrival_truth,
            self.ep_fall_depth_truth,
            self.ep_fall_slack_truth,
            self.ep_fall_arrival_truth,
            to_filter=self.filter
        )

        ep_levels = torch.tensor(
            [self.node_2_level.get(ep, -1) for ep in temporal_metadata['d_pins']],
            dtype=torch.int32
        )

        true_levels = torch.tensor(temporal_metadata['true_levels'], dtype=self.float_dtype)
        true_slacks = torch.tensor(temporal_metadata['true_slacks'], dtype=self.float_dtype)
        true_requireds = torch.tensor(temporal_metadata['true_requireds'], dtype=self.float_dtype)
        true_arrivals = torch.tensor(temporal_metadata['true_arrivals'], dtype=self.float_dtype)
        pred_arrivals = torch.tensor(temporal_metadata['pred_arrivals'], dtype=self.float_dtype)
        pred_slacks = true_requireds - pred_arrivals

        plot_ep_correlation(
            "",
            true_slacks,
            pred_slacks,
            ep_levels,
            true_levels,
            self.save_dir,
            len(self.pinName_2_Gid),
            self.design_name,
            cmap='RdYlBu_r',
            tight_layout=True,
            plot_format='png',
            s=46,
            alpha=1,
            rasterized=True, # save memory
        )

        analysis_dir = os.path.join(self.save_dir, 'plots')
        os.makedirs(analysis_dir, exist_ok=True)

        with open(os.path.join(analysis_dir, 'anal.csv'), 'w') as analysis_file:
            analysis_file.write(
                'epName,true_slack,pred_slack,diff_slack,true_arr,pred_arr,diff_arr,required,spName,riseFall\n')
            diff_slacks = true_slacks - pred_slacks
            diff_arrivals = true_arrivals - pred_arrivals

            sort_indices = torch.argsort(torch.abs(diff_slacks), descending=True)

            for record_index in sort_indices.tolist():
                analysis_file.write(
                    f'{temporal_metadata["epNames"][record_index]},'
                    f'{true_slacks[record_index]},'
                    f'{pred_slacks[record_index]},'
                    f'{diff_slacks[record_index]},'
                    f'{true_arrivals[record_index]},'
                    f'{pred_arrivals[record_index]},'
                    f'{diff_arrivals[record_index]},'
                    f'{true_requireds[record_index]},'
                    f'{temporal_metadata["spNames"][record_index]},'
                    f'{temporal_metadata["riseFalls"][record_index]}\n'
                )

        return True


