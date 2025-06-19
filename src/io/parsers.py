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
# @file io/parsers.py
# @brief functions to parse various input files

import os
import math
import glob
import collections
from typing import Dict, List, Set, Tuple, Optional, Union, Any

import polars as pl
import torch
import gc

# Use absolute imports without src prefix
from .serialization import save_pickle, load_pickle


def read_cell_libcell_file(
    file_path: str,
    save_dir: str,
    use_cache: bool = True
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]], bool]:
    """
    Read cell to library cell mapping file

    Args:
        file_path: Path to the cell mapping CSV file
        save_dir: Directory to save/load cache
        use_cache: Whether to use cached results if available

    Returns:
        Tuple of (cell_to_libcell_map, cell_to_funcid_map, funcid_to_libcell_map, success)
    """
    # Initialize return values
    cell_to_libcell = {}
    cell_to_funcid = {}
    funcid_to_libcell = collections.defaultdict(set)

    # Check if file exists
    if not os.path.exists(file_path):
        print(f'[read cell libCell file] error: cannot find file {file_path}')
        return cell_to_libcell, cell_to_funcid, funcid_to_libcell, False

    # Check for cached results
    if use_cache:
        cache_path1 = os.path.join(save_dir, "cellName_2_orgLibCell.pkl")
        cache_path2 = os.path.join(save_dir, "cellName_2_funcId.pkl")
        cache_path3 = os.path.join(save_dir, "funcId_2_libCellNames.pkl")

        if all(os.path.exists(p) for p in [cache_path1, cache_path2, cache_path3]):
            cell_to_libcell = load_pickle(cache_path1, {})
            cell_to_funcid = load_pickle(cache_path2, {})
            funcid_to_libcell = load_pickle(cache_path3, collections.defaultdict(set))
            return cell_to_libcell, cell_to_funcid, funcid_to_libcell, True

    # Parse the file
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if not parts:
                continue

            cell_name = parts[0]
            lib_cell = parts[1]
            func_id = ','.join(parts[2:])

            cell_to_libcell[cell_name] = lib_cell

            if func_id != 'unknown':
                cell_to_funcid[cell_name] = func_id
                funcid_to_libcell[func_id].add(lib_cell)

    # Save results
    save_pickle(cell_to_libcell, os.path.join(save_dir, "cellName_2_orgLibCell.pkl"))
    save_pickle(cell_to_funcid, os.path.join(save_dir, "cellName_2_funcId.pkl"))
    save_pickle(funcid_to_libcell, os.path.join(save_dir, "funcId_2_libCellNames.pkl"))

    return cell_to_libcell, cell_to_funcid, funcid_to_libcell, True


def read_no_timing_pin_file(
    file_path: str,
    save_dir: str,
    use_cache: bool = True
) -> Tuple[Set[str], bool]:
    """
    Read file containing pins to exclude from timing analysis

    Args:
        file_path: Path to the no-timing pins CSV file
        save_dir: Directory to save/load cache
        use_cache: Whether to use cached results if available

    Returns:
        Tuple of (notiming_pins_set, success)
    """
    # Initialize return values
    notiming_pins = set()

    # Check if file exists
    if not os.path.exists(file_path):
        print(f'[read noTiming warning] cannot find file {file_path}')
        return notiming_pins, True  # Not an error, just empty set

    # Check for cached results
    if use_cache:
        cache_path = os.path.join(save_dir, "noTiming_pinNames_set.pkl")
        if os.path.exists(cache_path):
            notiming_pins = load_pickle(cache_path, set())
            return notiming_pins, True

    # Read file with Polars
    try:
        df = pl.read_csv(file_path, infer_schema_length=0)
        notiming_pins = set(df['full_name'])

        # Save results
        save_pickle(notiming_pins, os.path.join(save_dir, "noTiming_pinNames_set.pkl"))

        # Clean up
        del df
        gc.collect()

        return notiming_pins, True
    except Exception as e:
        print(f"Error reading no-timing pins file: {e}")
        return set(), False


def read_valid_pin_file(
    file_path: str,
    save_dir: str,
    use_cache: bool = True
) -> Tuple[Set[str], bool]:
    """
    Read file containing pins to include in timing analysis

    Args:
        file_path: Path to the valid pins CSV file
        save_dir: Directory to save/load cache
        use_cache: Whether to use cached results if available

    Returns:
        Tuple of (valid_pins_set, success)
    """
    # Initialize return values
    valid_pins = set()

    # Check if file exists
    if not os.path.exists(file_path):
        print(f'[read valid pin file] error: cannot find file {file_path}')
        return valid_pins, False

    # Check for cached results
    if use_cache:
        cache_path = os.path.join(save_dir, "valid_pinNames_set.pkl")
        if os.path.exists(cache_path):
            valid_pins = load_pickle(cache_path, set())
            return valid_pins, True

    # Read file with Polars
    try:
        df = pl.read_csv(file_path, infer_schema_length=0)
        valid_pins = set(df['full_name'])

        # Save results
        save_pickle(valid_pins, os.path.join(save_dir, "valid_pinNames_set.pkl"))

        # Clean up
        del df
        gc.collect()

        return valid_pins, True
    except Exception as e:
        print(f"Error reading valid pins file: {e}")
        return set(), False


def _is_column_empty(df: pl.DataFrame, column_name: str) -> bool:
    """Check if a column in dataframe is empty"""
    return df.select((pl.col(column_name) == '').all()).to_series().item()


def _filter_arc_dataframe(
    df: pl.DataFrame,
    mode: str = 'net',
    scale: float = 1.0
) -> Tuple[pl.DataFrame, bool]:
    """
    Process timing arc dataframe based on format (standard or POCV)

    Args:
        df: Input dataframe
        mode: 'cell' or 'net'
        scale: Scaling factor for delays

    Returns:
        Tuple of (processed_dataframe, is_pocv)
    """
    # Check if this is a POCV dataframe
    has_variation = not _is_column_empty(df, 'variation_delay_max_rise.mean')

    if not has_variation:
        # Standard timing format
        if mode == 'cell':
            df = df.select(['from_pin.full_name', 'to_pin.full_name', 'sense',
                           'delay_max_rise', 'delay_max_fall'])
        else:
            df = df.select(['from_pin.full_name', 'to_pin.full_name',
                           'delay_max_rise', 'delay_max_fall'])

        # Convert to float and drop nulls
        df = df.with_columns(
            pl.col("delay_max_rise").cast(pl.Float32),
            pl.col("delay_max_fall").cast(pl.Float32)
        ).drop_nulls()

        is_pocv = False
    else:
        # POCV format with mean and standard deviation
        df = df.with_columns(
            # Handle uninit values
            pl.when(pl.col('variation_delay_max_rise.std_dev') == "UNINIT").then(0.0)
              .otherwise(pl.col('variation_delay_max_rise.std_dev')).cast(pl.Float32)
              .alias('variation_delay_max_rise.std_dev'),
            pl.when(pl.col('variation_delay_max_fall.std_dev') == "UNINIT").then(0.0)
              .otherwise(pl.col('variation_delay_max_fall.std_dev')).cast(pl.Float32)
              .alias('variation_delay_max_fall.std_dev'),
            pl.col('variation_delay_max_rise.mean').cast(pl.Float32),
            pl.col('variation_delay_max_fall.mean').cast(pl.Float32)
        ).drop_nulls()

        # Compute corner delay (mean + 3*sigma)
        df = df.with_columns(
            (pl.col('variation_delay_max_rise.mean') + 3.0 * pl.col('variation_delay_max_rise.std_dev'))
            .alias('corner_rise_delay'),
            (pl.col('variation_delay_max_fall.mean') + 3.0 * pl.col('variation_delay_max_fall.std_dev'))
            .alias('corner_fall_delay')
        )

        # Select relevant columns
        if mode == 'cell':
            df = df.select([
                'from_pin.full_name', 'to_pin.full_name', 'sense',
                'corner_rise_delay', 'corner_fall_delay',
                'variation_delay_max_rise.mean', 'variation_delay_max_rise.std_dev',
                'variation_delay_max_fall.mean', 'variation_delay_max_fall.std_dev'
            ])
        else:
            df = df.select([
                'from_pin.full_name', 'to_pin.full_name',
                'corner_rise_delay', 'corner_fall_delay',
                'variation_delay_max_rise.mean', 'variation_delay_max_rise.std_dev',
                'variation_delay_max_fall.mean', 'variation_delay_max_fall.std_dev'
            ])

        is_pocv = True

    return df, is_pocv


def _extract_cell_arc_variations(
    df: pl.DataFrame,
    is_pocv: bool,
    scale: float = 1.0
) -> Dict:
    """
    Extract timing variation data from cell arc dataframe

    Args:
        df: Processed dataframe
        is_pocv: Whether using POCV format
        scale: Scaling factor for delays

    Returns:
        Dictionary of cell arc variations
    """
    variations = {}

    if not is_pocv:
        # Standard timing format
        # Group by arc and find max rise delay
        max_rise_rows = df.group_by(["from_pin.full_name", "to_pin.full_name", "sense"]).agg(
            pl.max("delay_max_rise").alias('final_max_rise')
        )
        df_max_rise = df.join(max_rise_rows, on=["from_pin.full_name", "to_pin.full_name", "sense"])
        df_max_rise = df_max_rise.filter(
            pl.col("delay_max_rise") == pl.col("final_max_rise")
        ).unique(subset=["from_pin.full_name", "to_pin.full_name", "sense"])

        # Create dictionary of rise variations
        rise_variations = {
            (row["from_pin.full_name"], row["to_pin.full_name"], row["sense"]):
            (row["delay_max_rise"] / scale,)
            for row in df_max_rise.to_dicts()
        }

        # Group by arc and find max fall delay
        max_fall_rows = df.group_by(["from_pin.full_name", "to_pin.full_name", "sense"]).agg(
            pl.max("delay_max_fall").alias('final_max_fall')
        )
        df_max_fall = df.join(max_fall_rows, on=["from_pin.full_name", "to_pin.full_name", "sense"])
        df_max_fall = df_max_fall.filter(
            pl.col("delay_max_fall") == pl.col("final_max_fall")
        ).unique(subset=["from_pin.full_name", "to_pin.full_name", "sense"])

        # Create dictionary of fall variations
        fall_variations = {
            (row["from_pin.full_name"], row["to_pin.full_name"], row["sense"]):
            (row["delay_max_fall"] / scale,)
            for row in df_max_fall.to_dicts()
        }
    else:
        # POCV format with mean and standard deviation
        # Group by arc and find max rise delay corner
        max_rise_rows = df.group_by(["from_pin.full_name", "to_pin.full_name", "sense"]).agg(
            pl.max("corner_rise_delay").alias('corner_max_rise')
        )
        df_max_rise = df.join(max_rise_rows, on=["from_pin.full_name", "to_pin.full_name", "sense"])
        df_max_rise = df_max_rise.filter(
            pl.col("corner_rise_delay") == pl.col("corner_max_rise")
        ).unique(subset=["from_pin.full_name", "to_pin.full_name", "sense"])

        # Create dictionary of rise variations
        rise_variations = {
            (row["from_pin.full_name"], row["to_pin.full_name"], row["sense"]):
            (row["variation_delay_max_rise.mean"] / scale,
             row["variation_delay_max_rise.std_dev"] / scale)
            for row in df_max_rise.to_dicts()
        }

        # Group by arc and find max fall delay corner
        max_fall_rows = df.group_by(["from_pin.full_name", "to_pin.full_name", "sense"]).agg(
            pl.max("corner_fall_delay").alias('corner_max_fall')
        )
        df_max_fall = df.join(max_fall_rows, on=["from_pin.full_name", "to_pin.full_name", "sense"])
        df_max_fall = df_max_fall.filter(
            pl.col("corner_fall_delay") == pl.col("corner_max_fall")
        ).unique(subset=["from_pin.full_name", "to_pin.full_name", "sense"])

        # Create dictionary of fall variations
        fall_variations = {
            (row["from_pin.full_name"], row["to_pin.full_name"], row["sense"]):
            (row["variation_delay_max_fall.mean"] / scale,
             row["variation_delay_max_fall.std_dev"] / scale)
            for row in df_max_fall.to_dicts()
        }

    # Verify we have the same arcs for rise and fall
    assert len(rise_variations) == len(fall_variations), "Mismatch in rise/fall variations count"

    # Combine rise and fall variations
    for key, rise_val in rise_variations.items():
        assert key in fall_variations, f"Missing fall variation for {key}"
        variations[key] = (*rise_val, *fall_variations[key])

    return variations


def read_cell_arc_file(
    file_path: str,
    save_dir: str,
    scale: float = 1.0,
    use_cache: bool = True
) -> Tuple[Dict, bool, bool]:
    """
    Read timing arcs for cells

    Args:
        file_path: Path to the cell arcs CSV file
        save_dir: Directory to save/load cache
        scale: Scaling factor for delays
        use_cache: Whether to use cached results if available

    Returns:
        Tuple of (cell_arc_variations, is_pocv, success)
    """
    # Check for cached results
    if use_cache:
        cache_path = os.path.join(save_dir, 'cell_arc_2_variation.pkl')
        if os.path.exists(cache_path):
            cell_arc_variations = load_pickle(cache_path, {})
            # Determine POCV by checking the first variation's length
            first_value = next(iter(cell_arc_variations.values()), None)
            is_pocv = first_value is not None and len(first_value) == 4
            return cell_arc_variations, is_pocv, True

    # Read file with Polars
    try:
        df = pl.read_csv(file_path, infer_schema_length=0)
        print(f'[cell arc parsing] raw df shape: {df.shape}')

        # Filter valid senses
        valid_senses = ['positive_unate', 'negative_unate', 'rising_edge', 'falling_edge']
        df = df.filter(pl.col('sense').is_in(valid_senses))
        print(f'[cell arc parsing] sense-filtered df shape: {df.shape}')

        # Process cell arcs based on format (standard vs POCV)
        df, is_pocv = _filter_arc_dataframe(df, mode='cell', scale=scale)
        print(f'[cell arc parsing] final filtered df shape: {df.shape}')

        # Extract timing arc data
        cell_arc_variations = _extract_cell_arc_variations(df, is_pocv, scale)
        print(f'[cell arc parsing] num valid cell arcs: {len(cell_arc_variations)}')

        # Save results
        save_pickle(cell_arc_variations, os.path.join(save_dir, 'cell_arc_2_variation.pkl'))

        # Clean up
        del df
        gc.collect()

        return cell_arc_variations, is_pocv, True
    except Exception as e:
        print(f"Error reading cell arc file: {e}")
        return {}, False, False


def _extract_net_arc_variations(
    df: pl.DataFrame,
    is_pocv: bool,
    scale: float = 1.0
) -> Dict:
    """
    Extract timing variation data from net arc dataframe

    Args:
        df: Processed dataframe
        is_pocv: Whether using POCV format
        scale: Scaling factor for delays

    Returns:
        Dictionary of net arc variations
    """
    variations = {}

    if not is_pocv:
        # Standard timing format
        # Group by arc and find max rise delay
        max_rise_rows = df.group_by(["from_pin.full_name", "to_pin.full_name"]).agg(
            pl.max("delay_max_rise").alias('final_max_rise')
        )
        df_max_rise = df.join(max_rise_rows, on=["from_pin.full_name", "to_pin.full_name"])
        df_max_rise = df_max_rise.filter(
            pl.col("delay_max_rise") == pl.col("final_max_rise")
        ).unique(subset=["from_pin.full_name", "to_pin.full_name"])

        # Create dictionary of rise variations
        rise_variations = {
            (row["from_pin.full_name"], row["to_pin.full_name"]):
            (row["delay_max_rise"] / scale,)
            for row in df_max_rise.to_dicts()
        }

        # Group by arc and find max fall delay
        max_fall_rows = df.group_by(["from_pin.full_name", "to_pin.full_name"]).agg(
            pl.max("delay_max_fall").alias('final_max_fall')
        )
        df_max_fall = df.join(max_fall_rows, on=["from_pin.full_name", "to_pin.full_name"])
        df_max_fall = df_max_fall.filter(
            pl.col("delay_max_fall") == pl.col("final_max_fall")
        ).unique(subset=["from_pin.full_name", "to_pin.full_name"])

        # Create dictionary of fall variations
        fall_variations = {
            (row["from_pin.full_name"], row["to_pin.full_name"]):
            (row["delay_max_fall"] / scale,)
            for row in df_max_fall.to_dicts()
        }
    else:
        # POCV format with mean and standard deviation
        # Group by arc and find max rise delay corner
        max_rise_rows = df.group_by(["from_pin.full_name", "to_pin.full_name"]).agg(
            pl.max("corner_rise_delay").alias('corner_max_rise')
        )
        df_max_rise = df.join(max_rise_rows, on=["from_pin.full_name", "to_pin.full_name"])
        df_max_rise = df_max_rise.filter(
            pl.col("corner_rise_delay") == pl.col("corner_max_rise")
        ).unique(subset=["from_pin.full_name", "to_pin.full_name"])

        # Create dictionary of rise variations
        rise_variations = {
            (row["from_pin.full_name"], row["to_pin.full_name"]):
            (row["variation_delay_max_rise.mean"] / scale,
             row["variation_delay_max_rise.std_dev"] / scale)
            for row in df_max_rise.to_dicts()
        }

        # Group by arc and find max fall delay corner
        max_fall_rows = df.group_by(["from_pin.full_name", "to_pin.full_name"]).agg(
            pl.max("corner_fall_delay").alias('corner_max_fall')
        )
        df_max_fall = df.join(max_fall_rows, on=["from_pin.full_name", "to_pin.full_name"])
        df_max_fall = df_max_fall.filter(
            pl.col("corner_fall_delay") == pl.col("corner_max_fall")
        ).unique(subset=["from_pin.full_name", "to_pin.full_name"])

        # Create dictionary of fall variations
        fall_variations = {
            (row["from_pin.full_name"], row["to_pin.full_name"]):
            (row["variation_delay_max_fall.mean"] / scale,
             row["variation_delay_max_fall.std_dev"] / scale)
            for row in df_max_fall.to_dicts()
        }

    # Verify we have the same arcs for rise and fall
    assert len(rise_variations) == len(fall_variations), "Mismatch in rise/fall variations count"

    # Combine rise and fall variations
    for key, rise_val in rise_variations.items():
        assert key in fall_variations, f"Missing fall variation for {key}"
        variations[key] = (*rise_val, *fall_variations[key])

    return variations


def read_net_arc_file(
    file_path: str,
    save_dir: str,
    scale: float = 1.0,
    use_cache: bool = True
) -> Tuple[Dict, bool, bool]:
    """
    Read timing arcs for nets

    Args:
        file_path: Path to the net arcs CSV file
        save_dir: Directory to save/load cache
        scale: Scaling factor for delays
        use_cache: Whether to use cached results if available

    Returns:
        Tuple of (net_arc_variations, is_pocv, success)
    """
    # Check for cached results
    if use_cache:
        cache_path = os.path.join(save_dir, 'net_arc_2_variation.pkl')
        if os.path.exists(cache_path):
            net_arc_variations = load_pickle(cache_path, {})
            # Determine POCV by checking the first variation's length
            first_value = next(iter(net_arc_variations.values()), None)
            is_pocv = first_value is not None and len(first_value) == 4
            return net_arc_variations, is_pocv, True

    # Read file with Polars
    try:
        df = pl.read_csv(file_path, infer_schema_length=0)

        # Process net arcs based on format (standard vs POCV)
        df, is_pocv = _filter_arc_dataframe(df, mode='net', scale=scale)

        # Extract timing arc data
        net_arc_variations = _extract_net_arc_variations(df, is_pocv, scale)
        print(f'[net arc parsing] num valid net arcs: {len(net_arc_variations)}')

        # Save results
        save_pickle(net_arc_variations, os.path.join(save_dir, 'net_arc_2_variation.pkl'))

        # Clean up
        del df
        gc.collect()

        return net_arc_variations, is_pocv, True
    except Exception as e:
        print(f"Error reading net arc file: {e}")
        return {}, False, False


def read_clock_latency_rpt(
    rpt_file: str,
    pin_to_id_map: Dict[str, int]
) -> Dict[str, Tuple[float, float, float]]:
    """
    Read clock latency report for setup/hold analysis

    Args:
        rpt_file: Path to the clock latency report file
        pin_to_id_map: Dictionary mapping pin names to their IDs

    Returns:
        Dictionary mapping startpoint names to (arrival, mean, std) tuples
    """
    sp_attributes = {}

    if not os.path.exists(rpt_file):
        return sp_attributes

    with open(rpt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            sp_name = parts[0]
            if sp_name not in pin_to_id_map:
                continue

            # Extract timing data
            sp_std = float(parts[4])
            sp_mean = float(parts[5])
            sp_arr = float(parts[6])

            # Verify mean+3*std = arr (standard sigma value in STA)
            if not math.isclose(sp_arr, sp_mean + 3.0 * sp_std, rel_tol=1e-5, abs_tol=1e-5):
                print(f"Warning: Inconsistent values for {sp_name}: arr={sp_arr}, mean+3*std={sp_mean + 3.0*sp_std}")

            sp_attributes[sp_name] = (sp_arr, sp_mean, sp_std)

    return sp_attributes


def read_sp_file(
    sp_file: str,
    rpt_file: str,
    save_dir: str,
    pin_to_id_map: Dict[str, int],
    scale: float = 1.0,
    use_cache: bool = True
) -> Tuple[Dict[str, Tuple[float, float, float]], bool]:
    """
    Read startpoint attributes file for timing propagation

    Args:
        sp_file: Path to the startpoint attributes CSV file
        rpt_file: Path to the additional clock latency report
        save_dir: Directory to save/load cache
        pin_to_id_map: Dictionary mapping pin names to their IDs
        scale: Scaling factor for delays
        use_cache: Whether to use cached results if available

    Returns:
        Tuple of (sp_attributes, success)
    """
    # Check if file exists
    if not os.path.exists(sp_file):
        print(f'[read sp file] error: cannot find file {sp_file}')
        return {}, False

    # Check for cached results
    if use_cache:
        cache_path = os.path.join(save_dir, "spName_2_attributes.pkl")
        if os.path.exists(cache_path):
            sp_attributes = load_pickle(cache_path, {})
            return sp_attributes, True

    # Initialize startpoint attributes
    sp_attributes = {}

    try:
        # Read and process file using Polars
        df = pl.read_csv(sp_file, infer_schema_length=0)
        print(f'[read sp file] initial df size: {df.shape}')

        # Filter to keep only pins in our graph
        df = df.filter(pl.col('sp_name').is_in(set(pin_to_id_map.keys())))
        print(f'[read sp file] df size after name filtering: {df.shape}')

        # Convert and clean numeric columns
        df = df.with_columns(
            pl.col('launch_clock_latency').cast(pl.Float32),
            pl.col('clock_latency_mean').cast(pl.Float32),
            pl.col('clock_latency_std').cast(pl.Float32)
        ).drop_nulls()

        # Extract timing attributes with scaling
        for row in df.to_dicts():
            sp_attributes[row['sp_name']] = (
                row['launch_clock_latency'] / scale,
                row['clock_latency_mean'] / scale,
                row['clock_latency_std'] / scale
            )

        print(f'[read sp file] # sps from get_timing_paths alone: {len(sp_attributes)}')

        # Read additional clock latency report if it exists
        if os.path.exists(rpt_file):
            clock_latencies = read_clock_latency_rpt(rpt_file, pin_to_id_map)
            sp_attributes.update(clock_latencies)
            print(f'[read sp file] # sps after reading clk rpt: {len(sp_attributes)}')

        # Save results
        save_pickle(sp_attributes, os.path.join(save_dir, "spName_2_attributes.pkl"))

        # Clean up
        del df
        gc.collect()

        return sp_attributes, True
    except Exception as e:
        print(f"Error reading startpoint file: {e}")
        return {}, False


def read_ep_file(
    ep_file: str,
    save_dir: str,
    scale: float = 1.0,
    use_cache: bool = True
) -> Tuple[Dict, Dict, Dict, Dict, bool]:
    """
    Read endpoint attributes file for timing propagation

    Args:
        ep_file: Path to the endpoint attributes CSV file
        save_dir: Directory to save/load cache
        scale: Scaling factor for delays
        use_cache: Whether to use cached results if available

    Returns:
        Tuple of (ep_attributes, ep_to_sp_map, ep_to_ck_map, ep_to_latency_map, success)
    """
    # Initialize dictionaries
    ep_attributes = {}
    ep_to_sp_map = {}
    ep_to_ck_map = {}
    ep_to_latency_map = {}

    # Check if file exists
    if not os.path.exists(ep_file):
        print(f'[read ep file] error: cannot find file {ep_file}')
        return ep_attributes, ep_to_sp_map, ep_to_ck_map, ep_to_latency_map, False

    # Check for cached results
    if use_cache:
        cache_path1 = os.path.join(save_dir, "epName_riseFall_2_attributes.pkl")
        cache_path2 = os.path.join(save_dir, "epName_riseFall_2_spName.pkl")
        cache_path3 = os.path.join(save_dir, "epName_riseFall_2_launch_clock_latency.pkl")
        cache_path4 = os.path.join(save_dir, "epName_riseFall_2_ckPinName.pkl")

        if all(os.path.exists(p) for p in [cache_path1, cache_path2, cache_path3]):
            ep_attributes = load_pickle(cache_path1, {})
            ep_to_sp_map = load_pickle(cache_path2, {})
            ep_to_latency_map = load_pickle(cache_path3, {})

            if os.path.exists(cache_path4):
                ep_to_ck_map = load_pickle(cache_path4, {})

            return ep_attributes, ep_to_sp_map, ep_to_ck_map, ep_to_latency_map, True

    try:
        # Read file with Polars
        df = pl.read_csv(ep_file, infer_schema_length=0)
        print(f'[read ep file] initial df size: {df.shape}')

        # Convert and clean numeric columns
        df = df.with_columns(
            pl.col('ep_max_rise_slack').cast(pl.Float32),
            pl.col('ep_max_rise_arrival').cast(pl.Float32),
            pl.col('ep_max_fall_slack').cast(pl.Float32),
            pl.col('ep_max_fall_arrival').cast(pl.Float32),
            pl.col('launch_clock_latency').cast(pl.Float32),
            pl.col('clock_latency_mean').cast(pl.Float32),
            pl.col('clock_latency_std').cast(pl.Float32),
            pl.col('path_slack').cast(pl.Float32),
            pl.col('path_arrival').cast(pl.Float32),
            pl.col('path_required').cast(pl.Float32),
            pl.col('path_num_points').cast(pl.Float32),
            pl.col('path_crpr').cast(pl.Float32)
        ).drop_nulls()

        print(f'[read ep file] df size after filtering: {df.shape}')

        # Extract endpoint attributes with scaling
        for row in df.to_dicts():
            ep_name = row['ep_name']
            rise_fall = row['rise_fall']

            # Create tuple key
            key = (ep_name, rise_fall)

            # Store path attributes
            ep_attributes[key] = (
                row['path_arrival'] / scale,
                row['path_required'] / scale,
                row['path_slack'] / scale,
                row['path_num_points'] / scale,
                row['ep_max_rise_slack'] / scale,
                row['ep_max_fall_slack'] / scale,
                row['ep_max_rise_arrival'] / scale,
                row['ep_max_fall_arrival'] / scale,
                row['path_crpr'] / scale
            )

            # Store startpoint mapping
            ep_to_sp_map[key] = row['sp_name']

            # Store launch clock latency
            ep_to_latency_map[key] = row['launch_clock_latency']

            # Store clock pin name if available
            if 'ep_ck_name' in row:
                ep_to_ck_map[key] = row['ep_ck_name']

        # Save results
        save_pickle(ep_attributes, os.path.join(save_dir, "epName_riseFall_2_attributes.pkl"))
        save_pickle(ep_to_sp_map, os.path.join(save_dir, "epName_riseFall_2_spName.pkl"))
        save_pickle(ep_to_latency_map, os.path.join(save_dir, "epName_riseFall_2_launch_clock_latency.pkl"))

        if 'ep_ck_name' in df.columns:
            save_pickle(ep_to_ck_map, os.path.join(save_dir, "epName_riseFall_2_ckPinName.pkl"))

        # Clean up
        del df
        gc.collect()

        return ep_attributes, ep_to_sp_map, ep_to_ck_map, ep_to_latency_map, True
    except Exception as e:
        print(f"Error reading endpoint file: {e}")
        return {}, {}, {}, {}, False


def read_pocv_rpt(file_path: str) -> Dict[str, List[float]]:
    """
    Read POCV guardband/std report file

    Args:
        file_path: Path to the POCV report file

    Returns:
        Dictionary mapping library cells to [rise, fall] values
    """
    result = {}

    if not os.path.exists(file_path):
        return result

    lib_cell = None

    with open(file_path, 'r') as f:
        for line in f:
            if 'lib_cell:' in line:
                lib_cell = line.strip().split()[-1].split('/')[-1]
            elif 'Cell delay' in line:
                parts = line.strip().split()

                # Parse rise value
                if parts[-3] == '--':
                    rise_value = 1.0
                else:
                    rise_value = float(parts[-3])

                # Parse fall value
                if parts[-1] == '--':
                    fall_value = 1.0
                else:
                    fall_value = float(parts[-1])

                # Store values
                if lib_cell is None:
                    result['design'] = [rise_value, fall_value]
                else:
                    result[lib_cell] = [rise_value, fall_value]

    return result


def read_pocvm_file(
    folder_path: str,
    save_dir: str,
    use_cache: bool = True
) -> Tuple[Dict, Dict, Dict, bool]:
    """
    Read Process, Operating conditions, and Voltage variation Model files

    Args:
        folder_path: Directory containing POCVM files
        save_dir: Directory to save/load cache
        use_cache: Whether to use cached results if available

    Returns:
        Tuple of (net_scaling, guardband, std_coef, success)
    """
    # Check for cached results
    if use_cache:
        cache_path1 = os.path.join(save_dir, "net_2_pocvScaling.pkl")
        cache_path2 = os.path.join(save_dir, "libCell_2_riseFallguardband.pkl")
        cache_path3 = os.path.join(save_dir, "libCell_2_riseFallStd.pkl")

        if all(os.path.exists(p) for p in [cache_path1, cache_path2, cache_path3]):
            net_scaling = load_pickle(cache_path1, {})
            guardband = load_pickle(cache_path2, {})
            std_coef = load_pickle(cache_path3, {})
            return net_scaling, guardband, std_coef, True

    # Initialize dictionaries with defaults
    net_scaling = {'net': [1.0, 1.0]}
    guardband = {'design': [1.0, 1.0]}
    std_coef = {'design': [1.0, 1.0]}

    # Load from files if they exist
    try:
        # Read net delay static pocv file
        pocv_files = glob.glob(os.path.join(folder_path, '*all.rpt'))
        if pocv_files:
            with open(pocv_files[0], 'r') as f:
                for line in f:
                    if 'Net delay static' in line:
                        parts = line.strip().split()
                        net_scaling['net'] = [float(parts[-3]), float(parts[-1])]  # rise, fall
                        break

        # Read guardband pocv file
        pocv_files = glob.glob(os.path.join(folder_path, '*pocv.rpt'))
        if pocv_files:
            guardband = read_pocv_rpt(pocv_files[0])

        # Read standard deviation pocv file
        std_files = glob.glob(os.path.join(folder_path, '*std.rpt'))
        if std_files:
            std_coef = read_pocv_rpt(std_files[0])

        if 'design' not in std_coef:
            std_coef['design'] = [1.0, 1.0]
        if 'design' not in guardband:
            guardband['design'] = [1.0, 1.0]

        # Save results
        save_pickle(net_scaling, os.path.join(save_dir, "net_2_pocvScaling.pkl"))
        save_pickle(guardband, os.path.join(save_dir, "libCell_2_riseFallguardband.pkl"))
        save_pickle(std_coef, os.path.join(save_dir, "libCell_2_riseFallStd.pkl"))

        return net_scaling, guardband, std_coef, True
    except Exception as e:
        print(f"Error reading POCVM files: {e}")
        return net_scaling, guardband, std_coef, False

