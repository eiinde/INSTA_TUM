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
# @file io/serialization.py
# @brief Serialization utilities for saving/loading data


import os
import pickle
import torch
from typing import Any, Optional, Union


def save_pickle(data: Any, file_path: str) -> None:
    """
    Save data to a pickle file with proper directory creation

    Args:
        data: Any Python object to save
        file_path: Path to the output file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path: str, default: Any = None) -> Any:
    """
    Load data from a pickle file with error handling

    Args:
        file_path: Path to the input file
        default: Value to return if file doesn't exist or has errors

    Returns:
        Loaded object or default value if loading fails
    """
    if not os.path.exists(file_path):
        return default

    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except (pickle.PickleError, EOFError, ImportError) as e:
        print(f"Error loading pickle from {file_path}: {e}")
        return default


def save_torch_tensor(tensor: torch.Tensor, file_path: str) -> None:
    """
    Save a PyTorch tensor to disk

    Args:
        tensor: PyTorch tensor to save
        file_path: Path to the output file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(tensor, file_path)


def load_torch_tensor(
    file_path: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> Optional[torch.Tensor]:
    """
    Load a PyTorch tensor from disk with optional device/dtype conversion

    Args:
        file_path: Path to the tensor file
        device: Optional device to load tensor to
        dtype: Optional dtype to convert tensor to

    Returns:
        Loaded tensor or None if loading fails
    """
    if not os.path.exists(file_path):
        return None

    try:
        tensor = torch.load(file_path, map_location='cpu')

        if device is not None:
            tensor = tensor.to(device)

        if dtype is not None:
            tensor = tensor.to(dtype)

        return tensor
    except Exception as e:
        print(f"Error loading tensor from {file_path}: {e}")
        return None


def check_cache_exists(cache_files: Union[str, list]) -> bool:
    """
    Check if cache files exist

    Args:
        cache_files: Single file path or list of file paths to check

    Returns:
        True if all files exist, False otherwise
    """
    if isinstance(cache_files, str):
        return os.path.exists(cache_files)

    return all(os.path.exists(f) for f in cache_files)
