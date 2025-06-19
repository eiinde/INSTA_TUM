"""
Global constants for INSTA timing analysis
"""

import torch

# Default computation parameters
DEFAULT_DEVICE = torch.device('cuda:0')
DEFAULT_FLOAT_DTYPE = torch.float32
DEFAULT_TOPK = 256
DEFAULT_SCALING = 1.0

# File paths
DEFAULT_INPUT_FOLDER = 'insta_inputs'
DEFAULT_OUTPUT_FOLDER = 'outputs'

# Standard library paths
DEFAULT_LIB_PATH = '/raid/yilu/lagrange/libraries'
NESTED_LIB_DICT_PATH = f"{DEFAULT_LIB_PATH}/asap7_nested_lib_dict.pkl"
NESTED_PIN_DICT_PATH = f"{DEFAULT_LIB_PATH}/asap7_nested_pin_dict.pkl"

# Valid timing arc senses
VALID_SENSE_TYPES = [
    'positive_unate',
    'negative_unate',
    'rising_edge',
    'falling_edge'
]

# Default statistical parameters
DEFAULT_SIGMA = 3.0
DEFAULT_TEMPERATURE = 10.0

# POCV defaults
DEFAULT_GUARDBAND = [1.0, 1.0]
DEFAULT_STD_COEF = [1.0, 1.0]
DEFAULT_SCALING_FACTOR = [1.0, 1.0]
