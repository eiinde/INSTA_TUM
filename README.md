# INSTA

A GPU-accelerated, Differentiable Static Timing Analysis Engine for Industrial Physical Design Applications

## License

This software is released under the NVIDIA Source Code License.
See [LICENSE.txt](LICENSE.txt) for full terms.

## [Note] Key Requirements for Pre-Compiled Kernels
- Ubuntu 20.04
- CUDA 11.8

# Installation

## by docker (recommended)
'''
cd <INSTA directory>
docker build -t insta_image .
'''

## by conda/mamba
```
bash Mambaforge.sh -b -p "./conda"
source conda/etc/profile.d/conda.sh
source conda/etc/profile.d/mamba.sh
mamba env create -f insta_env.yaml
mamba activate insta
```

# Using INSTA

## Initialization
```
from src.core.insta import INSTA
insta = INSTA()
insta.do_set_insta_path('testcase/aes_cipher_top/', 'aes')
insta.do_initialization()
```

## Differentiable Timing Propgation & Gradient Extraction
```
insta.do_diff_propagation(plot=True)
(-insta.tns).backward()
insta.do_extract_arc_grads()
```
you can obtain "insta.netArc_2_riseFallGrads" and "insta.cellArc_2_riseFallGrads"

## Top-K arrival propagation (K is fixed to 256)
```
insta.do_eval_propagation(plot=True)
```

