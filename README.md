# INSTA

A GPU-accelerated, Differentiable Static Timing Analysis Engine for Industrial Physical Design Applications

INSTA is the first-ever differentiable GPU-STA framework that achieves near-perfect endpoint slack correlation to an industry-standard commercial signoff tool with OCV support for advanced technology nodes. On a design with 15 millions pins, INSTA performs full-graph timing propagation in less than 0.1 seconds with 0.999 correlation to the reference signoff tool.

INSTA's GPU-accelerated Top-K statistical arrival propagation CUDA kernel efficiently manages CPPR, a must-handle timing pessimism in advanced technology nodes.

INSTA’s capability as a fast timing evaluator can be used in physical design flow directly, bringing 25x runtime improvement over reference tool’s incremental timing analysis.
It can also be used to create “timing gradients” to enable truly global, differentiable timing optimization for physical design tasks.

## License

This software is released under the NVIDIA Source Code License.
See [LICENSE.txt](LICENSE.txt) for full terms.

## [Note] Key Requirements for Pre-Compiled Kernels
- Ubuntu 20.04
- CUDA 11.8

# Installation

## by docker (recommended)
```
cd <INSTA directory>
docker build -t insta_image .
docker run --gpus all -it --rm   -v "$(pwd)":/workspace   -w /workspace   --name instadevn   insta_image:latest   bash
```

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

