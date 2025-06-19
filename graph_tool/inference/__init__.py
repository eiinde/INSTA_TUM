#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2024 Tiago de Paula Peixoto <tiago@skewed.de>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


"""
``graph_tool.inference``
------------------------

This module contains algorithms for the identification of large-scale network
structure via the statistical inference of generative models.

.. note::

   An introduction to the concepts used here, as well as a basic HOWTO is
   included in the cookbook section: :ref:`inference-howto`.

Nonparametric stochastic block model inference
++++++++++++++++++++++++++++++++++++++++++++++

High-level functions
====================

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   minimize_blockmodel_dl
   minimize_nested_blockmodel_dl

State classes
=============

.. currentmodule:: graph_tool.inference

.. autosummary::
   :nosignatures:
   :toctree: autosummary
   :template: class.rst

   BlockState
   OverlapBlockState
   LayeredBlockState
   NestedBlockState
   PPBlockState
   RankedBlockState
   ModularityState
   NormCutState
   TemperingState
   CliqueState

Abstract base classes
=====================

.. currentmodule:: graph_tool.inference

.. autosummary::
   :nosignatures:
   :toctree: autosummary
   :template: class.rst

   EntropyState
   MCMCState
   MultiflipMCMCState
   MultilevelMCMCState
   GibbsMCMCState
   MulticanonicalMCMCState
   ExhaustiveSweepState
   DrawBlockState

Sampling and minimization
=========================

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   mcmc_equilibrate
   mcmc_anneal
   multicanonical_equilibrate

.. autosummary::
   :nosignatures:
   :toctree: autosummary
   :template: class.rst

   MulticanonicalState

Comparing and manipulating partitions
=====================================

.. autosummary::
   :nosignatures:
   :toctree: autosummary
   :template: class.rst

   PartitionModeState
   ModeClusterState
   PartitionCentroidState

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   partition_overlap
   nested_partition_overlap
   variation_information
   mutual_information
   reduced_mutual_information
   contingency_graph
   shuffle_partition_labels
   order_partition_labels
   order_nested_partition_labels
   align_partition_labels
   align_nested_partition_labels
   partition_overlap_center
   nested_partition_overlap_center
   nested_partition_clear_null
   contiguous_map
   nested_contiguous_map

Auxiliary functions
===================

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   mf_entropy
   bethe_entropy
   microstate_entropy
   marginal_multigraph_entropy
   half_edge_graph
   get_block_edge_gradient

Auxiliary classes
=================

.. autosummary::
   :nosignatures:
   :toctree: autosummary
   :template: class-no-inherit.rst

   PartitionHist
   BlockPairHist

Nonparametric network reconstruction
++++++++++++++++++++++++++++++++++++

Reconstruction from direct measurements
=======================================

.. autosummary::
   :nosignatures:
   :toctree: autosummary
   :template: class.rst

   LatentLayerBaseState
   LatentMultigraphBlockState
   LatentClosureBlockState
   MeasuredBlockState
   MeasuredClosureBlockState
   MixedMeasuredBlockState
   UncertainBlockState
   UncertainBaseState

Expectation-maximization inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   latent_multigraph

Reconstruction from dynamics and behavior
=========================================

.. autosummary::
   :nosignatures:
   :toctree: autosummary
   :template: class.rst

   DynamicsBlockStateBase
   BPBlockStateBase
   PseudoNormalBlockState
   NormalGlauberBlockState
   LinearNormalBlockState
   LVBlockState
   EpidemicsBlockState
   IsingBlockStateBase
   IsingGlauberBlockState
   CIsingGlauberBlockState
   PseudoIsingBlockState
   PseudoCIsingBlockState


Semiparametric stochastic block model inference
+++++++++++++++++++++++++++++++++++++++++++++++

State classes
=============

.. autosummary::
   :nosignatures:
   :toctree: autosummary
   :template: class.rst

   EMBlockState

Expectation-maximization Inference
==================================

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   em_infer

Large-scale descriptors
+++++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   modularity

"""

__all__ = ["minimize_blockmodel_dl",
           "minimize_nested_blockmodel_dl",
           "BlockState",
           "OverlapBlockState",
           "LayeredBlockState",
           "NestedBlockState",
           "PPBlockState",
           "RankedBlockState",
           "PartitionCentroidState",
           "PartitionModeState",
           "ModeClusterState",
           "ModularityState",
           "NormCutState",
           "LatentMultigraphBlockState",
           "UncertainBlockState",
           "MeasuredBlockState",
           "UncertainBaseState",
           "MixedMeasuredBlockState",
           "DynamicsBlockStateBase",
           "EpidemicsBlockState",
           "BPBlockStateBase",
           "IsingBlockStateBase",
           "IsingGlauberBlockState",
           "PseudoIsingBlockState",
           "CIsingGlauberBlockState",
           "PseudoCIsingBlockState",
           "PseudoNormalBlockState",
           "LinearNormalBlockState",
           "NormalGlauberBlockState",
           "LVBlockState",
           "LatentLayerBaseState",
           "LatentClosureBlockState",
           "MeasuredClosureBlockState",
           "HistState",
           "CliqueState",
           "EntropyState",
           "MCMCState",
           "MultiflipMCMCState",
           "MultilevelMCMCState",
           "GibbsMCMCState",
           "MulticanonicalMCMCState",
           "ExhaustiveSweepState",
           "DrawBlockState",
           "mcmc_equilibrate",
           "mcmc_anneal",
           "TemperingState",
           "multicanonical_equilibrate",
           "MulticanonicalState",
           "EMBlockState",
           "em_infer",
           "mf_entropy",
           "bethe_entropy",
           "microstate_entropy",
           "marginal_multigraph_entropy",
           "marginal_multigraph_sample",
           "marginal_graph_sample",
           "marginal_multigraph_lprob",
           "marginal_graph_lprob",
           "PartitionHist",
           "BlockPairHist",
           "half_edge_graph",
           "get_block_edge_gradient",
           "get_hierarchy_tree",
           "modularity",
           "latent_multigraph",
           "partition_overlap",
           "nested_partition_overlap",
           "contingency_graph",
           "contiguous_map",
           "nested_contiguous_map",
           "align_partition_labels",
           "align_nested_partition_labels",
           "shuffle_nested_partition_labels",
           "shuffle_partition_labels",
           "order_partition_labels",
           "order_nested_partition_labels",
           "partition_overlap_center",
           "nested_partition_overlap_center",
           "nested_partition_clear_null",
           "variation_information",
           "mutual_information",
           "reduced_mutual_information"]

from . base_states import *
from . blockmodel import *
from . overlap_blockmodel import *
from . layered_blockmodel import *
from . nested_blockmodel import *
from . uncertain_blockmodel import *
from . reconstruction import *
from . mcmc import *
from . minimize import *
from . blockmodel_em import *
from . util import *
from . modularity import *
from . norm_cut import *
from . latent_multigraph import *
from . partition_centroid import *
from . partition_modes import *
from . planted_partition import *
from . ranked import *
from . latent_layers import *
from . histogram import *
from . clique_decomposition import *

libinference.init_cache()
