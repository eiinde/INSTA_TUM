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

from .. import Graph, GraphView, _get_rng, _prop, PropertyMap, \
    perfect_prop_hash, Vector_size_t, group_vector_property
from . blockmodel import DictState
from . util import *
import numpy as np

from . base_states import *

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

@entropy_state_signature
class NormCutState(MCMCState, MultiflipMCMCState, MultilevelMCMCState,
                   GibbsMCMCState, DrawBlockState):
    r"""Obtain the partition of a network according to the normalized cut.

    .. warning::

       Do not use this approach in the analysis of networks without
       understanding the consequences. This algorithm is included only for
       comparison purposes. In general, the inference-based approaches based on
       :class:`~graph_tool.inference.BlockState`,
       :class:`~graph_tool.inference.NestedBlockState`, and
       :class:`~graph_tool.inference.PPBlockState` should be
       universally preferred.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be partitioned.
    b : :class:`~graph_tool.PropertyMap` (optional, default: ``None``)
        Initial partition. If not supplied, a partition into a single group will
        be used.

    """

    def __init__(self, g, b=None):
        EntropyState.__init__(self)
        self.g = GraphView(g, directed=False)
        if b is None:
            self.b = self.g.new_vp("int32_t")
        elif isinstance(b, PropertyMap):
            self.b = b.copy("int32_t")
        else:
            self.b = self.g.new_vp("int32_t", vals=b)

        self.er = Vector_size_t()
        self.err = Vector_size_t()

        self.bg = self.g
        self._abg = self.bg._get_any()
        self._state = libinference.make_norm_cut_state(self)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        g = copy.deepcopy(self.g, memo)
        b = copy.deepcopy(self.b, memo)
        return self.copy(g=g, b=b)

    def copy(self, g=None, b=None):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""
        return NormCutState(g=g if g is not None else self.g,
                            b=b if b is not None else self.b)

    def __getstate__(self):
        return dict(g=self.g, b=self.b)

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        return "<NormCutState object with %d blocks, for graph %s, at 0x%x>" % \
            (self.get_B(), str(self.g), id(self))

    def _gen_eargs(self, args):
        return libinference.modularity_entropy_args()

    def get_B(self):
        r"Returns the total number of blocks."
        return len(np.unique(self.b.fa))

    def get_Be(self):
        r"""Returns the effective number of blocks, defined as :math:`e^{H}`, with
        :math:`H=-\sum_r\frac{n_r}{N}\ln \frac{n_r}{N}`, where :math:`n_r` is
        the number of nodes in group r.
        """
        w = np.array(np.bincount(self.b.fa), dtype="double")
        w = w[w>0]
        w /= w.sum()
        return np.exp(-(w*log(w)).sum())

    @copy_state_wrap
    def _entropy(self):
        r"""Return the normalized cut. See :meth:`~NormCutState.norm_cut`."""

        eargs = self._get_entropy_args(locals(), ignore=["self", "kwargs"])

        return self._state.entropy(eargs)

    def _get_entropy_args(self, kwargs, ignore=None):
        kwargs = dict(self._entropy_args, **kwargs)
        if ignore is not None:
            for a in ignore:
                kwargs.pop(a, None)
        ea = libinference.norm_cut_entropy_args()
        if len(kwargs) > 0:
            raise ValueError("unrecognized entropy arguments: " +
                             str(list(kwargs.keys())))
        return ea

    def norm_cut(self):
        r"""Return the normalized cut.

        Notes
        -----

        The normalized cut is defined as

        .. math::

           B - \sum_r \frac{e_{rr}}{e_r}

        Where :math:`B` is the number of groups, :math:`e_{rr}` is twice
        the number of edges between nodes of group :math:`r`, and :math:`e_r` is
        the sum of degrees of nodes in group :math:`r`.

        """

        return self.entropy()

    def _mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.norm_cut_mcmc_sweep(mcmc_state, self._state,
                                                _get_rng())

    def _multiflip_mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.norm_cut_multiflip_mcmc_sweep(mcmc_state,
                                                          self._state,
                                                          _get_rng())

    def _multilevel_mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.norm_cut_multilevel_mcmc_sweep(mcmc_state,
                                                           self._state,
                                                           _get_rng())
    def _gibbs_sweep_dispatch(self, gibbs_state):
        return libinference.norm_cut_gibbs_sweep(gibbs_state, self._state,
                                                 _get_rng())
