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

def modularity(g, b, gamma=1., weight=None):
    r"""
    Calculate Newman's (generalized) modularity of a network partition.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    b : :class:`~graph_tool.VertexPropertyMap`
        Vertex property map with the community partition.
    gamma : ``float`` (optional, default: ``1.``)
        Resolution parameter.
    weight : :class:`~graph_tool.EdgePropertyMap` (optional, default: None)
        Edge property map with the optional edge weights.

    Returns
    -------
    Q : float
        Newman's modularity.

    Notes
    -----

    Given a specific graph partition specified by `prop`, Newman's modularity
    [newman-modularity-2006]_ is defined as:

    .. math::

          Q = \frac{1}{2E} \sum_r e_{rr}- \frac{e_r^2}{2E}

    where :math:`e_{rs}` is the number of edges which fall between
    vertices in communities s and r, or twice that number if :math:`r = s`, and
    :math:`e_r = \sum_s e_{rs}`.

    If weights are provided, the matrix :math:`e_{rs}` corresponds to the sum
    of edge weights instead of number of edges, and the value of :math:`E`
    becomes the total sum of edge weights.

    Examples
    --------
    >>> g = gt.collection.data["football"]
    >>> gt.modularity(g, g.vp.value_tsevans)
    0.5744393497...

    References
    ----------
    .. [newman-modularity-2006] M. E. J. Newman, "Modularity and community
       structure in networks", Proc. Natl. Acad. Sci. USA 103, 8577-8582 (2006),
       :doi:`10.1073/pnas.0601602103`, :arxiv:`physics/0602124`
    """

    if b.value_type() not in ["bool", "int16_t", "int32_t", "int64_t",
                              "unsigned long"]:
        b = perfect_prop_hash([b])[0]
    Q = libinference.modularity(g._Graph__graph, gamma,
                                _prop("e", g, weight),
                                _prop("v", g, b))
    return Q


@entropy_state_signature
class ModularityState(MCMCState, MultiflipMCMCState, MultilevelMCMCState,
                      GibbsMCMCState, DrawBlockState):
    r"""Obtain the partition of a network according to the maximization of Newman's modularity.

    .. danger:: Using modularity maximization is almost always **a terrible idea**.

       Modularity maximization is a substantially inferior method to the
       inference-based ones that are implemented in ``graph-tool``, since it
       does not possess any kind of statistical regularization. Among many other
       problems, the method tends to massively overfit empirical data.

       For a more detailed explanation see `“Modularity maximization
       considered harmful”
       <https://skewed.de/tiago/blog/modularity-harmful>`_, as well as
       [peixoto-descriptive-2023]_.

       Do not use this approach in the analysis of networks without
       understanding the consequences. This algorithm is included only for
       comparison purposes. In general, the inference-based approaches based
       on :class:`~graph_tool.inference.BlockState`,
       :class:`~graph_tool.inference.NestedBlockState`, and
       :class:`~graph_tool.inference.PPBlockState` should be universally
       preferred.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be partitioned.
    b : :class:`~graph_tool.PropertyMap` (optional, default: ``None``)
        Initial partition. If not supplied, a partition into a single group will
        be used.
    eweight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge multiplicities (for multigraphs).
    entropy_args: ``dict`` (optional, default: ``{}``)
        Override default arguments for :meth:`~ModularityState.entropy()`
        method and releated operations.
    """

    def __init__(self, g, b=None, eweight=None, entropy_args={}):
        EntropyState.__init__(self, entropy_args=entropy_args)
        self.g = GraphView(g, directed=False)
        if b is None:
            self.b = self.g.new_vp("int32_t")
        elif isinstance(b, PropertyMap):
            self.b = self.g.own_property(b).copy("int32_t")
        else:
            self.b = self.g.new_vp("int32_t", vals=b)

        if eweight is None:
            eweight = g.new_ep("int", 1)
        else:
            eweight = g.own_property(eweight)
        self.eweight = eweight

        self.bg = self.g.base
        if self.bg.is_directed():
            self.bg = GraphView(self.bg, directed=False)
        self._abg = self.bg._get_any()
        self._state = libinference.make_modularity_state(self)

    def __copy__(self):
        return self.copy()

    def copy(self, g=None, b=None):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""
        return ModularityState(g=g if g is not None else self.g,
                               b=b if b is not None else self.b)

    def __getstate__(self):
        state = EntropyState.__getstate__(self)
        return dict(state, g=self.g, b=self.b)

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        return "<ModularityState object with %d blocks, for graph %s, at 0x%x>" % \
            (self.get_B(), str(self.g), id(self))

    def _gen_eargs(self, args):
        return libinference.modularity_entropy_args()

    def get_B(self):
        r"Returns the total number of blocks."
        rs = np.unique(self.b.fa)
        return len(rs)

    def get_Be(self):
        r"""Returns the effective number of blocks, defined as :math:`e^{H}`, with
        :math:`H=-\sum_r\frac{n_r}{N}\ln \frac{n_r}{N}`, where :math:`n_r` is
        the number of nodes in group r.
        """
        w = np.bincount(self.b.fa)
        w = np.array(w, dtype="double")
        w = w[w>0]
        w /= w.sum()
        return np.exp(-(w*log(w)).sum())

    @copy_state_wrap
    def _entropy(self, gamma=1.):
        r"""Return the unnormalized negative generalized modularity.

        Notes
        -----

        The unnormalized negative generalized modularity is defined as

        .. math::

           -\sum_{ij}\left(A_{ij}-\gamma \frac{k_ik_j}{2E}\right)

        Where :math:`A_{ij}` is the adjacency matrix, :math:`k_i` is the degree
        of node :math:`i`, and :math:`E` is the total number of edges.

        """

        eargs = self._get_entropy_args(locals(), ignore=["self", "kwargs"])

        return self._state.entropy(eargs)

    def _get_entropy_args(self, kwargs, ignore=None):
        kwargs = dict(self._entropy_args, **kwargs)
        if ignore is not None:
            for a in ignore:
                kwargs.pop(a, None)
        ea = libinference.modularity_entropy_args()
        ea.gamma = kwargs["gamma"]
        del kwargs["gamma"]
        if len(kwargs) > 0:
            raise ValueError("unrecognized entropy arguments: " +
                             str(list(kwargs.keys())))
        return ea

    def modularity(self, gamma=1):
        r"""Return the generalized modularity.

        Notes
        -----

        The generalized modularity is defined as

        .. math::

           \frac{1}{2E}\sum_{ij}\left(A_{ij}-\gamma \frac{k_ik_j}{2E}\right)

        Where :math:`A_{ij}` is the adjacency matrix, :math:`k_i` is the degree
        of node :math:`i`, and :math:`E` is the total number of edges.

        """

        Q = self.entropy(gamma=gamma)
        return -Q / (2 * self.g.num_edges())

    def _mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.modularity_mcmc_sweep(mcmc_state, self._state,
                                                  _get_rng())

    def _multiflip_mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.modularity_multiflip_mcmc_sweep(mcmc_state,
                                                            self._state,
                                                            _get_rng())

    def _multilevel_mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.modularity_multilevel_mcmc_sweep(mcmc_state,
                                                             self._state,
                                                             _get_rng())
    def _gibbs_sweep_dispatch(self, gibbs_state):
        return libinference.modularity_gibbs_sweep(gibbs_state, self._state,
                                                   _get_rng())

