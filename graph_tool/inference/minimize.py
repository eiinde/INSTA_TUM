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

import numpy
from . util import *
from . blockmodel import *
from . nested_blockmodel import *

def minimize_blockmodel_dl(g, state=BlockState, state_args={},
                           multilevel_mcmc_args={}):
    r"""Fit the stochastic block model, by minimizing its description length using an
    agglomerative heuristic.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        The graph.
    state : SBM-like state class (optional, default: :class:`~graph_tool.inference.BlockState`)
        Type of model that will be used. Must be derived from :class:`~graph_tool.inference.MultilevelMCMCState`.
    state_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to appropriate state constructor (e.g.
        :class:`~graph_tool.inference.BlockState`)
    multilevel_mcmc_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to :meth:`~graph_tool.inference.MultilevelMCMCState.multilevel_mcmc_sweep`.

    Returns
    -------
    min_state : type given by parameter ``state``
        State with minimum description length.

    Notes
    -----

    This function is a convenience wrapper around
    :meth:`~graph_tool.inference.MultilevelMCMCState.multilevel_mcmc_sweep`.

    See [peixoto-efficient-2014]_ for details on the algorithm.

    This algorithm has a complexity of :math:`O(V \ln^2 V)`, where :math:`V` is
    the number of nodes in the network.

    Examples
    --------

    .. testsetup:: mdl

       gt.seed_rng(43)
       np.random.seed(43)

    .. doctest:: mdl

       >>> g = gt.collection.data["polbooks"]
       >>> state = gt.minimize_blockmodel_dl(g)
       >>> state.draw(pos=g.vp["pos"], vertex_shape=state.get_blocks(),
       ...            output="polbooks_blocks_mdl.svg")
       <...>

    .. figure:: polbooks_blocks_mdl.*
       :align: center

       Block partition of a political books network, which minimizes the
       description length of the network according to the degree-corrected
       stochastic blockmodel.


    .. testsetup:: mdl_overlap

       gt.seed_rng(42)
       np.random.seed(42)

    .. doctest:: mdl_overlap

       >>> g = gt.collection.data["polbooks"]
       >>> state = gt.minimize_blockmodel_dl(g, state=gt.OverlapBlockState)
       >>> state.draw(pos=g.vp["pos"], output="polbooks_overlap_blocks_mdl.svg")
       <...>

    .. figure:: polbooks_overlap_blocks_mdl.*
       :align: center

       Overlapping partition of a political books network, which minimizes the
       description length of the network according to the overlapping
       degree-corrected stochastic blockmodel.

    .. doctest:: mdl_pp

       >>> g = gt.collection.data["celegansneural"]
       >>> state = gt.minimize_blockmodel_dl(g, state=gt.PPBlockState)
       >>> state.draw(output="celegans_mdl_pp.pdf")
       <...>

    .. testcleanup:: mdl_pp

       conv_png("celegans_mdl_pp.pdf")

    .. figure:: celegans_mdl_pp.png
       :align: center
       :width: 60%

       Assortative partition of the *C. elegans* neural network, which minimizes
       the description length of the network according to the degree-corrected
       planted-partition blockmodel.

    References
    ----------
    .. [peixoto-efficient-2014] Tiago P. Peixoto, "Efficient Monte Carlo and greedy
       heuristic for the inference of stochastic block models", Phys. Rev. E 89,
       012804 (2014), :doi:`10.1103/PhysRevE.89.012804`, :arxiv:`1310.4378`.
    """

    state = state(g, **state_args)

    args = dict(niter=1, beta=numpy.inf, force_accept=True)
    args.update(multilevel_mcmc_args)

    state.multilevel_mcmc_sweep(**args)

    return state

def minimize_nested_blockmodel_dl(g, init_bs=None,
                                  state=NestedBlockState, state_args={},
                                  multilevel_mcmc_args={}):
    r"""Fit the nested stochastic block model, by minimizing its description length
    using an agglomerative heuristic.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        The graph.
    init_bs : iterable of iterable of ``int``s (optional, default: ``None``)
        Initial hierarchical partition.
    state : SBM state class (optional, default: :class:`~graph_tool.inference.NestedBlockState`)
        Type of model that will be used.
    state_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to appropriate state constructor (e.g.
        :class:`~graph_tool.inference.BlockState`)
    multilevel_mcmc_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to :meth:`~graph_tool.inference.MultilevelMCMCState.multilevel_mcmc_sweep`.

    Returns
    -------
    min_state : type given by parameter ``state``
        State with minimum description length.

    Notes
    -----
    This function is a convenience wrapper around
    :meth:`~graph_tool.inference.NestedBlockState.multilevel_mcmc_sweep`.

    See [peixoto-hierarchical-2014]_ for details on the algorithm.

    This algorithm has a complexity of :math:`O(E \ln^2 V)`, where :math:`E` and
    :math:`V` are the number of edges and nodes in the network, respectively.

    Examples
    --------
    .. testsetup:: nested_mdl

       gt.seed_rng(43)
       np.random.seed(43)

    .. doctest:: nested_mdl

       >>> g = gt.collection.data["power"]
       >>> state = gt.minimize_nested_blockmodel_dl(g)
       >>> state.draw(output="power_nested_mdl.pdf")
       (...)

    .. testcleanup:: nested_mdl

       conv_png("power_nested_mdl.pdf")

    .. figure:: power_nested_mdl.png
       :align: center
       :width: 60%

       Hierarchical Block partition of a power-grid network, which minimizes
       the description length of the network according to the nested
       (degree-corrected) stochastic blockmodel.


    .. doctest:: nested_mdl_overlap

       >>> g = gt.collection.data["celegansneural"]
       >>> state = gt.minimize_nested_blockmodel_dl(g, state_args=dict(overlap=True))
       >>> state.draw(output="celegans_nested_mdl_overlap.pdf")
       (...)

    .. testcleanup:: nested_mdl_overlap

       conv_png("celegans_nested_mdl_overlap.pdf")

    .. figure:: celegans_nested_mdl_overlap.png
       :align: center
       :width: 60%

       Overlapping block partition of the *C. elegans* neural network, which
       minimizes the description length of the network according to the nested
       overlapping degree-corrected stochastic blockmodel.

    References
    ----------
    .. [peixoto-hierarchical-2014] Tiago P. Peixoto, "Hierarchical block
       structures and high-resolution model selection in large networks ",
       Phys. Rev. X 4, 011047 (2014), :doi:`10.1103/PhysRevX.4.011047`,
       :arxiv:`1310.4377`.

    """

    state = state(g, bs=init_bs, **state_args)

    args = dict(niter=1, beta=numpy.inf)
    args.update(multilevel_mcmc_args)

    l = 0
    while l >= 0:

        ret = state.multilevel_mcmc_sweep(ls=[l], **args)

        if args.get("verbose", False):
            print(l, ret, state)

        if abs(ret[0]) < 1e-8:
            l -= 1
        else:
            l = min((l + 1, len(state.levels) - 1))

    return state
