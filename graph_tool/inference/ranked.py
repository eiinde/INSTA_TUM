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
    edge_endpoint_property
from . blockmodel import BlockState
from . base_states import *
from . util import *

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

import numpy as np
import math
from scipy.stats import rankdata

@entropy_state_signature
class RankedBlockState(MCMCState, MultiflipMCMCState, MultilevelMCMCState,
                       GibbsMCMCState, DrawBlockState):
    r"""Obtain the ordered partition of a network according to the ranked
    stochastic block model.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modelled.
    b : :class:`~graph_tool.PropertyMap` (optional, default: ``None``)
        Initial partition. If not supplied, a partition into a single group will
        be used.
    u : :class:`~graph_tool.PropertyMap` or iterable (optional, default: ``None``)
        Ordering of the group labels. It should contain a map from each group
        label to the unit interval :math:`[0,1]`, inidicating how the groups
        should be ordered. If not supplied, the numeric values of the group
        lalbels will be used to initialize the ordering.
    entropy_args: ``dict`` (optional, default: ``{}``)
        Override default arguments for :meth:`~RankedBlockState.entropy()`
        method and releated operations.

    References
    ----------
    .. [peixoto-ordered-2022] Tiago P. Peixoto, "Ordered community detection in
       directed networks", Phys. Rev. E 106, 024305 (2022),
       :doi:`10.1103/PhysRevE.106.024305`, :arxiv:`2203.16460`

    """

    def __init__(self, g, b=None, u=None, entropy_args={}, **kwargs):
        EntropyState.__init__(self, entropy_args=entropy_args)
        self.g = g
        self.ustate = BlockState(self.g, b=b, **kwargs)
        self.b = self.ustate.b

        if u is None:
            u = self.ustate.bg.new_vp("double")
            u.fa = np.linspace(0.1, .9, len(u.fa))

        if isinstance(u, PropertyMap):
            u = self.ustate.bg.own_property(u)
        else:
            u = self.ustate.bg.new_vp("double", vals=u)

        self.u = u

        self._state = libinference.make_ranked_state(self.ustate._state,
                                                     self)

        self.eweight = self.ustate.eweight
        self.vweight = self.ustate.vweight
        self.is_weighted = True
        self.overlap = False

    def __copy__(self):
        return self.copy()

    def copy(self, **kwargs):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""
        args = self.__getstate__()
        args.update(**kwargs)
        return RankedBlockState(**args)

    def __getstate__(self):
        state = EntropyState.__getstate__(self)
        state = dict(state, **self.ustate.__getstate__())
        return dict(state, u=self.u.a.copy())

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        return "<RankedBlockState object with %d blocks, %d upstream, %d downstream, and %d lateral edges,%s for graph %s, at 0x%x>" % \
            (self.get_B(), self.get_Es()[0], self.get_Es()[2],
             self.get_Es()[1],
             " degree-corrected," if self.ustate.deg_corr else "",
             str(self.g), id(self))

    def _couple_state(self, state, entropy_args):
        if state is None:
            self._coupled_state = None
            self._state.decouple_state()
        else:
            self._coupled_state = (state, entropy_args)
            eargs = self._get_entropy_args(entropy_args)
            self._state.couple_state(state._state, eargs)

    def get_block_state(self, b=None, vweight=None, deg_corr=False, **kwargs):
        r"""Returns a :class:`~graph_tool.inference.BlockState` corresponding to
        the block graph (i.e. the blocks of the current state become the nodes).
        """

        bstate = self.ustate.get_block_state(b=b, vweight=vweight,
                                             deg_corr=deg_corr,
                                             **kwargs)
        bg = GraphView(bstate.g, directed=False)
        return bstate.copy(g=bg)

    def get_blocks(self):
        r"""Returns the property map which contains the block labels for each vertex."""
        return self.b

    def get_state(self):
        """Alias to :meth:`~RankedBlockState.get_blocks`."""
        return self.g.own_property(self.ustate.get_blocks())

    def get_block_order(self):
        """Returns an array indexed by the group label containing its rank order."""
        idx = self.ustate.wr.fa == 0
        u = self.u.fa.copy()
        u[idx] = 1
        return rankdata(u, method='ordinal') - 1

    def get_vertex_order(self):
        """Returns a vertex :class:`~graph_tool.PropertyMap` with the rank order
        for every vertex."""
        u = self.b.copy()
        pmap(u, self.get_block_order())
        return u

    def get_vertex_position(self):
        """Returns a vertex :class:`~graph_tool.PropertyMap` with vertex
        positions in the unit interval :math:`[0,1]`."""
        u = self.get_vertex_order()
        u = u.copy("double")
        u.fa /= max((u.fa.max(), 1))
        return u

    def collect_vertex_marginals(self, p=None, b=None, update=1):
        r"""Collect the vertex marginal histogram, which counts the number of times a
        node was assigned to a given block.

        Parameters
        ----------
        p : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            Vertex property map with vector-type values, storing the previous block
            membership counts. If not provided, an empty histogram will be created.
        b : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            Vertex property map with group partition. If not provided, the
            state's partition will be used.
        update : int (optional, default: ``1``)
            Each call increases the current count by the amount given by this
            parameter.

        Returns
        -------
        p : :class:`~graph_tool.VertexPropertyMap`
            Vertex property map with vector-type values, storing the accumulated
            block membership counts.
        """

        if p is None:
            p = self.g.new_vp("vector<int>")
        if b is None:
            b = self.get_vertex_order()
        libinference.vertex_marginals(self.g._Graph__graph,
                                      _prop("v", self.g, b),
                                      _prop("v", self.g, p),
                                      update)
        return p

    def get_edge_dir(self):
        """Return an edge :class:`~graph_tool.PropertyMap` containing the edge
        direction: ``-1`` (downstream), ``0`` (lateral), ``+1`` (upstream).
        """

        u = self.b.copy("double")
        pmap(u, self.u)
        u_s = edge_endpoint_property(self.g, u, "source")
        u_t = edge_endpoint_property(self.g, u, "target")
        edir = self.g.new_ep("int")
        edir.a = u_s.a < u_t.a
        idx = edir.a == 0
        edir.a[idx] = (u_s.a > u_t.a)[idx]
        edir.a[idx] *= -1
        return edir

    def get_N(self):
        """Return the number of nodes."""
        return self.ustate.get_N()

    def get_E(self):
        """Return the number of edges."""
        return self.ustate.get_E()

    def get_Es(self):
        """Return the number of dowstream, lateral, and upstream edges."""
        return self._state.get_Es()

    def get_B(self):
        r"Returns the total number of blocks."
        return self.ustate.get_nonempty_B()

    def get_nonempty_B(self):
        r"Alias to :meth:`~RankedBlockState.get_B`."
        return self.get_B()

    def get_Be(self):
        r"""Returns the effective number of blocks, defined as :math:`e^{H}`, with
        :math:`H=-\sum_r\frac{n_r}{N}\ln \frac{n_r}{N}`, where :math:`n_r` is
        the number of nodes in group r.
        """
        return self.ustate.get_Be()

    def virtual_vertex_move(self, v, s, **kwargs):
        r"""Computes the entropy difference if vertex ``v`` is moved to block ``s``. The
        remaining parameters are the same as in
        :meth:`~graph_tool.inference.RankedBlockState.entropy`."""
        return self._state.virtual_move(int(v), self.b[v], s,
                                        self._get_entropy_args(dict(self._entropy_args,
                                                                    **kwargs)))

    def move_vertex(self, v, s):
        r"""Move vertex ``v`` to block ``s``."""
        self._state.move_vertex(int(v), int(s))

    @copy_state_wrap
    def _entropy(self, adjacency=True, dl=True, partition_dl=True,
                 degree_dl=True, degree_dl_kind="distributed", edges_dl=True,
                 dense=False, multigraph=True, deg_entropy=True, recs=True,
                 recs_dl=True, beta_dl=1., Bfield=True, exact=True, **kwargs):
        r"""Return the description length (negative joint log-likelihood). See
        :meth:`BlockState.entropy` for documentation."""

        eargs = self._get_entropy_args(locals())

        S = self._state.entropy(eargs, False)

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        return S

    def multiflip_mcmc_sweep(self, pmovelabel=1, **kwargs):
        """Call :meth:`MultiflipMCMCState.multiflip_mcmc_sweep` with default
        parameter ``pmovelabel=1``.`"""
        return super().multiflip_mcmc_sweep(pmovelabel=pmovelabel, **kwargs)

    def _gen_eargs(self, args):
        return libinference.entropy_args()

    def _get_entropy_args(self, kwargs, consume=False):
        if not consume:
            kwargs = kwargs.copy()
        deg_dl_kind = kwargs.get("degree_dl_kind",
                                 self._entropy_args["degree_dl_kind"])
        if deg_dl_kind == "entropy":
            kind = libinference.deg_dl_kind.ent
        elif deg_dl_kind == "uniform":
            kind = libinference.deg_dl_kind.uniform
        elif deg_dl_kind == "distributed":
            kind = libinference.deg_dl_kind.dist
        kwargs["degree_dl_kind"] = kind
        dl = kwargs.get("dl", self._entropy_args["dl"])
        ea = super()._get_entropy_args(kwargs, consume)
        if not dl:
            ea.partition_dl = False
            ea.degree_dl = False
            ea.edges_dl = False
            ea.recs_dl = False
        return ea

    def sample_graph(self, **kwargs):
        r"""Sample a new graph from the fitted model. See :meth:`BlockState.sample_graph` for documentation."""

        return self.ustate.sample_graph(**kwargs)

    def _mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.ranked_mcmc_sweep(mcmc_state, self._state,
                                              _get_rng())

    def _multiflip_mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.ranked_multiflip_mcmc_sweep(mcmc_state,
                                                        self._state,
                                                        _get_rng())

    def _multilevel_mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.ranked_multilevel_mcmc_sweep(mcmc_state,
                                                         self._state,
                                                         _get_rng())

    def _gibbs_sweep_dispatch(self, gibbs_state):
        return libinference.ranked_gibbs_sweep(gibbs_state, self._state,
                                               _get_rng())

    def get_edge_colors(self):
        """Return :class:`~graph_tool.EdgePropertyMap` containing the edge
        colors according to their rank direction: upstream (blue), downstream
        (red), lateral (grey).
        """
        edir = self.get_edge_dir()
        ecolor = self.g.new_ep("vector<double>")
        for e in self.g.edges():
            if edir[e] == 0:
                ecolor[e] = (0.1, 0.1, 0.3, 0.6)
            elif edir[e] == 1:
                ecolor[e] = (0.2823529411764706, 0.47058823529411764, 0.8156862745098039, .6)
            else:
                ecolor[e] = (0.8392156862745098, 0.37254901960784315, 0.37254901960784315, .8)
        return ecolor

    def draw(self, **kwargs):
        r"""Convenience wrapper to :func:`~graph_tool.draw.graph_draw` that
        draws the state of the graph as colors on the vertices and edges."""

        edir = self.get_edge_dir()
        ecolor = self.get_edge_colors()

        Es = self.get_Es()
        if Es[-1] < Es[0]:
            edir.a *= -1
        edir.a[edir.a == 0] = 2

        return super().draw(**dict(dict(edge_gradient=[], edge_color=ecolor,
                                        eorder=edir), **kwargs))
