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

from .. import Graph, GraphView, _get_rng, Vector_double, Vector_int64_t

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

from . util import *
from . base_states import *
from . blockmodel import lbinom

from .. topology import max_cliques

from scipy.special import gammaln
from itertools import combinations

class CliqueState(object):
    r"""The state of a clique decomposition of a given graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modelled.
    init_edges : ``bool`` (optional, default: ``False``)
        If ``True``, the state will be initialized with edges being
        occupied.
    init_max_cliques : ``bool`` (optional, default: ``True``)
        If ``True``, the state will be initialized with the maximal cliques.
    init_list : ``dict`` (optional, default: ``{}``)
        If given, this will give the initialization state. Keys are the clique
        nodes and values are the counts.

    Examples
    --------
    .. testsetup:: clique_decomposition

       gt.seed_rng(42)
       np.random.seed(42)

    .. doctest:: clique_decomposition

       >>> g = gt.collection.data["polbooks"]
       >>> state = gt.CliqueState(g)
       >>> state.mcmc_sweep(niter=10000)
       (...)
       >>> cliques = []
       >>> for v in state.f.vertices():      # iterate through factor graph
       ...     if state.is_fac[v]:
       ...         continue                  # skip over factors
       ...     print(state.c[v], state.x[v]) # clique occupation
       ...     if state.x[v] > 0:
       ...         cliques.append(state.c[v])
       ...     if len(cliques) > 10:
       ...         break
       array([0, 2, 4, 5], dtype=int32) 1
       array([0, 1, 3, 5], dtype=int32) 1
       array([0, 1, 5, 6], dtype=int32) 1
       array([0, 4, 5, 6], dtype=int32) 0
       array([2, 5, 7], dtype=int32) 1
       array([ 4, 28], dtype=int32) 1
       array([ 4,  6, 29], dtype=int32) 1
       array([ 4, 30, 31], dtype=int32) 0
       array([5, 6, 7], dtype=int32) 0
       array([ 7, 71], dtype=int32) 1
       array([ 7, 14, 58], dtype=int32) 1
       array([ 7, 58, 85], dtype=int32) 0
       array([ 7, 30, 58], dtype=int32) 1
       array([ 6, 10, 12], dtype=int32) 0
       array([ 6, 12, 18], dtype=int32) 1
       array([ 8, 12, 13, 32], dtype=int32) 1

    References
    ----------
    .. [young-hypergraph-2021] Young, JG., Petri, G., Peixoto, T.P. "Hypergraph
       reconstruction from network data", Commun Phys 4, 135
       (2021), :doi:`10.1038/s42005-021-00637-w`, :arxiv:`2008.04948`

    """

    def __init__(self, g, init_edges=False, init_max_cliques=True,
                 init_list=None):
        self.g = GraphView(g, directed=False)
        self.f = Graph(directed=False)             # factor graph
        self.f.set_fast_edge_removal(True)
        self.is_fac = self.f.new_vp("bool")        # is a factor
        self.c = self.f.new_vp("vector<int32_t>")  # coordinates
        self.is_max = self.f.new_vp("bool")        # is a max clique
        self.x = self.f.new_vp("int")              # activation status
        edges = {}
        for e in g.edges():
            v = self.f.add_vertex()
            self.is_fac[v] = 1
            st = tuple(sorted([int(e.source()), int(e.target())]))
            self.c[v] = st
            edges[st] = v
        for q in max_cliques(g):
            d = len(q)
            cs = tuple(sorted(q))
            v = self.f.add_vertex()
            self.c[v] = cs
            self.is_max[v] = True
            for s, t in combinations(q, 2):
                if s > t:
                    s, t = t, s
                u = edges[(s,t)]
                self.f.add_edge(v, u)
            if init_max_cliques:
                self.x[v] = 1
        if init_edges:
            coords = {}
            for v in self.f.vertices():
                if self.is_fac[v]:
                    continue
                coords[tuple(self.c[v].a)] = v
            for e in g.edges():
                st = tuple(sorted([int(e.source()), int(e.target())]))
                if st in coords:
                    v = coords[st]
                else:
                    v = self.f.add_vertex()
                    self.c[v] = st
                    u = edges[st]
                    self.f.add_edge(v, u)
                self.x[v] = 1
        if init_list is not None:
            coords = {}
            for v in self.f.vertices():
                if self.is_fac[v]:
                    continue
                coords[tuple(self.c[v].a)] = v
            for c, count in init_list.items():
                st = tuple(sorted(c))
                if st in coords:
                    v = coords[st]
                else:
                    v = self.f.add_vertex()
                    self.c[v] = st
                    for s, t in combinations(q, 2):
                        if s > t:
                            s, t = t, s
                        u = edges[(s,t)]
                        self.f.add_edge(v, u)
                self.x[v] = count
        self.reset_factors()

        self.lpd = []
        self.D = max([len(self.c[v]) for v in self.f.vertices()])
        self.N = self.g.num_vertices()
        self.E = self.g.num_edges()

        self.reset_Ed()

    def __copy__(self):
        return self.copy()

    def copy(self, **kwargs):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""

        return CliqueState(**dict(self.__getstate__(), **kwargs))

    def __getstate__(self):
        init_list = {}
        for v in self.f.vertices():
            if self.is_fac[v]:
                continue
            init_list[tuple(self.c[v])] = self.x[v]
        return dict(g=self.g, init_list=init_list)

    def __setstate__(self, state):
        self.__init__(**state)

    def reset_factors(self):
        """Reset factor constraints"""
        for v in self.f.vertices():
            if self.is_fac[v] == 1:
                self.x[v] = sum(self.x[w] for w in v.out_neighbors())

    def reset_Ed(self):
        """Reset edge counts"""
        self.Ed = zeros(self.D + 1, dtype="int32")
        for v in self.f.vertices():
            if self.is_fac[v]:
                continue
            d = len(self.c[v])
            self.Ed[d] += self.x[v]

    def get_nEd(self):
        """Get fraction of edge counts per clique size"""
        Ed = asarray(self.Ed.copy(), dtype="float")
        for d in range(2, len(Ed)):
            Ed[d] *= (d * (d-1)) // 2
        Ed /= self.g.num_edges()
        return Ed

    @copy_state_wrap
    def entropy(self, **kwargs):
        """Get the description length, a.k.a. the negative log-likelihood."""
        L = 0
        for d in range(2, len(self.Ed)):
            L += libinference.clique_L_over(self.N, d, self.Ed[d], self.D, self.E)
        L -= gammaln(self.x.fa[logical_not(self.is_fac.fa)] + 1).sum()
        return -L

    @mcmc_sweep_wrap
    def mcmc_sweep(self, niter=1, beta=1):
        """Runs ``niter`` iterations of a Metropolis-Hastings MCMC, with inverse
        temperature ``beta``, to sample clique decompositions. This returns the change
        in description length and the number of moves."""
        return libinference.clique_iter_mh(self.f._Graph__graph,
                                           self.c._get_any(), self.x._get_any(),
                                           self.is_fac._get_any(),
                                           self.is_max._get_any(), self.Ed,
                                           self.N, self.E, beta, niter,
                                           _get_rng())
