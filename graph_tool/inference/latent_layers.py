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

from .. import _prop, Graph, GraphView, _get_rng, PropertyMap, \
    edge_endpoint_property, Vector_size_t
from .. generation import generate_triadic_closure, remove_parallel_edges, \
    remove_self_loops
from .. spectral import adjacency

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

from . blockmodel import *
from . nested_blockmodel import *
from . base_states import _bm_test
from . uncertain_blockmodel import UncertainBaseState

import numpy.random

class LatentLayerBaseState(EntropyState):
    r"""Base state for uncertain latent layer network inference."""

    def get_ec(self, ew=None):
        """Return edge property map with layer membership."""
        if ew is None:
            ew = self.ew
        ec = []
        for u, ew in zip(self.us, ew):
            w = self.g.copy_property(ew, g=u)
            ec.append(w)
        return ec

    def collect_marginal(self, gs=None, total=False):
        r"""Collect marginal inferred network during MCMC runs.

        Parameters
        ----------
        g : list of :class:`~graph_tool.Graph` (optional, default: ``None``)
            Previous marginal graphs.

        Returns
        -------
        g : list :class:`~graph_tool.Graph`
            New list of marginal graphs, each with internal edge
            :class:`~graph_tool.EdgePropertyMap` ``"eprob"``, containing the
            marginal probabilities for each edge.

        Notes
        -----
        The posterior marginal probability of an edge :math:`(i,j)` is defined as

        .. math::

           \pi_{ij} = \sum_{\boldsymbol A}A_{ij}P(\boldsymbol A|\boldsymbol D)

        where :math:`P(\boldsymbol A|\boldsymbol D)` is the posterior
        probability given the data.

        This function returns a list with the marginal graphs for every layer.
        """

        if gs is None:
            gs = []
            L = len(self.us)
            if total:
                L += 1
            for l in range(L):
                g = Graph(directed=self.g.is_directed())
                g.add_vertex(self.g.num_vertices())
                g.gp.count = g.new_gp("int", 0)
                g.ep.count = g.new_ep("int")
                gs.append(g)

        for l, g in enumerate(gs):
            if "eprob" not in g.ep:
                g.ep.eprob = g.new_ep("double")

            u = self.us[l] if l < len(self.us) else self.g
            if l == 0:
                es = edge_endpoint_property(u, u.vertex_index, "source")
                et = edge_endpoint_property(u, u.vertex_index, "target")
                u = GraphView(u, efilt=es.fa != et.fa)
            libinference.collect_marginal(g._Graph__graph,
                                          u._Graph__graph,
                                          _prop("e", g, g.ep.count))
            g.gp.count += 1
            g.ep.eprob.fa = g.ep.count.fa
            g.ep.eprob.fa /= g.gp.count
        return gs

    def collect_marginal_multigraph(self, gs=None):
        r"""Collect marginal latent multigraphs during MCMC runs.

        Parameters
        ----------
        g : list of :class:`~graph_tool.Graph` (optional, default: ``None``)
            Previous marginal multigraphs.

        Returns
        -------
        g : list of :class:`~graph_tool.Graph`
            New marginal multigraphs, each with internal edge
            :class:`~graph_tool.EdgePropertyMap` ``"w"`` and ``"wcount"``,
            containing the edge multiplicities and their respective counts.

        Notes
        -----

        The mean posterior marginal multiplicity distribution of a multi-edge
        :math:`(i,j)` is defined as

        .. math::

           \pi_{ij}(w) = \sum_{\boldsymbol G}\delta_{w,G_{ij}}P(\boldsymbol G|\boldsymbol D)

        where :math:`P(\boldsymbol G|\boldsymbol D)` is the posterior
        probability of a multigraph :math:`\boldsymbol G` given the data.

        This function returns a list with the marginal graphs for every layer.
        """

        if gs is None or len(gs) != len(self.us):
            gs = []
            for l in range(len(self.us)):
                g = Graph(directed=self.g.is_directed())
                g.add_vertex(self.g.num_vertices())
                g.ep.w = g.new_ep("vector<int>")
                g.ep.wcount = g.new_ep("vector<int>")
                gs.append(g)

        for l, g in enumerate(gs):
            u = self.us[l]
            ew = self.ew[l]
            libinference.collect_marginal_count(g._Graph__graph,
                                                u._Graph__graph,
                                                _prop("e", u, ew),
                                                _prop("e", g, g.ep.w),
                                                _prop("e", g, g.ep.wcount))
        return gs

    def _gen_eargs(self, args):
        if isinstance(self.bstates[0], NestedBlockState):
            bstate = self.bstates[0].levels[0]
        else:
            bstate = self.bstates[0]
        ea = bstate._get_entropy_args(args, consume=True)
        return libinference.uentropy_args(ea)

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_latent_layers_sweep(mcmc_state,
                                                     self._state,
                                                     _get_rng())

    @mcmc_sweep_wrap
    def _algo_sweep(self, algo, r=.5, **kwargs):
        kwargs = kwargs.copy()
        beta = kwargs.get("beta", 1.)
        niter = kwargs.get("niter", 1)
        verbose = kwargs.get("verbose", False)
        entropy_args = self._get_entropy_args(kwargs.get("entropy_args", {}))
        kwargs.get("entropy_args", {}).pop("latent_edges", None)
        kwargs.get("entropy_args", {}).pop("density", None)
        state = self._state

        mcmc_state = DictState(dict(kwargs, **locals()))

        if numpy.random.random() < r:
            for s in self.bstates:
                s._clear_egroups()
            return self._mcmc_sweep(mcmc_state)
        else:
            bstate = numpy.random.choice(self.bstates)
            bstate._clear_egroups()
            return algo(bstate, **dict(kwargs, test=False))


    def mcmc_sweep(self, r=.5, multiflip=True, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection sampling MCMC to
        sample network partitions and latent edges. The parameter ``r`` controls
        the probability with which edge move will be attempted, instead of
        partition moves. The remaining keyword parameters will be passed to
        :meth:`~graph_tool.inference.BlockState.mcmc_sweep` or
        :meth:`~graph_tool.inference.BlockState.multiflip_mcmc_sweep`,
        if ``multiflip=True``.
        """

        if multiflip:
            return self._algo_sweep(lambda s, **kw: s.multiflip_mcmc_sweep(**kw),
                                    r=r, **kwargs)
        else:
            return self._algo_sweep(lambda s, **kw: s.mcmc_sweep(**kw),
                                    r=r, **kwargs)

    def multiflip_mcmc_sweep(self, **kwargs):
        r"""Alias for :meth:`~LatentLayerBaseState.mcmc_sweep` with ``multiflip=True``."""
        return self.mcmc_sweep(multiflip=True, **kwargs)


@entropy_state_signature
class LatentClosureBlockState(LatentLayerBaseState):
    r"""Inference state of the stochastic block model with latent triadic closure
    edges.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Observed graph.
    L : ``int`` (optional, default: ``1``)
        Maximum number of triadic closure generations.
    b : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Inital partition (or hierarchical partition ``nested=True``).
    aE : ``float`` (optional, default: ``NaN``)
        Expected total number of edges used in prior. If ``NaN``, a flat
        prior will be used instead.
    nested : ``boolean`` (optional, default: ``True``)
        If ``True``, a :class:`~graph_tool.inference.NestedBlockState`
        will be used, otherwise
        :class:`~graph_tool.inference.BlockState`.
    state_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to
        :class:`~graph_tool.inference.NestedBlockState` or
        :class:`~graph_tool.inference.BlockState`.
    g_orig : :class:`~graph_tool.Graph` (optional, default: ``None``)
        Original graph, if ``g`` is used to initialize differently from a graph with no triadic closure edges.
    ew : list of :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        List of edge property maps of ``g``, containing the initial weights
        (counts) at each triadic generation.
    ex : list of :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        List of edge property maps of ``g``, each containing a list of integers
        with the ego graph memberships of every edge, for every triadic
        generation.
    entropy_args: ``dict`` (optional, default: ``{}``)
        Override default arguments for
        :meth:`~LatentClosureBlockState.entropy()` method and releated
        operations.

    References
    ----------
    .. [peixoto-disentangling-2022] Tiago P. Peixoto, "Disentangling homophily,
       community structure and triadic closure in networks", Phys. Rev. X 12,
       011004 (2022), :doi:`10.1103/PhysRevX.12.011004`, :arxiv:`2101.02510`

    """

    def __init__(self, g, L=1, b=None, aE=numpy.nan, nested=True,
                 state_args={}, g_orig=None, ew=None, ex=None, entropy_args={},
                 **kwargs):
        EntropyState.__init__(self, entropy_args=entropy_args)

        self.g = g
        self.us = []
        self.ew = []
        self.ex = []

        if ew is None:
            ew = [g.new_ep("int", val=1)] + [g.new_ep("int", val=0) for l in range(L)]
        if ex is None:
            ex = [g.new_ep("vector<int>") for l in range(L+1)]

        for w, x in zip(ew, ex):
            u = GraphView(g, efilt=w.fa > 0)
            u.ep.w = w
            u.ep.x = x
            u = Graph(u, prune=True)
            self.us.append(u)
            self.ew.append(u.ep.w)
            self.ex.append(u.ep.x)

        self.nested = nested
        self.state_args = state_args

        if nested:
            self.bstate = NestedBlockState(self.us[0], eweight=self.ew[0], bs=b,
                                           **state_args)
            self.ew[0] = self.bstate.levels[0].eweight
            self.b = self.bstate.levels[0].b
        else:
            self.bstate = BlockState(self.us[0], eweight=self.ew[0], b=b,
                                     **state_args)
            self.ew[0] = self.bstate.eweight
            self.b = self.bstate.b

        if nested:
            self.pstate = self.bstate.levels[0]
        else:
            self.pstate = self.bstate

        self._entropy_args.update(self.pstate._entropy_args)

        self.aE = aE
        if numpy.isnan(aE):
            self.E_prior = False
        else:
            self.E_prior = True

        if g_orig is None:
            self.g_orig = g
            self.g = g = g.copy()
        else:
            self.g_orig = g_orig

        self.og = self.g._get_any()
        self.eweight = self.g.new_ep("int")
        self.oa = [u._get_any() for u in self.us]
        self.oaw = [w._get_any() for w in self.ew]
        self.ox = [x._get_any() for x in self.ex]
        self.L = len(self.us)
        self.m = [u.new_ep("vector<int>") for u in self.us]
        self.om = [m._get_any() for m in self.m]
        self.M = [u.new_vp("int") for u in self.us]
        self.oM = [M._get_any() for M in self.M]
        self.E = [u.new_vp("int") for u in self.us]
        self.oE = [E._get_any() for E in self.E]
        self.bstates = [self.bstate]

        self.measured = kwargs.get("measured", False)
        self.ag_orig = self.g_orig._get_any()
        self.n = kwargs.get("n", self.g_orig.new_ep("int", 1))
        self.x = kwargs.get("x", self.g_orig.new_ep("int", 1))
        self.n_default = kwargs.get("n_default", 1)
        self.x_default = kwargs.get("x_default", 0)
        fn_params = kwargs.get("fn_params", {})
        fp_params = kwargs.get("fp_params", {})
        self.alpha = fn_params.get("alpha", 1)
        self.beta = fn_params.get("beta", 1)
        self.mu = fp_params.get("mu", 1)
        self.nu = fp_params.get("nu", 1)
        self.lp = kwargs.get("lp", numpy.nan)
        self.lq = kwargs.get("lq", numpy.nan)
        self.max_w = kwargs.get("max_w", 1 << 16)
        self.self_loops = True

        if nested:
            bstate = self.bstate.levels[0]._state
        else:
            bstate = self.bstate._state

        ret = libinference.make_latent_closure_state(bstate,
                                                     self.pstate._state,
                                                     self, self.L)

        self._cstates = ret[:self.L]
        self._state = ret[-1]

        cstate = self._cstates[0]
        if nested:
            bstate = self.bstate.levels[0]._state
        else:
            bstate = self.bstate._state
        pstate = self.pstate._state

    def __getstate__(self):
        state = EntropyState.__getstate__(self)
        return dict(state, g=self.g, L=self.L-1,
                    b=self.bstate.get_bs() if self.nested else self.bstate.b.copy(),
                    aE=self.aE, nested=self.nested,
                    state_args=self.state_args, g_orig=self.g_orig,
                    ew=self.get_ec(self.ew), ex=self.get_ec(self.ex))

    def __setstate__(self, state):
        self.__init__(**state)

    def copy(self, **kwargs):
        """Return a copy of the state."""
        return LatentClosureBlockState(**dict(self.__getstate__(), **kwargs))

    def __copy__(self):
        return self.copy()

    def __repr__(self):
        return "<LatentClosureBlockState object with (%s) closure edges, and %s, at 0x%x>" % \
            (", ".join([str(w.fa.sum()) for w in self.ew[1:]]), repr(self.bstate), id(self))

    @copy_state_wrap
    def _entropy(self, latent_edges=True, density=False, aE=1., sbm=True, **kwargs):
        """Return the entropy, i.e. negative log-likelihood."""
        eargs = self._get_entropy_args(locals())
        S = self._state.entropy(eargs)
        S += self.bstates[0].entropy(**kwargs)
        for s in self._cstates[1:]:
            S += s.entropy()
        return S

    def sample_graph(self, sample_sbm=True, canonical_sbm=False,
                     sample_params=True, canonical_closure=True):
        """Sample graph from inferred model.

        Parameters
        ----------
        sample_sbm : ``boolean`` (optional, default: ``True``)
            If ``True``, the substrate network will be sampled anew from the SBM
            parameters. Otherwise, it will be the same as the current posterior
            state.
        canonical_sbm : ``boolean`` (optional, default: ``False``)
            If ``True``, the canonical SBM will be used, otherwise the
            microcanonical SBM will be used.
        sample_params : ``bool`` (optional, default: ``True``)
            If ``True``, and ``canonical_sbm == True`` the count parameters
            (edges between groups and node degrees) will be sampled from their
            posterior distribution conditioned on the actual state. Otherwise,
            their maximum-likelihood values will be used.
        canonical_closure : ``boolean`` (optional, default: ``True``)
            If ``True``, the canonical version of triadic clousre will be used
            (i.e. conditioned on a probability), otherwise the microcanonical
            version will be used (i.e. conditional on the count number).

        Returns
        -------
        u : list :class:`~graph_tool.Graph`
            Sampled graph, with internal edge
            :class:`~graph_tool.EdgePropertyMap` ``"gen"``, containing the
            triadic generation of each edge.

        """
        if sample_sbm:
            if self.nested:
                bstate = self.bstate.levels[0]
            else:
                bstate = self.bstate
            u = bstate.sample_graph(self_loops=False, multigraph=False,
                                    canonical=canonical_sbm,
                                    sample_params=sample_params)
        else:
            u = self.us[0].copy()

        u.ep.gen = u.new_ep("int")

        for l, (g, w) in enumerate(zip(self.us[1:], self.ew[1:])):

            t = u.own_property(self.E[l + 1])

            if canonical_closure:
                M = self.M[l + 1]
                t = t.copy("double")
                idx = t.a > 0
                t.fa[idx] = numpy.random.beta(t.a[idx] + 1, (M.a - t.a)[idx] + 1)

            if t.a.sum() == 0:
                break

            old = u.new_ep("bool", True)

            curr = u.new_ep("bool", vals=u.ep.gen.fa == l)
            generate_triadic_closure(u, curr=curr, t=t, probs=canonical_closure)

            new = GraphView(u, efilt=numpy.logical_not(old.fa))
            remove_parallel_edges(new)

            gen = new.own_property(u.ep.gen)
            gen.fa = l + 1

        return u

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_latent_closure_sweep(mcmc_state,
                                                      self._state,
                                                      _get_rng())

    @mcmc_sweep_wrap
    def _algo_sweep(self, algo, r=.5, **kwargs):
        kwargs = kwargs.copy()
        beta = kwargs.get("beta", 1.)
        niter = kwargs.get("niter", 1)
        verbose = kwargs.get("verbose", False)
        entropy_args = self._get_entropy_args(kwargs.get("entropy_args", {}))
        kwargs.get("entropy_args", {}).pop("latent_edges", None)
        kwargs.get("entropy_args", {}).pop("density", None)
        state = self._state

        mcmc_state = DictState(dict(kwargs, **locals()))

        if numpy.random.random() < r:
            for s in self.bstates:
                s._clear_egroups()
            mcmc_state.niter *= len(self.us)
            return self._mcmc_sweep(mcmc_state)
        else:
            bstate = self.bstates[0]
            bstate._clear_egroups()
            return algo(bstate, **dict(kwargs, test=False))

        return dS, nattempts, nmoves

    def collect_marginal(self, gs=None):
        r"""Collect marginal inferred network during MCMC runs.

        Parameters
        ----------
        g : list of :class:`~graph_tool.Graph` (optional, default: ``None``)
            Previous marginal graphs.

        Returns
        -------
        g : list :class:`~graph_tool.Graph`

            New list of marginal graphs, each with internal
            :class:`~graph_tool.EdgePropertyMap` ``"eprob"``, containing the
            marginal probabilities for each edge, as well as
            :class:`~graph_tool.VertexPropertyMap` ``"t"``, ``"m"``, ``"c"``,
            containing the average number of closures, open triads, and fraction
            of closed triads on each node.

        Notes
        -----
        The posterior marginal probability of an edge :math:`(i,j)` is defined as

        .. math::

           \pi_{ij} = \sum_{\boldsymbol A}A_{ij}P(\boldsymbol A|\boldsymbol D)

        where :math:`P(\boldsymbol A|\boldsymbol D)` is the posterior
        probability given the data.

        This function returns a list with the marginal graphs for every layer.

        """

        gs = LatentLayerBaseState.collect_marginal(self, gs, total=self.measured)
        for l in range(len(self.us)):
            E = self.E[l]
            M = self.M[l]
            u = gs[l]

            tsum = u.vp.get("tsum", None)
            if tsum is None:
                tsum = u.vp.tsum = u.new_vp("int")
                u.vp.msum = u.new_vp("int")
                u.vp.t = u.new_vp("double")
                u.vp.m = u.new_vp("double")
                u.vp.csum = u.new_vp("double")
                u.vp.c = u.new_vp("double")

            msum = u.vp.msum
            t = u.vp.t
            m = u.vp.m
            csum = u.vp.csum
            c = u.vp.c

            tsum.a += E.a
            msum.a += M.a
            idx = M.a > 0
            csum.a[idx] += E.a[idx] / M.a[idx]

            t.a = tsum.a / u.gp.count
            m.a = msum.a / u.gp.count
            c.a = csum.a / u.gp.count

        return gs

class MeasuredClosureBlockState(LatentClosureBlockState, UncertainBaseState):
    r"""Inference state of a measured graph, using the stochastic block model with
    triadic closure as a prior.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Measured graph.
    n : :class:`~graph_tool.EdgePropertyMap`
        Edge property map of type ``int``, containing the total number of
        measurements for each edge.
    x : :class:`~graph_tool.EdgePropertyMap`
        Edge property map of type ``int``, containing the number of
        positive measurements for each edge.
    n_default : ``int`` (optional, default: ``1``)
        Total number of measurements for each non-edge.
    x_default : ``int`` (optional, default: ``0``)
        Total number of positive measurements for each non-edge.
    L : ``int`` (optional, default: ``1``)
        Maximum number of triadic closure generations.
    b : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Inital partition (or hierarchical partition ``nested=True``).
    fn_params : ``dict`` (optional, default: ``dict(alpha=1, beta=1)``)
        Beta distribution hyperparameters for the probability of missing
        edges (false negatives).
    fp_params : ``dict`` (optional, default: ``dict(mu=1, nu=1)``)
        Beta distribution hyperparameters for the probability of spurious
        edges (false positives).
    aE : ``float`` (optional, default: ``NaN``)
        Expected total number of edges used in prior. If ``NaN``, a flat
        prior will be used instead.
    nested : ``boolean`` (optional, default: ``True``)
        If ``True``, a :class:`~graph_tool.inference.NestedBlockState`
        will be used, otherwise
        :class:`~graph_tool.inference.BlockState`.
    state_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to
        :class:`~graph_tool.inference.NestedBlockState` or
        :class:`~graph_tool.inference.BlockState`.
    bstate : :class:`~graph_tool.inference.NestedBlockState` or :class:`~graph_tool.inference.BlockState`  (optional, default: ``None``)
        If passed, this will be used to initialize the block state
        directly.
    g_orig : :class:`~graph_tool.Graph` (optional, default: ``None``)
        Original graph, if ``g`` is used to initialize differently from a graph with no triadic closure edges.
    ew : list of :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        List of edge property maps of ``g``, containing the initial weights
        (counts) at each triadic generation.
    ex : list of :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        List of edge property maps of ``g``, each containing a list of integers
        with the ego graph memberships of every edge, for every triadic
        generation.

    References
    ----------
    .. [peixoto-disentangling-2022] Tiago P. Peixoto, "Disentangling homophily,
       community structure and triadic closure in networks", Phys. Rev. X 12,
       011004 (2022), :doi:`10.1103/PhysRevX.12.011004`, :arxiv:`2101.02510`
    """

    def __init__(self, g, n, x, n_default=1, x_default=0, L=1, b=None,
                 fn_params=dict(alpha=1, beta=1), fp_params=dict(mu=1, nu=1),
                 aE=numpy.nan, nested=True, state_args={}, bstate=None,
                 g_orig=None, ew=None, ex=None, **kwargs):

        UncertainBaseState.__init__(self, g, nested=nested, state_args=state_args,
                                    bstate=bstate, **kwargs)
        LatentClosureBlockState.__init__(self, g, L=L, b=b, aE=aE,
                                         nested=nested, state_args=state_args,
                                         g_orig=g_orig,
                                         ew=ew, ex=ex, n=n, x=x,
                                         n_default=n_default,
                                         x_default=x_default,
                                         fn_params=fn_params,
                                         fp_params=fp_params, measured=True)
    def __getstate__(self):
        return dict(g=self.g, n=self.n, x=self.x,
                    n_default=self.n_default,
                    x_default=self.x_default, L=self.L-1,
                    b=self.bstate.get_bs() if self.nested else self.bstate.b.copy(),
                    fn_params=dict(alpha=self.alpha, beta=self.beta),
                    fp_params=dict(mu=self.mu, nu=self.nu),
                    aE=self.aE, nested=self.nested,
                    state_args=self.state_args, g_orig=self.g_orig,
                    ew=self.get_ec(self.ew), ex=self.get_ec(self.ex))

    def __setstate__(self, state):
        self.__init__(**state)

    def copy(self, **kwargs):
        """Return a copy of the state."""
        return MeasuredClosureBlockState(**dict(self.__getstate__(), **kwargs))

    def __repr__(self):
        return "<MeasuredClosureBlockState object with (%s) closure edges, and %s, at 0x%x>" % \
            (", ".join([str(w.fa.sum()) for w in self.ew[1:]]), repr(self.bstate), id(self))

    def get_graph(self):
        r"""Return the current inferred graph."""
        es = edge_endpoint_property(self.g, self.g.vertex_index, "source")
        et = edge_endpoint_property(self.g, self.g.vertex_index, "target")
        u = GraphView(self.g, efilt=numpy.logical_and(self.eweight.fa > 0,
                                                      es.fa != et.fa))
        return u

    def set_hparams(self, alpha, beta, mu, nu):
        """Set edge and non-edge hyperparameters."""
        self._state.set_hparams(alpha, beta, mu, nu)
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nu = nu

    def get_p_posterior(self):
        """Get beta distribution parameters for the posterior probability of missing edges."""
        T = self._state.get_T()
        M = self._state.get_M()
        return M - T + self.alpha, T + self.beta

    def get_q_posterior(self):
        """Get beta distribution parameters for the posterior probability of spurious edges."""
        N = self._state.get_N()
        X = self._state.get_X()
        T = self._state.get_T()
        M = self._state.get_M()
        return X - T + self.mu, N - X - (M - T) + self.nu
