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
    EdgePropertyMap, VertexPropertyMap, edge_endpoint_property, Vector_int32_t, \
    Vector_size_t

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

from . base_states import *

from . blockmodel import *
from . nested_blockmodel import *

import numpy as np
import scipy.special

import collections.abc

@entropy_state_signature
class UncertainBaseState(EntropyState):
    r"""Base state for uncertain network inference."""

    def __init__(self, g, nested=True, state_args={}, bstate=None,
                 self_loops=False, init_empty=False, max_m=1 << 16,
                 entropy_args={}):
        EntropyState.__init__(self, entropy_args=entropy_args)

        self.g = g

        state_args = dict(state_args)
        if bstate is None:
            if init_empty:
                self.u = Graph(g.num_vertices(), directed=g.is_directed())
                self.eweight = self.u.new_ep("int", val=1)
            elif "g" in state_args:
                self.u = state_args.pop("g")
                self.eweight = state_args.pop("eweight",
                                              self.u.new_ep("int", val=1))
            else:
                self.u = g.copy()
                self.eweight = self.u.new_ep("int", val=1)
        else:
            self.u = bstate.g.copy()
            if nested:
                self.eweight = bstate.levels[0].eweight
            else:
                self.eweight = bstate.eweight
            self.eweight = self.u.own_property(self.eweight.copy())
            if nested:
                bstate = bstate.copy(g=self.u,
                                     state_args=dict(bstate.state_args,
                                                     eweight=self.eweight))
            else:
                bstate = bstate.copy(g=self.u, eweight=self.eweight)

        self.u.set_fast_edge_removal()

        self.self_loops = self_loops
        N = self.u.num_vertices()
        if self.u.is_directed():
            if self_loops:
                M = N * N
            else:
                M = N * (N - 1)
        else:
            if self_loops:
                M = (N * (N + 1)) / 2
            else:
                M = (N * (N - 1)) / 2

        self.M = M

        if bstate is None:
            if nested:
                state_args["state_args"] = state_args.get("state_args", {})
                state_args["state_args"].update(dict(eweight=self.eweight))
                self.nbstate = NestedBlockState(self.u, **state_args)
                self.bstate = self.nbstate.levels[0]
            else:
                self.nbstate = None
                self.bstate = BlockState(self.u, eweight=self.eweight,
                                         **state_args)
        else:
            if nested:
                self.nbstate = bstate
                self.bstate = bstate.levels[0]
            else:
                self.nbstate = None
                self.bstate = bstate

        self._entropy_args.update(self.bstate._entropy_args)
        self._entropy_args.update(entropy_args)

        edges = self.g.get_edges()
        edges = numpy.concatenate((edges,
                                   numpy.ones(edges.shape,
                                              dtype=edges.dtype) * (N + 1)))
        self.slist = Vector_size_t(init=edges[:,0])
        self.tlist = Vector_size_t(init=edges[:,1])

        self.max_m = max_m

        init_q_cache()

    def __getstate__(self):
        state = EntropyState.__getstate__(self)
        return dict(state, g=self.g, nested=self.nbstate is not None,
                    bstate=(self.nbstate if self.nbstate is not None else self.bstate),
                    self_loops=self.self_loops, max_m=self.max_m)

    def __setstate__(self, state):
        self.__init__(**state)

    def copy(self, **kwargs):
        """Return a copy of the state."""
        args = dict(self.__getstate__(), **kwargs)
        return type(self)(**args)

    def __copy__(self):
        return self.copy()

    def _gen_eargs(self, args):
        ea = self.bstate._get_entropy_args(args, consume=True)
        return libinference.uentropy_args(ea)

    def get_block_state(self):
        """Return the underlying block state, which can be either
        :class:`~graph_tool.inference.BlockState` or
        :class:`~graph_tool.inference.NestedBlockState`.
        """
        if self.nbstate is None:
            return self.bstate
        else:
            return self.nbstate

    @copy_state_wrap
    def _entropy(self, latent_edges=True, density=False, aE=1., sbm=True, **kwargs):
        """Return the description length, i.e. the negative log-likelihood."""
        eargs = self._get_entropy_args(locals())
        S = self._state.entropy(eargs)
        if sbm:
            if self.nbstate is None:
                S += self.bstate.entropy(**kwargs)
            else:
                S += self.nbstate.entropy(**kwargs)
        return S

    def virtual_remove_edge(self, u, v, dm=1, entropy_args={}):
        """Return the difference in description length if edge :math:`(u, v)`
        with multiplicity ``dm`` would be removed.
        """
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.remove_edge_dS(int(u), int(v), dm, entropy_args)

    def virtual_add_edge(self, u, v, dm=1, entropy_args={}):
        """Return the difference in description length if edge :math:`(u, v)`
        would be added with multiplicity ``dm``.
        """
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.add_edge_dS(int(u), int(v), dm, entropy_args)

    def remove_edge(self, u, v, dm=1):
        r"""Remove edge :math:`(u, v)` with multiplicity ``dm``."""
        return self._state.remove_edge(int(u), int(v), dm)

    def add_edge(self, u, v, dm=1):
        r"""Add edge :math:`(u, v)` with multiplicity ``dm``."""
        return self._state.add_edge(int(u), int(v), dm)

    def set_state(self, g, w):
        r"""Set all edge multiplicities via :class:`~graph_tool.EdgePropertyMap` ``w``."""
        if w.value_type() != "int32_t":
            w = w.copy("int32_t")
        self._state.set_state(g._Graph__graph, w._get_any())


    @mcmc_sweep_wrap
    def edge_mcmc_sweep(self, beta=1, niter=1, verbose=False, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection
        sampling MCMC to sample latent edges.

        Parameters
        ----------
        beta : ``float`` (optional, default: ``np.inf``)
            Inverse temperature parameter.
        niter : ``int`` (optional, default: ``1``)
            Number of sweeps.
        verbose : ``boolean`` (optional, default: ``False``)
            If ``verbose == True``, detailed information will be displayed.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nmoves : ``int``
            Number of variables moved.
        """
        kwargs = kwargs.copy()
        edges_only = kwargs.pop("edges_only", False)
        slist = self.slist
        tlist = self.tlist
        entropy_args = kwargs.pop("entropy_args", {}).copy()
        entropy_args = self._get_entropy_args(entropy_args)
        debug = kwargs.pop("debug", False)
        state = self._state

        mcmc_state = DictState(dict(kwargs, **locals()))

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))

        return self._mcmc_sweep(mcmc_state)

    #@mcmc_sweep_wrap
    def sbm_mcmc_sweep(self, multiflip=True, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection
        sampling MCMC to sample node partitions. The remaining keyword
        parameters will be passed to
        :meth:`~graph_tool.inference.BlockState.mcmc_sweep` or
        :meth:`~graph_tool.inference.BlockState.multiflip_mcmc_sweep`, if
        ``multiflip=True``.
        """
        if self.nbstate is None:
            self.bstate._clear_egroups()
        else:
            self.nbstate._clear_egroups()
        bstate = self.nbstate
        if bstate is None:
            bstate = self.bstate
        if multiflip:
            return bstate.multiflip_mcmc_sweep(**kwargs)
        else:
            return bstate.mcmc_sweep(**kwargs)

    def mcmc_sweep(self, beta=1, niter=1, pedges=.5, multiflip=True,
                   edge_mcmc_args=dict(), sbm_mcmc_args=dict(), **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection
        sampling MCMC to sample network partitions and latent edges. The
        parameter ``pedges`` controls the probability with which edge moves will
        be attempted, instead of partition moves. The remaining keyword
        parameters will be passed to
        :meth:`~graph_tool.inference.BlockState.mcmc_sweep` or
        :meth:`~graph_tool.inference.BlockState.multiflip_mcmc_sweep`, if
        ``multiflip=True``.

        """
        if numpy.random.random() < pedges:
            return self.edge_mcmc_sweep(**dict(dict(dict(beta=beta, niter=niter),
                                                    **edge_mcmc_args), **kwargs))
        else:
            return self.sbm_mcmc_sweep(**dict(dict(dict(beta=beta, niter=niter,
                                                        multiflip=multiflip),
                                                   **sbm_mcmc_args), **kwargs))

    def get_edge_prob(self, u, v, entropy_args={}, epsilon=1e-8):
        r"""Return conditional posterior log-probability of edge :math:`(u,v)`."""
        ea = self._get_entropy_args(entropy_args)
        return self._state.get_edge_prob(u, v, ea, epsilon)

    def get_edges_prob(self, elist, entropy_args={}, epsilon=1e-8):
        r"""Return conditional posterior log-probability of an edge list, with
        shape :math:`(E,2)`."""
        ea = self._get_entropy_args(entropy_args)
        elist = numpy.asarray(elist, dtype="uint64")
        probs = numpy.zeros(elist.shape[0])
        self._state.get_edges_prob(elist, probs, ea, epsilon)
        return probs

    def get_graph(self):
        r"""Return the current inferred graph."""
        if self.self_loops:
            u = GraphView(self.u, efilt=self.eweight.fa > 0)
        else:
            es = edge_endpoint_property(self.u, self.u.vertex_index, "source")
            et = edge_endpoint_property(self.u, self.u.vertex_index, "target")
            u = GraphView(self.u, efilt=numpy.logical_and(self.eweight.fa > 0,
                                                          es.fa != et.fa))
        return u

    def collect_marginal(self, g=None):
        r"""Collect marginal inferred network during MCMC runs.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph` (optional, default: ``None``)
            Previous marginal graph.

        Returns
        -------
        g : :class:`~graph_tool.Graph`
            New marginal graph, with internal edge :class:`~graph_tool.EdgePropertyMap`
            ``"eprob"``, containing the marginal probabilities for each edge.

        Notes
        -----
        The posterior marginal probability of an edge :math:`(i,j)` is defined as

        .. math::

           \pi_{ij} = \sum_{\boldsymbol A}A_{ij}P(\boldsymbol A|\boldsymbol D)

        where :math:`P(\boldsymbol A|\boldsymbol D)` is the posterior
        probability given the data.

        """

        if g is None:
            g = Graph(directed=self.g.is_directed())
            g.add_vertex(self.g.num_vertices())
            g.gp.count = g.new_gp("int", 0)
            g.ep.count = g.new_ep("int")

        if "eprob" not in g.ep:
            g.ep.eprob = g.new_ep("double")

        u = self.get_graph()
        libinference.collect_marginal(g._Graph__graph,
                                      u._Graph__graph,
                                      _prop("e", g, g.ep.count))
        g.gp.count += 1
        g.ep.eprob.fa = g.ep.count.fa
        g.ep.eprob.fa /= g.gp.count
        return g

    def collect_marginal_multigraph(self, g=None):
        r"""Collect marginal latent multigraph during MCMC runs.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph` (optional, default: ``None``)
            Previous marginal multigraph.

        Returns
        -------
        g : :class:`~graph_tool.Graph`
            New marginal graph, with internal edge
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

        """

        if g is None:
            g = Graph(directed=self.g.is_directed())
            g.add_vertex(self.g.num_vertices())
            g.ep.w = g.new_ep("vector<int>")
            g.ep.wcount = g.new_ep("vector<int>")

        libinference.collect_marginal_count(g._Graph__graph,
                                            self.u._Graph__graph,
                                            _prop("e", self.u, self.eweight),
                                            _prop("e", g, g.ep.w),
                                            _prop("e", g, g.ep.wcount))
        return g

class UncertainBlockState(UncertainBaseState):
    r"""Inference state of an uncertain graph, using the stochastic block model as a
    prior.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Measured graph.
    q : :class:`~graph_tool.EdgePropertyMap`
        Edge probabilities in range :math:`[0,1]`.
    q_default : ``float`` (optional, default: ``0.``)
        Non-edge probability in range :math:`[0,1]`.
    nested : ``boolean`` (optional, default: ``True``)
        If ``True``, a :class:`~graph_tool.inference.NestedBlockState`
        will be used, otherwise
        :class:`~graph_tool.inference.BlockState`.
    state_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to
        :class:`~graph_tool.inference.NestedBlockState` or
        :class:`~graph_tool.inference.BlockState`.
    bstate : :class:`~graph_tool.inference.NestedBlockState` or :class:`~graph_tool.inference.BlockState` (optional, default: ``None``)
        If passed, this will be used to initialize the block state
        directly.
    self_loops : bool (optional, default: ``False``)
        If ``True``, it is assumed that the uncertain graph can contain
        self-loops.

    References
    ----------
    .. [peixoto-reconstructing-2018] Tiago P. Peixoto, "Reconstructing
       networks with unknown and heterogeneous errors", Phys. Rev. X 8
       041011 (2018). :doi:`10.1103/PhysRevX.8.041011`, :arxiv:`1806.07956`
    """

    def __init__(self, g, q, q_default=0., nested=True, state_args={},
                 bstate=None, self_loops=False, **kwargs):

        super().__init__(g, nested=nested, state_args=state_args, bstate=bstate,
                         self_loops=self_loops, **kwargs)
        self._q = q
        self._q_default = q_default

        self.p = (q.fa.sum() + (self.M - g.num_edges()) * q_default) / self.M

        self.q = self.g.new_ep("double", vals=log(q.fa) - log1p(-q.fa))
        self.q.fa -= log(self.p) - log1p(-self.p)
        if q_default > 0:
            self.q_default = log(q_default) - log1p(q_default)
            self.q_default -= log(self.p) - log1p(-self.p)
        else:
            self.q_default = -numpy.inf

        self.S_const = (log1p(-q.fa[q.fa<1]).sum() +
                        log1p(-q_default) * (self.M - self.g.num_edges())
                        - self.M * log1p(-self.p))

        self._state = libinference.make_uncertain_state(self.bstate._state,
                                                        self)
    def __getstate__(self):
        state = super().__getstate__()
        return dict(state,  q=self._q, q_default=self._q_default)

    def __repr__(self):
        return "<UncertainBlockState object with %s, at 0x%x>" % \
            (self.nbstate if self.nbstate is not None else self.bstate,
             id(self))

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_uncertain_sweep(mcmc_state,
                                                 self._state,
                                                 _get_rng())

class LatentMultigraphBlockState(UncertainBaseState):
    r"""Inference state of an erased Poisson multigraph, using the stochastic
    block model as a prior.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Measured graph.
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
    self_loops : bool (optional, default: ``False``)
        If ``True``, it is assumed that the uncertain graph can contain
        self-loops.

    References
    ----------
    .. [peixoto-latent-2020] Tiago P. Peixoto, "Latent Poisson models for
       networks with heterogeneous density", Phys. Rev. E 102 012309 (2020)
       :doi:`10.1103/PhysRevE.102.012309`, :arxiv:`2002.07803`
    """

    def __init__(self, g, nested=True, state_args={},
                 bstate=None, self_loops=False, **kwargs):

        super().__init__(g, nested=nested, state_args=state_args, bstate=bstate,
                         self_loops=self_loops, **kwargs)

        self.q = self.g.new_ep("double", val=numpy.inf)
        self.q_default = -numpy.inf
        self.S_const = 0

        self._state = libinference.make_uncertain_state(self.bstate._state,
                                                        self)
    def __repr__(self):
        return "<LatentMultigraphBlockState object with %s, at 0x%x>" % \
            (self.nbstate if self.nbstate is not None else self.bstate,
             id(self))

    def _mcmc_sweep(self, mcmc_state):
        mcmc_state.edges_only = True
        return libinference.mcmc_uncertain_sweep(mcmc_state,
                                                 self._state,
                                                 _get_rng())

class MeasuredBlockState(UncertainBaseState):
    r"""Inference state of a measured graph, using the stochastic block model as a
    prior.

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
    lp : ``float`` (optional, default: ``NaN``)
        Log-probability of missing edges (false negatives). If given as ``NaN``,
        it is assumed this is an unknown sampled from a Beta distribution, with
        hyperparameters given by ``fn_params`.  Otherwise the  values of
        ``fn_params`` are ignored.
    lq : ``float`` (optional, default: ``NaN``)
        Log-probability of spurious edges (false positives). If given as
        ``NaN``, it is assumed this is an unknown sampled from a Beta
        distribution, with hyperparameters given by ``fp_params`. Otherwise the
        values of ``fp_params`` are ignored.
    fn_params : ``dict`` (optional, default: ``dict(alpha=1, beta=1)``)
        Beta distribution hyperparameters for the probability of missing
        edges (false negatives).
    fp_params : ``dict`` (optional, default: ``dict(mu=1, nu=1)``)
        Beta distribution hyperparameters for the probability of spurious
        edges (false positives).
    nested : ``boolean`` (optional, default: ``True``)
        If ``True``, a :class:`~graph_tool.inference.NestedBlockState`
        will be used, otherwise
        :class:`~graph_tool.inference.BlockState`.
    state_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to
        :class:`~graph_tool.inference.NestedBlockState` or
        :class:`~graph_tool.inference.BlockState`.
    bstate : :class:`~graph_tool.inference.NestedBlockState` or :class:`~graph_tool.inference.BlockState` (optional, default: ``None``)
        If passed, this will be used to initialize the block state
        directly.
    self_loops : bool (optional, default: ``False``)
        If ``True``, it is assumed that the uncertain graph can contain
        self-loops.

    References
    ----------
    .. [peixoto-reconstructing-2018] Tiago P. Peixoto, "Reconstructing
       networks with unknown and heterogeneous errors", Phys. Rev. X 8
       041011 (2018). :doi:`10.1103/PhysRevX.8.041011`, :arxiv:`1806.07956`

    """

    def __init__(self, g, n, x, n_default=1, x_default=0, lp=numpy.nan,
                 lq=numpy.nan, fn_params=dict(alpha=1, beta=1),
                 fp_params=dict(mu=1, nu=1), nested=True,
                 state_args={}, bstate=None, self_loops=False, **kwargs):

        super().__init__(g, nested=nested, state_args=state_args, bstate=bstate,
                         **kwargs)

        self.n = n
        self.x = x
        self.n_default = n_default
        self.x_default = x_default
        self.alpha = fn_params.get("alpha", 1)
        self.beta = fn_params.get("beta", 1)
        self.mu = fp_params.get("mu", 1)
        self.nu = fp_params.get("nu", 1)
        self.lp = lp
        self.lq = lq

        self._state = libinference.make_measured_state(self.bstate._state,
                                                       self)

    def __getstate__(self):
        state = super().__getstate__()
        return dict(state, n=self.n, x=self.x, n_default=self.n_default,
                    x_default=self.x_default,
                    fn_params=dict(alpha=self.alpha, beta=self.beta),
                    fp_params=dict(mu=self.mu, nu=self.nu), lp=self.lp,
                    lq=self.lq)

    def __repr__(self):
        return "<MeasuredBlockState object with %s, at 0x%x>" % \
            (self.nbstate if self.nbstate is not None else self.bstate,
             id(self))

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_measured_sweep(mcmc_state,
                                                self._state,
                                                _get_rng())

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

class MixedMeasuredBlockState(UncertainBaseState):
    r"""Inference state of a measured graph with heterogeneous errors, using the
    stochastic block model as a prior.

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
    x_default : ``int`` (optional, default: ``1``)
        Total number of positive measurements for each non-edge.
    fn_params : ``dict`` (optional, default: ``dict(alpha=1, beta=10)``)
        Beta distribution hyperparameters for the probability of missing
        edges (false negatives).
    fp_params : ``dict`` (optional, default: ``dict(mu=1, nu=10)``)
        Beta distribution hyperparameters for the probability of spurious
        edges (false positives).
    nested : ``boolean`` (optional, default: ``True``)
        If ``True``, a :class:`~graph_tool.inference.NestedBlockState`
        will be used, otherwise
        :class:`~graph_tool.inference.BlockState`.
    state_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to
        :class:`~graph_tool.inference.NestedBlockState` or
        :class:`~graph_tool.inference.BlockState`.
    bstate : :class:`~graph_tool.inference.NestedBlockState` or :class:`~graph_tool.inference.BlockState` (optional, default: ``None``)
        If passed, this will be used to initialize the block state
        directly.
    self_loops : bool (optional, default: ``False``)
        If ``True``, it is assumed that the uncertain graph can contain
        self-loops.

    References
    ----------
    .. [peixoto-reconstructing-2018] Tiago P. Peixoto, "Reconstructing
       networks with unknown and heterogeneous errors", Phys. Rev. X 8
       041011 (2018). :doi:`10.1103/PhysRevX.8.041011`, :arxiv:`1806.07956`
    """

    def __init__(self, g, n, x, n_default=1, x_default=0,
                 fn_params=dict(alpha=1, beta=10), fp_params=dict(mu=1, nu=10),
                 nested=True, state_args={}, bstate=None,
                 self_loops=False, **kwargs):

        super().__init__(g, nested=nested, state_args=state_args, bstate=bstate,
                         **kwargs)
        self.n = n
        self.x = x
        self.n_default = n_default
        self.x_default = x_default
        self.alpha = fn_params.get("alpha", 1)
        self.beta = fn_params.get("beta", 10)
        self.mu = fp_params.get("mu", 1)
        self.nu = fp_params.get("nu", 10)

        self._state = None

        self.q = self.g.new_ep("double")
        self.sync_q()

        self._state = libinference.make_uncertain_state(self.bstate._state,
                                                        self)

    def sync_q(self):
        ra, rb = self.transform(self.n.fa, self.x.fa)
        self.q.fa = ra - rb
        dra, drb = self.transform(self.n_default, self.x_default)
        self.q_default = dra - drb

        self.S_const = (self.M - self.g.num_edges()) * drb + rb.sum()

        if self._state is not None:
            self._state.set_q_default(self.q_default)
            self._state.set_S_const(self.S_const)

    def transform(self, na, xa):
        ra = (scipy.special.betaln(na - xa + self.alpha, xa + self.beta) -
              scipy.special.betaln(self.alpha, self.beta))
        rb = (scipy.special.betaln(xa + self.mu, na - xa + self.nu) -
              scipy.special.betaln(self.mu, self.nu))
        return ra, rb

    def set_hparams(self, alpha, beta, mu, nu):
        """Set edge and non-edge hyperparameters."""
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nu = nu
        self.sync_q()

    def __getstate__(self):
        state = super().__getstate__()
        return dict(state, n=self.n, x=self.x, n_default=self.n_default,
                    x_default=self.x_default,
                    fn_params=dict(alpha=self.alpha, beta=self.beta),
                    fp_params=dict(mu=self.mu, nu=self.nu))

    def __repr__(self):
        return "<MixedMeasuredBlockState object with %s, at 0x%x>" % \
            (self.nbstate if self.nbstate is not None else self.bstate,
             id(self))

    @mcmc_sweep_wrap
    def h_mcmc_step(self, hstep=1, **kwargs):
        dS = nt = nm = 0
        niter = kwargs.pop("niter", 1)
        latent_edges = kwargs.pop("entropy_args", {}).get("latent_edges", True)
        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        for i in range(niter):
            hs = [self.alpha, self.beta, self.mu, self.nu]
            j = numpy.random.randint(len(hs))

            f_dh = [max((0, hs[j] - hstep)), hs[j] + hstep]
            pf = 1./(f_dh[1] - f_dh[0])

            old_hs = hs[j]
            hs[j] = f_dh[0] + numpy.random.random() * (f_dh[1] - f_dh[0])

            b_dh = [max((0, hs[j] - hstep)), hs[j] + hstep]
            pb = 1./min((1, hs[j]))

            density = False

            ea = self._gen_eargs(dict(latent_edges=latent_edges,
                                      density=density))
            Sb = self._state.entropy(ea)
            self.set_hparams(*hs)
            Sa = self._state.entropy(ea)

            nt += 1
            if Sa < Sb or numpy.random.random() < exp(-(Sa-Sb) + log(pb) - log(pf)):
                dS += Sa - Sb
                nm +=1
            else:
                hs[j] = old_hs
                self.set_hparams(*hs)
        return (dS, nt, nm)

    def mcmc_sweep(self, pedges=.5, ph=.1, hstep=1, multiflip=True, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection sampling MCMC to
        sample network partitions and latent edges. The parameter ``pedges``
        controls the probability with which edge move will be attempted, instead
        of partition moves. The parameter ``ph`` controls the relative
        probability with which hyperparamters moves will be attempted, and
        ``hstep`` is the size of the step.

        The remaining keyword parameters will be passed to
        :meth:`~graph_tool.inference.BlockState.mcmc_sweep` or
        :meth:`~graph_tool.inference.BlockState.multiflip_mcmc_sweep`,
        if ``multiflip=True``.

        """

        if numpy.random.random() < ph:
            return self.h_mcmc_step(hstep=hstep, niter=kwargs.get("niter", 1),
                                    entropy_args=kwargs.get("entropy_args", {}))
        else:
            return super().mcmc_sweep(pedges=pedges, multiflip=multiflip,
                                      **kwargs)

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_uncertain_sweep(mcmc_state,
                                                 self._state,
                                                 _get_rng())


def marginal_multigraph_entropy(g, ecount):
    r"""Compute the entropy of the marginal latent multigraph distribution.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Marginal multigraph.
    ecount : :class:`~graph_tool.EdgePropertyMap`
        Vector-valued edge property map containing the counts of edge
        multiplicities.

    Returns
    -------
    eh : :class:`~graph_tool.EdgePropertyMap`
        Marginal entropy of edge multiplicities.

    Notes
    -----

    The mean posterior marginal multiplicity distribution of a multi-edge
    :math:`(i,j)` is defined as

    .. math::

       \pi_{ij}(w) = \sum_{\boldsymbol G}\delta_{w,G_{ij}}P(\boldsymbol G|\boldsymbol D)

    where :math:`P(\boldsymbol G|\boldsymbol D)` is the posterior
    probability of a multigraph :math:`\boldsymbol G` given the data.

    The corresponding entropy is therefore given (in nats) by

    .. math::

       \mathcal{S}_{ij} = -\sum_w\pi_{ij}(w)\ln \pi_{ij}(w).

    """

    eh = g.new_ep("double")
    libinference.marginal_count_entropy(g._Graph__graph,
                                        _prop("e", g, ecount),
                                        _prop("e", g, eh))
    return eh

def marginal_multigraph_sample(g, ews, ecount):

    w = g.new_ep("int")
    libinference.marginal_multigraph_sample(g._Graph__graph,
                                                _prop("e", g, ews),
                                                _prop("e", g, ecount),
                                                _prop("e", g, w),
                                                _get_rng())
    return w

def marginal_multigraph_lprob(g, ews, ecount, ew):

    L = libinference.marginal_multigraph_lprob(g._Graph__graph,
                                               _prop("e", g, ews),
                                               _prop("e", g, ecount),
                                               _prop("e", g, ew))
    return L

def marginal_graph_sample(g, ep):
    w = g.new_ep("int")
    libinference.marginal_graph_sample(g._Graph__graph,
                                       _prop("e", g, ep),
                                       _prop("e", g, w),
                                       _get_rng())
    return w

def marginal_graph_lprob(g, ep, w):
    L = libinference.marginal_graph_lprob(g._Graph__graph,
                                          _prop("e", g, ep),
                                          _prop("e", g, w))
    return L
