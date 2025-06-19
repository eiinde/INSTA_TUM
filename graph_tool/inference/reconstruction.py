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

from .. import _prop, _get_rng, Vector_int32_t

from .. spectral import adjacency
from .. generation import complete_graph, generate_triadic_closure, \
    remove_parallel_edges
from .. dynamics import IsingGlauberState, NormalState, IsingBPState, \
    NormalBPState, LinearNormalState, LVState

from . uncertain_blockmodel import *
from . util import pmap

from abc import ABC
import inspect
import numpy as np
import numbers

import scipy.stats

@entropy_state_signature
class DynamicsBlockStateBase(UncertainBaseState):
    def __init__(self, s, g=None, t=[], x=1, x_range=(-np.inf, np.inf), theta=0,
                 theta_range=(-np.inf, np.inf), disable_xdist=False,
                 disable_tdist=False, nested=True, state_args={}, bstate=None,
                 self_loops=False, max_m=1 << 16, nmax_extend=8,
                 entropy_args={}, **kwargs):
        r"""Base state for network reconstruction based on dynamical or
        behavioral data, using the stochastic block model as a prior. This class
        is not meant to be instantiated directly, only indirectly via one of its
        subclasses. Nevertheless, its paramteres are inherited, and are
        documented as follows.

        Parameters
        ----------
        s : :class:`~numpy.ndarray` of shape ``(N,M)`` or ``list`` of :class:`~graph_tool.VertexPropertyMap` or :class:`~graph_tool.VertexPropertyMap`
            Time series or independent samples used for reconstruction.

            If the type is :class:`~numpy.ndarray`, it should correspond to a
            ``(N,M)`` data matrix with ``M`` samples for all ``N`` nodes.

            If the parameter ``g`` is provided, this can be optionally a list of
            of :class:`~graph_tool.VertexPropertyMap` objects, where each entry
            in this list must be a :class:`~graph_tool.VertexPropertyMap` with
            type ``vector<int>`` or ``vector<double>``, depending on wether the
            model has discrete or continuous state values. If a single property
            map is given, then a single time series is assumed.

            If the parameter ``t`` below is given, each property map value for a
            given node should contain only the states for the same points in
            time given by that parameter.
        g : :class:`~graph_tool.Graph` (optional, default: ``None``)
            Initial graph state. If not provided, an empty graph will be assumed.
        t : ``list`` of :class:`~graph_tool.VertexPropertyMap` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``[]``)
            If nonempty, this allows for a compressed representation of the
            time-series parameter ``s``, corresponding only to points in time
            where the state of each node changes. Each entry in this list
            must be a :class:`~graph_tool.VertexPropertyMap` with type
            ``vector<int>`` containing the points in time where the state of
            each node change. The corresponding state of the nodes at these
            times are given by parameter ``s``. If a single property map is
            given, then a single time series is assumed.
        x : :class:`~graph_tool.EdgePropertyMap` or ``float`` (optional, default: ``1.``)
            Initial value of the edge weights. If a ``float`` is given, the edge
            weights will be assumed to be the same for all edges.
        x_range : pair of ``float``s (optional, default: ``(-np.inf, np.inf)``)
            Determines the allowed range of edge weights.
        theta : :class:`~graph_tool.EdgePropertyMap` or ``float`` (optional, default: ``0.``)
            Initial value of the node parameters. If a ``float`` is given, the
            node weights will be assumed to be the same for all edges.
        theta_range : pair of ``float``s (optional, default: ``(-np.inf, np.inf)``)
            Determines the allowed range of the node parameters.
        disable_xdist : ``boolean`` (optional, default: ``False``)
            If ``True`` the, quantization of the edge weights will be turned off,
            and :math:`L_1` regularization will be used instead.
        disable_tdist : ``boolean`` (optional, default: ``False``)
            If ``True`` the, quantization of the node parameters will be turned off,
            and :math:`L_1` regularization will be used instead.
        nested : ``boolean`` (optional, default: ``True``)
            If ``True``, a :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState`
            will be used, otherwise
            :class:`~graph_tool.inference.blockmodel.BlockState`.
        state_args : ``dict`` (optional, default: ``{}``)
            Arguments to be passed to
            :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or
            :class:`~graph_tool.inference.blockmodel.BlockState`.
        bstate : :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or :class:`~graph_tool.inference.blockmodel.BlockState` (optional, default: ``None``)
            If passed, this will be used to initialize the block state
            directly.
        self_loops : bool (optional, default: ``False``)
            If ``True``, it is assumed that the inferred graph can contain
            self-loops.
        state_args : ``int`` (optional, default: ``1 << 16``)
            Maximum latent edge multiplicity allowed.
        nmax_extend : ``int`` (optional, default: ``8``)
            Maximum number of range expansions during bisection search.
        entropy_args: ``dict`` (optional, default: ``{}``)
            Override default arguments for
            :meth:`~DynamicsBlockStateBase.entropy()` method and releated
            operations.
        """

        directed = kwargs.pop("directed", None)
        if g is None:
            if directed is None:
                directed = False
            g = Graph(s.shape[0], directed=directed)
        elif directed is None:
            directed = g.is_directed()
        if g.is_directed() != directed:
            g = GraphView(g, directed=directed)

        if isinstance(s, np.ndarray):
            if kwargs.pop("discrete", False):
                s = g.new_vp("vector<int>", vals=s)
            else:
                s = g.new_vp("vector<double>", vals=s)

        UncertainBaseState.__init__(self, g, nested=nested,
                                    state_args=state_args, bstate=bstate,
                                    self_loops=self_loops,
                                    entropy_args=entropy_args)

        if isinstance(s, PropertyMap):
            s = [s]
        if isinstance(t, PropertyMap):
            t = [t]

        self.s = [self.u.own_property(x) for x in s]
        self.t = [self.u.own_property(x) for x in t]

        if x is None:
            x = self.u.new_ep("double")
        elif isinstance(x, EdgePropertyMap):
            x = self.u.copy_property(x, g=x.get_graph(), value_type="double")
        else:
            x = self.u.new_ep("double", val=x)

        self.x = x
        self.xmin_bound = x_range[0]
        self.xmax_bound = x_range[1]

        if theta is None:
            theta = self.u.new_vp("double")
        elif isinstance(theta, VertexPropertyMap):
            theta = self.u.copy_property(theta, g=theta.get_graph(),
                                         value_type="double")
        else:
            theta = self.u.new_vp("double", val=theta)

        self.theta = theta
        self.tmin_bound = theta_range[0]
        self.tmax_bound = theta_range[1]

        self.disable_xdist = disable_xdist
        self.disable_tdist = disable_tdist

        if disable_xdist:
            self.update_entropy_args(xdist=False,
                                     sbm=entropy_args.get("sbm", False),
                                     delta=entropy_args.get("delta", 0))
        if disable_tdist:
            self.update_entropy_args(tdist=False)

        kwargs = kwargs.copy()
        for k in kwargs.keys():
            v = kwargs[k]
            if isinstance(v, PropertyMap):
                kwargs[k] = g.own_property(v)
            elif (isinstance(v, collections.abc.Iterable) and len(v) > 0 and
                  isinstance(v[0], PropertyMap)):
                kwargs[k] = [g.own_property(x) for x in v]

        self.rg = kwargs.pop("rg", None)
        self.elist = kwargs.pop("elist", None)

        self.params = kwargs
        self.os = [ns._get_any() for ns in self.s]
        self.ot = [nt._get_any() for nt in self.t]
        self.max_m = max_m
        self.nmax_extend = nmax_extend

        self._state = libinference.make_dynamics_state(self.bstate._state, self)
        self._dstate = self._make_state()
        self._state.set_dstate(self._dstate)

    def set_params(self, params):
        r"""Sets the model parameters via the dictionary ``params``."""
        self.params = dict(self.params, **params)
        self._state.set_params(self.params)

    def get_params(self, params):
        r"""Gets the model parameters via the dictionary ``params``."""
        return dict(self.params)

    def __getstate__(self):
        state = UncertainBaseState.__getstate__(self)
        return dict(state, s=self.s, t=self.t, x=self.x,
                    x_range=(self.xmin_bound, self.xmax_bound),
                    theta=self.theta,
                    theta_range=(self.tmin_bound, self.tmax_bound),
                    disable_xdist=self.disable_xdist,
                    disable_tdist=self.disable_tdist,
                    rg=self.rg, elist=self.elist, max_m=self.max_m,
                    **self.params)

    def __repr__(self):
        return "<%s object with %s, %d edge categories and %d node categories, at 0x%x>" % \
            (self.__class__.__name__,
             self.nbstate if self.nbstate is not None else self.bstate,
             len(self.get_xvals()), len(self.get_tvals()),
             id(self))

    def _gen_eargs(self, args):
        uea = UncertainBaseState._gen_eargs(self, args)
        return libinference.dentropy_args(uea)

    @copy_state_wrap
    def _entropy(self, latent_edges=True, density=False, aE=1, sbm=True,
                 xdist=True, tdist=True, xl1=1, tl1=1, alpha=1, delta=1e-8,
                 normal=False, mu=0, sigma=1, **kwargs):
        r"""Return the description length, i.e. the negative joint
        log-likelihood.

        Parameters
        ----------
        latent_edges : ``boolean`` (optional, default: ``True``)
            If ``True``, the adjacency term of the description length will be
            included.
        density : ``boolean`` (optional, default: ``False``)
            If ``True``, a geometric prior for the total number of edges will be
            included.
        aE : ``double`` (optional, default: ``1``)
            If ``density=True``, this will correspond to the expected number of
            edges according to the geometric prior.
        sbm : ``boolean`` (optional, default: ``True``)
            If ``True``, SBM description length will be included.
        xdist : ``boolean`` (optional, default: ``True``)
            If ``True``, the quantized edge weight distribution description
            length will be included.
        tdist : ``boolean`` (optional, default: ``True``)
            If ``True``, the quantized node parameter distribution description
            length will be included.
        xl1 : ``float`` (optional, default: ``1``)
            Specifies the :math:`\lambda` parameter for :math:`L_1`
            regularization for the edge weights if ``xdist == False``, or the
            Laplace hyperprior for the discrete categories if ``xdist == True``.
        tl1 : ``float`` (optional, default: ``1``)
            Specifies the :math:`\lambda` parameter for :math:`L_1`
            regularization for the node paraemters if ``tdist == False``, or the
            Laplace hyperprior for the discrete categories if ``tdist == True``.
        delta : ``float`` (optional, default: ``1e-8``)
            Specifies the precision parameter for the discrete categories.
        normal : ``boolean`` (optional, default: ``False``)
            If ``True``, a normal distribution will be used for the weight
            priors.
        mu : ``double`` (optional, default: ``0``)
            If ``normal == True``, this will be the mean of the normal distribution.
        sigma : ``double`` (optional, default: ``1``)
            If ``normal == True``, this will be the standard deviation of the
            normal distribution.

        Notes
        -----

        The "entropy" of the state is the negative log-likelihood of the
        generative model for the data :math:`\boldsymbol S`, that includes the
        inferred weighted adjacency matrix :math:`\boldsymbol{X}`, the node
        parameters :math:`\boldsymbol{\theta}`, and the SBM node partition
        :math:`\boldsymbol{b},` given by

        .. math::

           \begin{aligned}
           \Sigma(\boldsymbol{S},\boldsymbol{X},\boldsymbol{\theta}|\lambda_x,\lambda_{\theta},\Delta)
                  = &- \ln P(\boldsymbol{S}|\boldsymbol{X},\boldsymbol{\theta})\\
                    &- \ln P(\boldsymbol{X}|\boldsymbol{A},\lambda_x, \Delta)\\
                    &- \ln P(\boldsymbol{A},\boldsymbol{b})\\
                    &- \ln P(\boldsymbol{\theta}, \lambda_{\theta}, \Delta).
           \end{aligned}

        The term :math:`P(\boldsymbol{S}|\boldsymbol{X},\boldsymbol{\theta})` is
        given by the particular generative model being used and
        :math:`P(\boldsymbol{A},\boldsymbol{b})` by the SBM. The weight
        ditribution is given by the quantized model

        .. math::

           P(\boldsymbol X|\boldsymbol A,\lambda_x,\Delta) =
           \frac{\prod_{k}m_{k}!\times \mathrm{e}^{-\lambda_x \sum_k |z_k|}(\mathrm{e}^{\lambda\Delta} - 1)^{K}}
           {E!{E-1 \choose K-1}2^{K}\max(E,1)}

        where :math:`\boldsymbol z` are the :math:`K` discrete weight
        categories, and analogously

        .. math::

             P(\boldsymbol\theta|\lambda_{\theta},\Delta)
               =\frac{\prod_{k}n_{k}!\times
                      \mathrm{e}^{-\lambda \sum_k |u_k|}
                      \sinh(\lambda_{\theta}\Delta)^{K_{\theta}-\mathbb{1}_{0\in\boldsymbol u}}
                      (1-\mathrm{e}^{-\lambda_{\theta}\Delta})^{\mathbb{1}_{0\in\boldsymbol u}}}
                     {N!{N-1 \choose K_{\theta}-1}N},

        is the node parameter quantized distribution. For more details see
        [peixoto-network-2024]_.

        References
        ----------
        .. [peixoto-network-2024] Tiago P. Peixoto, "Network reconstruction via
           the minimum description length principle", :arxiv:`2405.01015`
        .. [peixoto-scalable-2024] Tiago P. Peixoto, "Scalable network
           reconstruction in subquadratic time", :arxiv:`2401.01404`
        """

        eargs = self._get_entropy_args(locals())
        S = self._state.entropy(eargs)
        if sbm:
            if self.nbstate is None:
                S += self.bstate.entropy(**kwargs)
            else:
                S += self.nbstate.entropy(**kwargs)
        return S

    def _get_elist(self, k, elist_args={}):
        elist_args = elist_args.copy()
        if self.elist is not None and self.elist[0] == k:
            return self.elist[1]
        elist, w = self.get_candidate_edges(k, **elist_args)
        self.elist = (k, elist, w)
        return elist

    def mcmc_sweep(self, beta=np.inf, niter=1, edge=True, edge_swap=True,
                   edge_multiflip=True, theta=True, theta_multiflip=True,
                   sbm=True, xvals=True, tvals=True, k=1, keep_elist=False,
                   verbose=False, elist_args={}, edge_mcmc_args={},
                   edge_swap_mcmc_args={}, edge_multiflip_mcmc_args={},
                   xvals_mcmc_args={}, theta_mcmc_args={},
                   theta_multiflip_mcmc_args={}, tvals_mcmc_args={},
                   sbm_mcmc_args={}, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection
        sampling MCMC to sample latent edges and network partitions.

        Parameters
        ----------
        beta : ``float`` (optional, default: ``np.inf``)
            Inverse temperature parameter.
        niter : ``int`` (optional, default: ``1``)
            Number of sweeps.
        k : ``int`` (optional, default: ``1``)
            :math:`\kappa` parameter to be passed to :meth:`~.DynamicsBlockStateBase.get_candidate_edges`.
        elist_args : ``dict`` (optiona, default: ``{}``)
            Paramters to pass to call :meth:`~.DynamicsBlockStateBase.get_candidate_edges`.
        keep_elist : ``boolean`` (optional, default: ``False``)
            If ``True``, the candidate edge list from last call will be re-used
            (if it exists).
        edge : ``boolean`` (optiona, default: ``True``)
            Whether to call :meth:`~.DynamicsBlockStateBase.edge_mcmc_sweep`.
        edge_mcmc_args : ``dict`` (optiona, default: ``{}``)
            Paramters to pass to call :meth:`~.DynamicsBlockStateBase.edge_mcmc_sweep`.
        edge_swap : ``boolean`` (optiona, default: ``True``)
            Whether to call :meth:`~.DynamicsBlockStateBase.swap_mcmc_sweep`.
        edge_mcmc_args : ``dict`` (optiona, default: ``{}``)
            Paramters to pass to call :meth:`~.DynamicsBlockStateBase.swap_mcmc_sweep`.
        edge_multiflip : ``boolean`` (optiona, default: ``True``)
            Whether to call :meth:`~.DynamicsBlockStateBase.edge_multiflip_mcmc_sweep`.
        edge_multiflip_mcmc_args : ``dict`` (optiona, default: ``{}``)
            Paramters to pass to call :meth:`~.DynamicsBlockStateBase.edge_multiflip_mcmc_sweep`.
        theta : ``boolean`` (optiona, default: ``True``)
            Whether to call :meth:`~.DynamicsBlockStateBase.theta_mcmc_sweep`.
        theta_mcmc_args : ``dict`` (optiona, default: ``{}``)
            Paramters to pass to call :meth:`~.DynamicsBlockStateBase.theta_mcmc_sweep`.
        theta_multiflip : ``boolean`` (optiona, default: ``True``)
            Whether to call :meth:`~.DynamicsBlockStateBase.theta_multiflip_mcmc_sweep`.
        theta_multiflip_mcmc_args : ``dict`` (optiona, default: ``{}``)
            Paramters to pass to call :meth:`~.DynamicsBlockStateBase.theta_multiflip_mcmc_sweep`.
        sbm : ``boolean`` (optiona, default: ``True``)
            Whether to call :meth:`~.DynamicsBlockStateBase.sbm_mcmc_sweep`.
        sbm_mcmc_args : ``dict`` (optiona, default: ``{}``)
            Paramters to pass to call :meth:`~.DynamicsBlockStateBase.sbm_mcmc_sweep`.
        xvals : ``boolean`` (optiona, default: ``True``)
            Whether to call :meth:`~.DynamicsBlockStateBase.xvals_sweep`.
        xvals_mcmc_args : ``dict`` (optiona, default: ``{}``)
            Paramters to pass to call :meth:`~.DynamicsBlockStateBase.xvals_sweep`.
        tvals : ``boolean`` (optiona, default: ``True``)
            Whether to call :meth:`~.DynamicsBlockStateBase.tvals_sweep`.
        tvals_mcmc_args : ``dict`` (optiona, default: ``{}``)
            Paramters to pass to call :meth:`~.DynamicsBlockStateBase.tvals_sweep`.
        verbose : ``boolean`` (optional, default: ``False``)
            If ``verbose == True``, detailed information will be displayed.
        **kwargs : ``dict`` (optional, default: ``{}``)
            Remaining keyword parameters will be passed to all individual MCMC
            functions.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nmoves : ``int``
            Number of variables moved.

        """

        ret = (0, 0, 0)
        if self.tmax_bound != self.tmin_bound:
            if theta:
                if verbose:
                    print("theta_mcmc_sweep:")
                tret = self.theta_mcmc_sweep(**dict(dict(beta=beta, niter=niter,
                                                         **kwargs), **theta_mcmc_args))
                ret = (sum(x) for x in zip(ret, tret))
            if not self.disable_tdist:
                if theta_multiflip:
                    if verbose:
                        print("theta_multiflip_mcmc_sweep:")
                    tret = self.theta_multiflip_mcmc_sweep(**dict(dict(beta=beta,
                                                                       niter=10 * niter,
                                                                       **kwargs),
                                                                  **theta_multiflip_mcmc_args))
                    ret = (sum(x) for x in zip(ret, tret))
                if tvals:
                    if verbose:
                        print("tvals_sweep:")
                    eret = self.tvals_sweep(**dict(dict(beta=beta,
                                                        niter=10 * niter,
                                                        **kwargs),
                                                   **tvals_mcmc_args))
                    ret = (sum(x) for x in zip(ret, eret))
        if edge:
            if verbose:
                print("edge_mcmc_sweep:")
            edge_mcmc_args = dict(edge_mcmc_args, k=k, keep_elist=keep_elist,
                                  elist_args=elist_args)
            eret = self.edge_mcmc_sweep(**dict(dict(beta=beta, niter=niter,
                                                    **kwargs), **edge_mcmc_args))
            ret = (sum(x) for x in zip(ret, eret))
            keep_elist = True
        if edge_swap:
            if verbose:
                print("swap_mcmc_sweep:")
            edge_swap_mcmc_args = dict(edge_swap_mcmc_args, k=k,
                                       keep_elist=keep_elist,
                                       elist_args=elist_args)
            eret = self.swap_mcmc_sweep(**dict(dict(beta=beta, niter=niter,
                                                    **kwargs), **edge_swap_mcmc_args))
            ret = (sum(x) for x in zip(ret, eret))
            keep_elist = True
        if self.xmax_bound != self.xmin_bound and not self.disable_xdist:
            if edge_multiflip:
                if verbose:
                    print("edge_multiflip_sweep:")
                eret = self.edge_multiflip_mcmc_sweep(**dict(dict(beta=beta,
                                                                  niter=10 * niter,
                                                                  **kwargs),
                                                             **edge_multiflip_mcmc_args))
                ret = (sum(x) for x in zip(ret, eret))
            if xvals:
                if verbose:
                    print("xvals_sweep:")
                eret = self.xvals_sweep(**dict(dict(beta=beta, niter=10 * niter,
                                                    **kwargs),
                                               **xvals_mcmc_args))
                ret = (sum(x) for x in zip(ret, eret))
        if sbm:
            if verbose:
                print("sbm_mcmc_sweep:")
            kwargs = kwargs.copy()
            eargs = kwargs.get("entropy_args", {}).copy()
            for k in list(eargs.keys()):
                if k not in self.bstate._entropy_args:
                    eargs.pop(k, None)
            kwargs["entropy_args"] = eargs
            sret = self.sbm_mcmc_sweep(**dict(dict(beta=beta, niter=niter,
                                                   **kwargs), **sbm_mcmc_args))
            ret = (sum(x) for x in zip(ret, sret))
        return tuple(ret)

    @mcmc_sweep_wrap
    def edge_mcmc_sweep(self, beta=np.inf, niter=1, k=1, elist_args={},
                        keep_elist=False, pold=1, pnew=1, pxu=.1, pm=1,
                        premove=1, maxiter=0, tol=1e-7, binary=True,
                        deterministic=False, sequential=True, parallel=True,
                        verbose=False, entropy_args={}, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection
        sampling MCMC to sample latent edges.

        Parameters
        ----------
        beta : ``float`` (optional, default: ``np.inf``)
            Inverse temperature parameter.
        niter : ``int`` (optional, default: ``1``)
            Number of sweeps.
        k : ``int`` (optional, default: ``1``)
            :math:`\kappa` parameter to be passed to :meth:`~.DynamicsBlockStateBase.get_candidate_edges`.
        elist_args : ``dict`` (optional, default: ``{}``)
            Paramters to pass to call :meth:`~.DynamicsBlockStateBase.get_candidate_edges`.
        keep_elist : ``boolean`` (optional, default: ``False``)
            If ``True``, the candidate edge list from last call will be re-used
            (if it exists).
        pold : ``float`` (optional, default: ``1``)
            Relative probability of proposing a new edge weight from existing
            categories.
        pnew : ``float`` (optional, default: ``1``)
            Relative probability of proposing a new edge weight from a new
            categories.
        pxu : ``float`` (optional, default: ``.1``)
            Probability of choosing from an existing category uniformly at
            random (instead of doing a bisection search).
        pm : ``float`` (optional, default: ``1``)
            Relative probability of doing edge multiplicity updates.
        premove : ``float`` (optional, default: ``1``)
            Relative probability of removing edges.
        maxiter : ``int`` (optional, default: ``0``)
            Maximum number of iterations for bisection search (``0`` means unlimited).
        tol : ``float`` (optional, default: ``1e-7``)
            Tolerance for bisection search.
        binary : ``boolean`` (optional, default: ``True``)
            If ``True``, the latent graph will be assumed to be a simple graph,
            otherwise a multigraph.
        deterministic : ``boolean`` (optional, default: ``False``)
            If ``True``, the the order of edge updates will be determinisitc,
            otherwise uniformly at random.
        sequential : ``boolean`` (optional, default: ``True``)
            If ``True``, a sweep will visit every edge candidate once, otherwise
            individiual updates will be chosen at random.
        parallel : ``boolean`` (optional, default: ``True``)
            If ``True``, the updates are performed in parallel, using locks on
            edges candidate incident on the same node.
        verbose : ``boolean`` (optional, default: ``False``)
            If ``verbose == True``, detailed information will be displayed.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`~.DynamicsBlockStateBase.entropy`.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nmoves : ``int``
            Number of variables moved.
        """

        elist_args = dict(elist_args, entropy_args=entropy_args)
        entropy_args = self._get_entropy_args(entropy_args)
        if not keep_elist:
            self.elist = None
        elist = self._get_elist(k, elist_args)
        state = self._state
        if np.isinf(beta):
            pxu = 0
        mcmc_state = DictState(dict(kwargs, **locals()))
        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        if parallel:
            return self._state.pseudo_mcmc_sweep(mcmc_state, _get_rng())
        else:
            return self._state.mcmc_sweep(mcmc_state, _get_rng())

    @mcmc_sweep_wrap
    def swap_mcmc_sweep(self, beta=np.inf, niter=1, k=1, elist_args={},
                        keep_elist=False, pmove=1, ptmove=1, pswap=1,
                        deterministic=False, sequential=True, parallel=True,
                        verbose=False, entropy_args={}, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection
        sampling MCMC to swap edge endpoints.

        Parameters
        ----------
        beta : ``float`` (optional, default: ``np.inf``)
            Inverse temperature parameter.
        niter : ``int`` (optional, default: ``1``)
            Number of sweeps.
        k : ``int`` (optional, default: ``1``)
            :math:`\kappa` parameter to be passed to :meth:`~.DynamicsBlockStateBase.get_candidate_edges`.
        elist_args : ``dict`` (optional, default: ``{}``)
            Paramters to pass to call :meth:`~.DynamicsBlockStateBase.get_candidate_edges`.
        keep_elist : ``boolean`` (optional, default: ``False``)
            If ``True``, the candidate edge list from last call will be re-used
            (if it exists).
        pmove : ``float`` (optional, default: ``1``)
            Relative probability of swaping the weights between two randomly
            chosen edges.
        ptmove : ``float`` (optional, default: ``1``)
            Relative probability of moving a single edge endpoint of an edge
            with a candidate edge.
        pswap : ``float`` (optional, default: ``1``)
            Relative probability of swapping the endpoints of two randomly
            selected edges.
        deterministic : ``boolean`` (optional, default: ``False``)
            If ``True``, the the order of edge updates will be determinisitc,
            otherwise uniformly at random.
        sequential : ``boolean`` (optional, default: ``True``)
            If ``True``, a sweep will visit every edge candidate once, otherwise
            individiual updates will be chosen at random.
        parallel : ``boolean`` (optional, default: ``True``)
            If ``True``, the updates are performed in parallel, using locks on
            edges candidate incident on the same node.
        verbose : ``boolean`` (optional, default: ``False``)
            If ``verbose == True``, detailed information will be displayed.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`~.DynamicsBlockStateBase.entropy`.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nmoves : ``int``
            Number of variables moved.
        """

        elist_args = dict(elist_args, entropy_args=entropy_args)
        entropy_args = self._get_entropy_args(entropy_args)
        if not keep_elist:
            self.elist = None
        elist = self._get_elist(k, elist_args)
        state = self._state
        if parallel:
            sequential = True
        mcmc_state = DictState(dict(kwargs, **locals()))
        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        if parallel:
            return self._state.pseudo_swap_mcmc_sweep(mcmc_state, _get_rng())
        else:
            return self._state.swap_mcmc_sweep(mcmc_state, _get_rng())

    @mcmc_sweep_wrap
    def edge_multiflip_mcmc_sweep(self, beta=np.inf, niter=1, pmerge=1, psplit=1,
                                  pmergesplit=1, pmovelabel=1, gibbs_sweeps=1,
                                  c=.1, maxiter=0, tol=1e-7, accept_stats=None,
                                  verbose=False, entropy_args={}, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection
        merge-split MCMC to sample discrete edge weight categories.

        Parameters
        ----------
        beta : ``float`` (optional, default: ``np.inf``)
            Inverse temperature parameter.
        niter : ``int`` (optional, default: ``1``)
            Number of sweeps.
        pmerge : ``float`` (optional, default: ``1``)
            Relative probability of merging two discrete categories.
        psplit : ``float`` (optional, default: ``1``)
            Relative probability of splitting two discrete categories.
        pmergesplit : ``float`` (optional, default: ``1``)
            Relative probability of simultaneoulsly merging and splitting two
            discrete categories.
        pmovelabel : ``float`` (optional, default: ``1``)
            Relative probability of moving the value of a discrete category.
        gibbs_sweeps : ``int`` (optional, default: ``1``)
            Number of Gibbs sweeps performed to achieve a split proposal.
        c : ``double`` (optional, default: ``.1``)
            Probability of choosing a category uniformly at random to perform a
            merge, otherwise an adjacent one is chosen.
        maxiter : ``int`` (optional, default: ``0``)
            Maximum number of iterations for bisection search (``0`` means unlimited).
        tol : ``float`` (optional, default: ``1e-7``)
            Tolerance for bisection search.
        accept_stats : ``dict`` (optional, default: ``None``)
            If provided, the dictionary will be updated with acceptance statistics.
        verbose : ``boolean`` (optional, default: ``False``)
            If ``verbose == True``, detailed information will be displayed.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`~.DynamicsBlockStateBase.entropy`.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nmoves : ``int``
            Number of variables moved.
        """

        kwargs = dict(**kwargs)
        entropy_args = kwargs.pop("entropy_args", {})

        E = self.u.num_edges()
        if E == 0:
            return (0, 0, 0)
        niter /= E
        gibbs_sweeps = max((gibbs_sweeps, 1))
        nproposal = Vector_size_t(4)
        nacceptance = Vector_size_t(4)
        force_move = kwargs.pop("force_move", False)
        mcmc_state = DictState(locals())
        mcmc_state.entropy_args = self._get_entropy_args(entropy_args)
        mcmc_state.state = self._state

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))

        ret = self._state.multiflip_mcmc_sweep(mcmc_state, _get_rng())

        if accept_stats is not None:
            for key in ["proposal", "acceptance"]:
                if key not in accept_stats:
                    accept_stats[key] = np.zeros(len(nproposal),
                                                 dtype="uint64")
            accept_stats["proposal"] += nproposal.a
            accept_stats["acceptance"] += nacceptance.a

        return ret

    @mcmc_sweep_wrap
    def xvals_sweep(self, beta=np.inf, niter=100, maxiter=0, tol=1e-7,
                    min_size=1, entropy_args={}):
        r"""Perform sweeps of a greedy update on the edge weight category
        values, based on bisection search.

        Parameters
        ----------
        beta : ``float`` (optional, default: ``np.inf``)
            Inverse temperature parameter.
        niter : ``int`` (optional, default: ``100``)
            Number of categories to update.
        maxiter : ``int`` (optional, default: ``0``)
            Maximum number of iterations for bisection search (``0`` means unlimited).
        tol : ``float`` (optional, default: ``1e-7``)
            Tolerance for bisection search.
        min_size : ``int`` (optional, default: ``1``)
            Minimum size of edge categories that will be updated.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`~.DynamicsBlockStateBase.entropy`.
        verbose : ``boolean`` (optional, default: ``False``)
            If ``verbose == True``, detailed information will be displayed.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nmoves : ``int``
            Number of variables moved.
        """

        xvals = self.get_xvals()
        r = min(niter/len(xvals), 1) if len(xvals) > 0 else 1
        niter = max(1, int(round(niter/len(xvals)))) if len(xvals) > 0 else 1
        ea = self._get_entropy_args(entropy_args)
        ret = (0, 0)
        for i in range(niter):
            eret = self._state.xvals_sweep(beta, r, maxiter, tol,
                                           min_size, ea, _get_rng())
            ret = tuple(sum(x) for x in zip(ret, eret))
        return ret

    def get_x(self):
        """Return latent edge weights."""
        return self.x.copy()

    def get_xvals(self):
        """Return latent edge weight categories."""
        return self._state.get_xvals()

    @mcmc_sweep_wrap
    def theta_mcmc_sweep(self, beta=np.inf, niter=1, pold=1, pnew=1, maxiter=0,
                         tol=1e-7, deterministic=False, sequential=True,
                         parallel=True, verbose=False, entropy_args={}, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection
        sampling MCMC to sample node parameters.

        Parameters
        ----------
        beta : ``float`` (optional, default: ``np.inf``)
            Inverse temperature parameter.
        niter : ``int`` (optional, default: ``1``)
            Number of sweeps.
        pold : ``float`` (optional, default: ``1``)
            Relative probability of proposing a new node value from existing
            categories.
        pnew : ``float`` (optional, default: ``1``)
            Relative probability of proposing a new node value from a new
            categories.
        maxiter : ``int`` (optional, default: ``0``)
            Maximum number of iterations for bisection search (``0`` means unlimited).
        tol : ``float`` (optional, default: ``1e-7``)
            Tolerance for bisection search.
        deterministic : ``boolean`` (optional, default: ``False``)
            If ``True``, the the order of node updates will be determinisitc,
            otherwise uniformly at random.
        sequential : ``boolean`` (optional, default: ``True``)
            If ``True``, a sweep will visit every node once, otherwise
            individiual updates will be chosen at random.
        parallel : ``boolean`` (optional, default: ``True``)
            If ``True``, the updates are performed in parallel.
        verbose : ``boolean`` (optional, default: ``False``)
            If ``verbose == True``, detailed information will be displayed.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nmoves : ``int``
            Number of variables moved.
        """

        entropy_args = self._get_entropy_args(entropy_args)
        state = self._state
        mcmc_state = DictState(dict(kwargs, **locals()))
        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        if parallel:
            return self._state.pseudo_mcmc_theta_sweep(mcmc_state, _get_rng())
        else:
            return self._state.mcmc_theta_sweep(mcmc_state, _get_rng())

    @mcmc_sweep_wrap
    def theta_multiflip_mcmc_sweep(self, beta=np.inf, pmerge=1, psplit=1,
                                   pmergesplit=1, pmovelabel=0, gibbs_sweeps=1,
                                   c=.1, niter=1, maxiter=0, tol=1e-7,
                                   entropy_args={}, accept_stats=None,
                                   verbose=False, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection
        merge-split MCMC to sample discrete node value categories.

        Parameters
        ----------
        beta : ``float`` (optional, default: ``np.inf``)
            Inverse temperature parameter.
        niter : ``int`` (optional, default: ``1``)
            Number of sweeps.
        pmerge : ``float`` (optional, default: ``1``)
            Relative probability of merging two discrete categories.
        psplit : ``float`` (optional, default: ``1``)
            Relative probability of splitting two discrete categories.
        pmergesplit : ``float`` (optional, default: ``1``)
            Relative probability of simultaneoulsly merging and splitting two
            discrete categories.
        pmovelabel : ``float`` (optional, default: ``1``)
            Relative probability of moving the value of a discrete category.
        gibbs_sweeps : ``int`` (optional, default: ``1``)
            Number of Gibbs sweeps performed to achieve a split proposal.
        c : ``double`` (optional, default: ``.1``)
            Probability of choosing a category uniformly at random to perform a
            merge, otherwise an adjacent one is chosen.
        maxiter : ``int`` (optional, default: ``0``)
            Maximum number of iterations for bisection search (``0`` means unlimited).
        tol : ``float`` (optional, default: ``1e-7``)
            Tolerance for bisection search.
        accept_stats : ``dict`` (optional, default: ``None``)
            If provided, the dictionary will be updated with acceptance statistics.
        verbose : ``boolean`` (optional, default: ``False``)
            If ``verbose == True``, detailed information will be displayed.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nmoves : ``int``
            Number of variables moved.
        """

        niter /= self.u.num_vertices()
        gibbs_sweeps = max((gibbs_sweeps, 1))
        nproposal = Vector_size_t(4)
        nacceptance = Vector_size_t(4)
        force_move = kwargs.pop("force_move", False)
        mcmc_state = DictState(locals())
        mcmc_state.entropy_args = self._get_entropy_args(entropy_args)
        mcmc_state.state = self._state

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))

        ret = self._state.multiflip_mcmc_theta_sweep(mcmc_state, _get_rng())

        if accept_stats is not None:
            for key in ["proposal", "acceptance"]:
                if key not in accept_stats:
                    accept_stats[key] = np.zeros(len(nproposal),
                                                 dtype="uint64")
            accept_stats["proposal"] += nproposal.a
            accept_stats["acceptance"] += nacceptance.a

        return ret

    @mcmc_sweep_wrap
    def tvals_sweep(self, beta=np.inf, niter=100, maxiter=0, min_size=1,
                    tol=1e-7, entropy_args={}):
        r"""Perform sweeps of a greedy update on the node category
        values, based on bisection search.

        Parameters
        ----------
        beta : ``float`` (optional, default: ``np.inf``)
            Inverse temperature parameter.
        niter : ``int`` (optional, default: ``100``)
            Number of categories to update.
        maxiter : ``int`` (optional, default: ``0``)
            Maximum number of iterations for bisection search (``0`` means unlimited).
        tol : ``float`` (optional, default: ``1e-7``)
            Tolerance for bisection search.
        min_size : ``int`` (optional, default: ``1``)
            Minimum size of node categories that will be updated.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`~.DynamicsBlockStateBase.entropy`.
        verbose : ``boolean`` (optional, default: ``False``)
            If ``verbose == True``, detailed information will be displayed.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nmoves : ``int``
            Number of variables moved.
        """

        tvals = self.get_tvals()
        r = min(niter/len(tvals), 1) if len(tvals) > 0 else 1
        niter = max(1, int(round(niter/len(tvals)))) if len(tvals) > 0 else 1
        ea = self._get_entropy_args(entropy_args)
        ret = (0, 0)
        for i in range(niter):
            eret = self._state.tvals_sweep(beta, r, maxiter, tol,
                                           min_size, ea, _get_rng())
            ret = tuple(sum(x) for x in zip(ret, eret))
        return ret

    def get_theta(self):
        """Return latent node values."""
        return self.theta.copy()

    def get_tvals(self):
        """Return latent node categories."""
        return self._state.get_tvals()

    def get_edge_prob(self, u, v, x, entropy_args={}, epsilon=1e-8):
        r"""Return conditional posterior log-probability of edge :math:`(u,v)`."""
        ea = self._get_entropy_args(entropy_args)
        return self._state.get_edge_prob(u, v, x, ea, epsilon)

    def get_edges_prob(self, elist, entropy_args={}, epsilon=1e-8):
        r"""Return conditional posterior log-probability of an edge list, with
        shape :math:`(E,2)`."""
        ea = self._get_entropy_args(entropy_args)
        elist = np.asarray(elist, dtype="double")
        probs = np.zeros(elist.shape[0])
        self._state.get_edges_prob(elist, probs, ea, epsilon)
        return probs

    def edge_cov(self, u, v, toffset=True, pearson=False):
        r"""Return the covariance (or Pearson correlation if ``pearson ==
        True``) between nodes :math:`u` and :math:`v`, according to their
        time-series.
        """
        return self._state.node_cov(u, v, toffset, pearson)

    def edge_TE(self, u, v):
        r"""Return the transfer entropy between nodes :math:`u` and :math:`v`,
        according to their time-series.
        """
        return self._state.node_TE(u, v)

    def edge_MI(self, u, v):
        r"""Return the mutual information between nodes :math:`u` and :math:`v`,
        according to their time-series.
        """
        return self._state.node_MI(u, v)

    def get_candidate_edges(self, k=1, r=1, max_rk="k", epsilon=0.01,
                            c_stop=False, max_iter=0, knn=False, gradient=None,
                            h=1e-6, f_max_iter=10, tol=1e-6, allow_edges=False,
                            include_edges=True, use_hint=True, nrandom=0,
                            keep_all=False, exact=False, return_graph=False,
                            keep_iter=False, entropy_args={}, verbose=False):
        r"""Return the :math:`\lfloor\kappa N\rceil` best edge candidates
        according to a stochastic second neighbor search.

        Parameters
        ----------
        k : ``float`` (optional, default: ``1``)
            :math:`\kappa` parameter.
        r : ``float`` (optional, default: ``1``)
            Fraction of second neighbors to consider during the search.
        max_rk : ``float`` (optional, default: ``"k"``)
            Maximum number of second-neighbors to be considered per iteration. A
            string value ``"k"`` means that this will match the number of first
            neighbors.
        epsilon : ``float`` (optional, default: ``.01``)
            Convergence criterion.
        c_stop : ``boolean`` (optional, default: ``False``)
            If ``True``, the clustering coefficient will be used for the
            convergence criterion.
        max_iter : ``int`` (optional, default: ``0``)
            Maximum number of iterations allowed (``0`` means unlimited).
        knn : ``boolean`` (optional, default: ``False``)
            If ``True``, the KNN graph will be returned.
        gradient : ``boolean`` (optional, default: ``None``)
            Whether to use the gradient to rank edges. If ``None``, it defaults
            to ``True`` is the number of edge categories is empty.
        h : ``float`` (optional, default: ``1e-8``)
            Step length used to compute the gradient with central finite
            difference.
        allow_edges : ``boolean`` (optional, default: ``False``)
            Permit currently present edges to be included in the search.
        use_hint : ``boolean`` (optional, default: ``True``)
            Use current edges as a hint during the search.
        nrandom : ``int`` (optional, default: ``0``)
            Add this many random entries to the list.
        keep_all : ``boolean`` (optional, default: ``False``)
            Keep all entries seen during the search, not only the best.
        exact : ``boolean`` (optional, default: ``False``)
            If ``True`` an exact quadratic algorithm will be used.
        return_graph : ``boolean`` (optional, default: ``False``)
            If ``True`` the result will be returned as graph and a property map.
        keep_iter : ``boolean`` (optional, default: ``False``)
            If ``True`` the result contain the iteration at which an entry has
            been found.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`~.DynamicsBlockStateBase.entropy`.

        Returns
        -------
        elist : :class:``~numpy.ndarray`` of shape ``(E, 2)``
            Best entries.
        a : :class:``~numpy.ndarray``
            Edge scores.

        """

        g = Graph(self.g.num_vertices(), fast_edge_removal=True)
        w = g.new_ep("double")
        ei = g.new_ep("int32_t")
        N = g.num_vertices()
        if knn:
            M = k
        else:
            M = k * self.g.num_vertices()
        M = max((int(round(M)), 1))
        if max_rk is None:
            max_rk = g.num_vertices()
        elif max_rk == "k":
            max_rk = k
        ea = self._get_entropy_args(entropy_args)
        if gradient is None:
            gradient = len(self.get_xvals()) == 0
        n_tot = self._state.get_candidate_edges(g._Graph__graph, M, r, max_rk,
                                                epsilon, c_stop, max_iter,
                                                _prop("e", g, w),
                                                _prop("e", g, ei),
                                                keep_iter, ea, exact, knn,
                                                keep_all, gradient, h,
                                                f_max_iter, tol, allow_edges,
                                                include_edges, use_hint,
                                                nrandom, verbose,
                                                _get_rng())
        if return_graph:
            if not knn and not self.u.is_directed():
                g.set_directed(False)
            if keep_iter:
                return g, w, ei, n_tot
            else:
                return g, w, n_tot
        elist = g.get_edges([w])
        idx = elist[:,2].argsort()
        a = elist[idx,2]
        elist = numpy.array((elist[idx,0], elist[idx,1]), dtype="int").T
        return elist, a

    def virtual_remove_edge(self, u, v, dm=1, entropy_args={}):
        """Return the difference in description length if edge :math:`(u, v)`
        with multiplicity ``dm`` would be removed.
        """
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.remove_edge_dS(int(u), int(v), dm, entropy_args)

    def virtual_add_edge(self, u, v, x, dm=1, entropy_args={}):
        """Return the difference in description length if edge :math:`(u, v)`
        would be added with multiplicity ``dm`` and weight ``x``.
        """
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.add_edge_dS(int(u), int(v), dm, x, entropy_args)

    def virtual_update_edge(self, u, v, nx, entropy_args={}):
        """Return the difference in description length if edge :math:`(u, v)`
        would take a new weight ``nx``.
        """
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.update_edge_dS(int(u), int(v), nx, entropy_args)

    def virtual_update_node(self, v, nt, entropy_args={}):
        """Return the difference in description length if node ``v``
        would take a new value ``nt``.
        """
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.update_node_dS(int(v), nt, entropy_args)

    def remove_edge(self, u, v, dm=1):
        r"""Remove edge :math:`(u, v)` with multiplicity ``dm``."""
        return self._state.remove_edge(int(u), int(v), dm)

    def add_edge(self, u, v, x, dm=1):
        r"""Add edge :math:`(u, v)` with multiplicity ``dm`` and weight ``x``."""
        return self._state.add_edge(int(u), int(v), dm, x)

    def update_edge(self, u, v, nx):
        r"""update edge :math:`(u, v)` with weight ``nx``."""
        return self._state.update_edge(int(u), int(v), nx)

    def update_node(self, v, nt):
        r"""update node :math:`(u, v)` with value ``nt``."""
        return self._state.update_node(int(v), nt)

    def bisect_x(self, u, v, maxiter=0, tol=1e-7, entropy_args={},
                 reversible=False, fb=False):
        r"""Perform a bisection search to find the best weight value for edge
        :math:`(u, v)`.

        Parameters
        ----------
        u : ``int`` or :class:`~graph_tool.Vertex`
            Source
        v : ``int`` or :class:`~graph_tool.Vertex`
            Target
        maxiter : ``int`` (optional, default: ``0``)
            Maximum number of iterations for bisection search (``0`` means unlimited).
        tol : ``float`` (optional, default: ``1e-7``)
            Tolerance for bisection search.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`~.DynamicsBlockStateBase.entropy`.
        reversible : ``boolean`` (optional, default: ``False``)
            Perform search in a manner that is usable for a reversible Makov
            chain.
        fb : ``boolean`` (optional, default: ``False``)
            Perform a Fibonacci (a.k.a. golden ratio) search, instead of a
            random bisection search.
        """
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.bisect_x(int(u), int(v), maxiter, tol, entropy_args,
                                    reversible, fb, _get_rng())

    def bisect_t(self, v, maxiter=0, tol=1e-7, entropy_args={},
                 reversible=False, fb=False):
        r"""Perform a bisection search to find the best value of node ``v``.

        Parameters
        ----------
        u : ``int`` or :class:`~graph_tool.Vertex`
            Source
        v : ``int`` or :class:`~graph_tool.Vertex`
            Target
        maxiter : ``int`` (optional, default: ``0``)
            Maximum number of iterations for bisection search (``0`` means unlimited).
        tol : ``float`` (optional, default: ``1e-7``)
            Tolerance for bisection search.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`~.DynamicsBlockStateBase.entropy`.
        reversible : ``boolean`` (optional, default: ``False``)
            Perform search in a manner that is usable for a reversible Makov
            chain.
        fb : ``boolean`` (optional, default: ``False``)
            Perform a Fibonacci (a.k.a. golden ratio) search, instead of a
            random bisection search.
        """
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.bisect_t(int(v), maxiter, tol, entropy_args,
                                    reversible, fb, _get_rng())

    def sample_x(self, u, v, beta=np.inf, maxiter=0, tol=1e-7, entropy_args={},
                 fb=False):
        r"""Sample a weight value for edge :math:`(u, v)` according to the
        conditional posterior.

        Parameters
        ----------
        u : ``int`` or :class:`~graph_tool.Vertex`
            Source
        v : ``int`` or :class:`~graph_tool.Vertex`
            Target
        maxiter : ``int`` (optional, default: ``0``)
            Maximum number of iterations for bisection search (``0`` means unlimited).
        tol : ``float`` (optional, default: ``1e-7``)
            Tolerance for bisection search.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`~.DynamicsBlockStateBase.entropy`.
        reversible : ``boolean`` (optional, default: ``False``)
            Perform search in a manner that is usable for a reversible Makov
            chain.
        fb : ``boolean`` (optional, default: ``False``)
            Perform a Fibonacci (a.k.a. golden ratio) search, instead of a
            random bisection search.
        """
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.sample_x(int(u), int(v), beta, maxiter, tol,
                                    entropy_args, fb, _get_rng())

    def sample_t(self, v, beta=np.inf, maxiter=0, tol=1e-7, entropy_args={}, fb=False):
        r"""Sample a value for node ``v`` according to the conditional posterior.

        Parameters
        ----------
        u : ``int`` or :class:`~graph_tool.Vertex`
            Source
        v : ``int`` or :class:`~graph_tool.Vertex`
            Target
        maxiter : ``int`` (optional, default: ``0``)
            Maximum number of iterations for bisection search (``0`` means unlimited).
        tol : ``float`` (optional, default: ``1e-7``)
            Tolerance for bisection search.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`~.DynamicsBlockStateBase.entropy`.
        reversible : ``boolean`` (optional, default: ``False``)
            Perform search in a manner that is usable for a reversible Makov
            chain.
        fb : ``boolean`` (optional, default: ``False``)
            Perform a Fibonacci (a.k.a. golden ratio) search, instead of a
            random bisection search.
        """
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.sample_t(int(v), beta, maxiter, tol, entropy_args,
                                    fb, _get_rng())

    def sample_xl1(self, minval, maxval, beta=np.inf, maxiter=0, tol=1e-7,
                   entropy_args={}):
        r"""Sample from the conditional posterior of the :math:`\lambda`
        parameter associated with the edge categories.

        Parameters
        ----------
        minval : ``float``
            Minimum value to consider.
        minval : ``float``
            Maximum value to consider.
        beta : ``float`` (optional, default: ``np.inf``)
            Inverse temperature parameter.
        niter : ``int`` (optional, default: ``1``)
            Number of iterations.
        maxiter : ``int`` (optional, default: ``0``)
            Maximum number of iterations for bisection search (``0`` means unlimited).
        tol : ``float`` (optional, default: ``1e-7``)
            Tolerance for bisection search.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`~.DynamicsBlockStateBase.entropy`.

        Returns
        -------
        xl1 : ``float``
            Sampled value of :math:`\lambda`.
        """
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.sample_xl1(minval, maxval, beta, maxiter, tol,
                                      entropy_args, _get_rng())

    def sample_tl1(self, minval, maxval, beta=np.inf, maxiter=0, tol=1e-7,
                   entropy_args={}):
        r"""Sample from the conditional posterior of the :math:`\lambda`
        parameter associated with the node categories.

        Parameters
        ----------
        minval : ``float``
            Minimum value to consider.
        minval : ``float``
            Maximum value to consider.
        beta : ``float`` (optional, default: ``np.inf``)
            Inverse temperature parameter.
        niter : ``int`` (optional, default: ``1``)
            Number of iterations.
        maxiter : ``int`` (optional, default: ``0``)
            Maximum number of iterations for bisection search (``0`` means unlimited).
        tol : ``float`` (optional, default: ``1e-7``)
            Tolerance for bisection search.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`~.DynamicsBlockStateBase.entropy`.

        Returns
        -------
        tl1 : ``float``
            Sampled value of :math:`\lambda`.
        """
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.sample_tl1(minval, maxval, beta, maxiter, tol,
                                      entropy_args, _get_rng())

    def sample_delta(self, minval, maxval, beta=np.inf, maxiter=0, tol=1e-7,
                     entropy_args={}):
        r"""Sample from the conditional posterior of the :math:`\Delta`
        parameter associated with the edge and node categories.

        Parameters
        ----------
        minval : ``float``
            Minimum value to consider.
        minval : ``float``
            Maximum value to consider.
        beta : ``float`` (optional, default: ``np.inf``)
            Inverse temperature parameter.
        niter : ``int`` (optional, default: ``1``)
            Number of iterations.
        maxiter : ``int`` (optional, default: ``0``)
            Maximum number of iterations for bisection search (``0`` means unlimited).
        tol : ``float`` (optional, default: ``1e-7``)
            Tolerance for bisection search.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`~.DynamicsBlockStateBase.entropy`.

        Returns
        -------
        delta : ``float``
            Sampled value of :math:`\Delta`.
        """
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.sample_delta(minval, maxval, beta, maxiter, tol,
                                        entropy_args, _get_rng())

    def sample_val_lprob(self, x, xc, beta=np.inf):
        """Compute probability of sampling value ``x`` from bisection history
        ``xc`` and inverse temperature ``beta``."""
        return self._state.sample_val_lprob(x, xc, beta)

    def quantize_x(self, delta):
        """Quantize weight values according to multiples of :math:`\Delta`."""
        self._state.quantize_x(delta)

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
            g.ep.eprob = g.new_ep("double")

        if "x" not in g.ep:
            g.ep.xsum = g.new_ep("double")
            g.ep.x2sum = g.new_ep("double")
            g.ep.x = g.new_ep("double")
            g.ep.xdev = g.new_ep("double")

        u = self.get_graph()
        x = self.get_x()
        libinference.collect_xmarginal(g._Graph__graph,
                                       u._Graph__graph,
                                       _prop("e", u, x),
                                       _prop("e", g, g.ep.count),
                                       _prop("e", g, g.ep.xsum),
                                       _prop("e", g, g.ep.x2sum))
        g.gp.count += 1
        g.ep.eprob.fa = g.ep.count.fa / g.gp.count
        g.ep.x.fa = g.ep.xsum.fa / g.gp.count
        g.ep.xdev.fa = sqrt(g.ep.x2sum.fa / g.gp.count - g.ep.x.fa ** 2)
        return g

class BPBlockStateBase(DynamicsBlockStateBase, ABC):
    def __init__(self, *args, **kwargs):
        r"""Base state for network reconstruction where the respective model can
        be used with belief-propagation."""

        DynamicsBlockStateBase.__init__(self, *args, **kwargs)

    def __getstate__(self):
        return dict(DynamicsBlockStateBase.__getstate__(self), alpha=self.alpha,
                    beta=self.beta)

    def get_S_bp(self, **kwargs):
        """Get negative model likelihood according to BP."""
        bstate = self.get_bp_state(converge=False)
        bstate.converge(**kwargs)
        S = 0
        for s in self.s:
            S -= bstate.log_prob(s)
        return S

    def get_elist_grad(self, h=1e-6, entropy_args={}):
        """Get edge list gradient."""
        entropy_args = self._get_entropy_args(entropy_args)
        ediff = []
        for u, v in self.elist[1]:
            ediff.append(self._state.edge_diff(u, v, h, entropy_args))
        return ediff

    def get_node_grad(self, h=1e-6, entropy_args={}):
        """Get node gradient."""
        entropy_args = self._get_entropy_args(entropy_args)
        ndiff = []
        for v in self.u.vertices():
            ndiff.append(self._state.node_diff(v, h, entropy_args))
        return ndiff

    def get_S(self):
        """Get negative model likelihood according to pseudo-likelihood."""
        return DynamicsBlockStateBase.entropy(self, sbm=False, tdist=False,
                                              xdist=False, density=False, xl1=0,
                                              tl1=0, alpha=1, test=False)

    @abstractmethod
    def get_bp_state(**kwargs):
        pass

class EpidemicsBlockState(DynamicsBlockStateBase):
    def __init__(self, s, g=None, exposed=False, theta=-1, x_range=(-np.inf, 1e-8),
                 theta_range=(-np.inf, 1e-8), **kwargs):
        r"""Inference state for network reconstruction based on epidemic dynamics, using
        the stochastic block model as a prior.

        Parameters
        ----------
        s : :class:`~numpy.ndarray` of shape ``(N,M)`` or ``list`` of :class:`~graph_tool.VertexPropertyMap` or :class:`~graph_tool.VertexPropertyMap`
            Time series used for reconstruction.

            If the type is :class:`~numpy.ndarray`, it should correspond to a
            ``(N,M)`` data matrix with ``M`` samples for all ``N`` nodes.

            A value of ``1`` means infected and ``0`` susceptible. Other
            values are allowed (e.g. for recovered), but their actual value is
            unimportant for reconstruction.

            If the parameter ``g`` is provided, this can be optionally a list of
            of :class:`~graph_tool.VertexPropertyMap` objects, where each entry
            in this list must be a :class:`~graph_tool.VertexPropertyMap` with
            type ``vector<int>``. If a single property map is given, then a
            single time series is assumed.

            If the parameter ``t`` below is given, each property map value for a
            given node should contain only the states for the same points in
            time given by that parameter.
        g : :class:`~graph_tool.Graph` (optional, default: ``None``)
            Initial graph state. If not provided, an empty graph will be assumed.
        exposed : ``boolean`` (optional, default: ``False``)
            If ``True``, the data is supposed to come from a SEI, SEIR,
            etc. model, where a susceptible node (valued ``0``) first transits
            to an exposed state (valued ``-1``) upon transmission, before
            transiting to the infective state (valued ``1``).
        **kwargs : (optional)
            Remaining parameters to be passed to :class:`~graph_tool.inference.DynamicsBlockStateBase`

        References
        ----------
        .. [peixoto-network-2019] Tiago P. Peixoto, "Network reconstruction and
           community detection from dynamics", Phys. Rev. Lett. 123 128301
           (2019), :doi:`10.1103/PhysRevLett.123.128301`, :arxiv:`1903.10833`
        .. [peixoto-network-2024] Tiago P. Peixoto, "Network reconstruction via
           the minimum description length principle", :arxiv:`2405.01015`
        .. [peixoto-scalable-2024] Tiago P. Peixoto, "Scalable network
           reconstruction in subquadratic time", :arxiv:`2401.01404`

        """
        directed = kwargs.pop("directed", None)
        if g is None and directed is None:
            directed = True
        DynamicsBlockStateBase.__init__(self, s, g=g, exposed=exposed,
                                        theta=theta, x_range=x_range,
                                        theta_range=theta_range,
                                        directed=directed, discrete=True,
                                        **kwargs)

    def _make_state(self):
        return libinference.make_epidemics_state(self._state, self.ot, self.os,
                                                 self.params)

    def get_t(self):
        """Return the latent edge transmission probabilities."""
        x = self.get_x()
        x.fa = 1-exp(x.fa)
        return x

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_epidemics_sweep(mcmc_state, self._state,
                                                 _get_rng())

class IsingBlockStateBase(ABC):
    @abstractmethod
    def __init__(self, s, g=None, has_zero=False, **kwargs):
        r"""Base state for network reconstruction based on the Ising model, using the
        stochastic block model as a prior.

        This class is not supposed to be instantiated directly.

        Instead one of its specialized subclasses must be used, which have the
        same signature: :class:`IsingGlauberBlockState`,
        :class:`PseudoIsingBlockState`, :class:`CIsingGlauberBlockState`,
        :class:`PseudoCIsingBlockState`.

        Parameters
        ----------
        s : :class:`~numpy.ndarray` of shape ``(N,M)`` or ``list`` of :class:`~graph_tool.VertexPropertyMap` or :class:`~graph_tool.VertexPropertyMap`
            Time series or independent samples used for reconstruction.

            If the type is :class:`~numpy.ndarray`, it should correspond to a
            ``(N,M)`` data matrix with ``M`` samples for all ``N`` nodes.

            The values must correspond to Ising states: ``-1`` or ``+1``

            If the parameter ``g`` is provided, this can be optionally a list of
            of :class:`~graph_tool.VertexPropertyMap` objects, where each entry
            in this list must be a :class:`~graph_tool.VertexPropertyMap` with
            type ``vector<int>``. If a single property map is given, then a
            single time series is assumed.

            If the parameter ``t`` below is given, each property map value for a
            given node should contain only the states for the same points in
            time given by that parameter.
        g : :class:`~graph_tool.Graph` (optional, default: ``None``)
            Initial graph state. If not provided, an empty graph will be assumed.
        has_zero : bool (optional, default: ``False``)
            If ``True``, the three-state "Ising" model with values ``{-1,0,1}``
            is used.
        **kwargs : (optional)
            Remaining parameters to be passed to :class:`~graph_tool.inference.DynamicsBlockStateBase`

        References
        ----------
        .. [ising-model] https://en.wikipedia.org/wiki/Ising_model
        .. [peixoto-network-2019] Tiago P. Peixoto, "Network reconstruction and
           community detection from dynamics", Phys. Rev. Lett. 123 128301
           (2019), :doi:`10.1103/PhysRevLett.123.128301`, :arxiv:`1903.10833`
        .. [peixoto-network-2024] Tiago P. Peixoto, "Network reconstruction via
           the minimum description length principle", :arxiv:`2405.01015`
        .. [peixoto-scalable-2024] Tiago P. Peixoto, "Scalable network
           reconstruction in subquadratic time", :arxiv:`2401.01404`

        """
        pass

    def get_dyn_state(self, s=None):
        """Return an :class:`~graph_tool.dynamics.IsingGlauberState` instance
        corresponding to the inferred model, optionally with initial state given
        by ``s``."""
        return IsingGlauberState(self.u, w=self.x, h=self.theta, s=s)

class IsingGlauberBlockState(DynamicsBlockStateBase, IsingBlockStateBase):
    def __init__(self, s, g=None, has_zero=False, **kwargs):
        r"""State for network reconstruction based on the Glauber dynamics of the Ising
        model, using the stochastic block model as a prior.

        See documentation for :class:`IsingBlockStateBase` for details.

        Notes
        -----

        This is a dynamical model with a transition likelihood for node
        :math:`i` to state :math:`s_i(t+1) \in \{-1,+1\}` given by:

        .. math::

           P(s_i(t+1)|\boldsymbol s(t), \boldsymbol A,
                      \boldsymbol x, \boldsymbol \theta) =
           \frac{\exp(s_i(t+1)\sum_jA_{ij}x_{ij}s_j(t) + \theta_is_i(t+1))}
           {2\cosh(\sum_jA_{ij}x_{ij}s_j(t) + \theta_i)}.

        """
        DynamicsBlockStateBase.__init__(self, s, g=g, has_zero=has_zero,
                                        discrete=True, **kwargs)

    def _make_state(self):
        return libinference.make_ising_glauber_state(self._state, self.ot,
                                                     self.os, self.params)

class CIsingGlauberBlockState(DynamicsBlockStateBase, IsingBlockStateBase):
    def __init__(self, s, g=None, has_zero=False, **kwargs):
        r"""State for network reconstruction based on the Glauber dynamics of the
        continuous Ising model, using the stochastic block model as a prior.

        See documentation for :class:`IsingBlockStateBase` for details. Note
        that in this case the ``s`` parameter should contain property maps of
        type ``vector<double>``, with values in the range :math:`[-1,1]`.

        Notes
        -----

        This is a dynamical model with a transition likelihood for node
        :math:`i` to state :math:`s_i(t+1) \in [-1,+1]` given by

        .. math::

           P(s_i(t+1)|\boldsymbol s(t), \boldsymbol A,
                      \boldsymbol x, \boldsymbol \theta) =
           \frac{\exp(s_i(t+1)\sum_jA_{ij}w_{ij}s_j(t) + h_is_i(t+1))}
           {Z(\sum_jA_{ij}w_{ij}s_j(t) + h_i)},

        with :math:`Z(x) = 2\sinh(x)/x`.
        """
        DynamicsBlockStateBase.__init__(self, s, g=g, has_zero=has_zero,
                                        **kwargs)

    def get_dyn_state(self, s=None):
        """Return an :class:`~graph_tool.dynamics.CIsingGlauberState` instance
        corresponding to the inferred model, optionally with initial state given
        by ``s``."""
        return CIsingGlauberState(self.u, w=self.x, h=self.theta, s=s)

    def _make_state(self):
        return libinference.make_cising_glauber_state(self._state, self.ot,
                                                      self.os, self.params)

class PseudoIsingBlockState(IsingBlockStateBase, BPBlockStateBase):
    def __init__(self, s, g=None, has_zero=False, **kwargs):
        r"""State for network reconstruction based on the equilibrium
        configurations of the Ising model, using the pseudolikelihood
        approximation and the stochastic block model as a prior.

        See documentation for :class:`IsingBlockStateBase` for details.

        Notes
        -----

        This is a equilibrium model with where the states :math:`\boldsymbol s`
        are sampled with probability

        .. math::

           P(\boldsymbol s | \boldsymbol A,
                      \boldsymbol x, \boldsymbol \theta) =
           \frac{\exp(\sum_jA_{ij}x_{ij}s_is_j + \sum_i\theta_is_i)}
           {Z(\boldsymbol A, \boldsymbol x, \boldsymbol \theta)},

        where :math:`Z(\boldsymbol A, \boldsymbol x, \boldsymbol \theta)` is an
        intractable normalization constant.

        Instead of computing this likelihood exactly, this model makes use of
        the pseudo-likelihood approximation [pseudo]_:

        .. math::

           P(\boldsymbol s | \boldsymbol A,
                      \boldsymbol x, \boldsymbol \theta) =
           \prod_{i<j}\frac{\exp(s_i\sum_{j\ne i}A_{ij}x_{ij}s_j + \theta_is_i)}
           {2\cosh(\sum_{j\ne i}A_{ij}x_{ij}s_j + \theta_i)}.

        References
        ----------
        .. [pseudo] https://en.wikipedia.org/wiki/Pseudolikelihood

        """
        BPBlockStateBase.__init__(self, s, g=g, has_zero=has_zero,
                                  directed=False, discrete=True, **kwargs)

    def get_bp_state(self, **kwargs):
        """Return an :class:`~graph_tool.dynamics.IsingBPState` instance
        corresponding to the inferred model."""
        return IsingBPState(self.u, x=self.x,
                            theta=self.theta,
                            has_zero=self.params.get("has_zero"), **kwargs)

    def _make_state(self):
        return libinference.make_pseudo_ising_state(self._state, self.ot,
                                                    self.os, self.params)

class PseudoCIsingBlockState(DynamicsBlockStateBase, IsingBlockStateBase):
    def __init__(self, s, g=None, has_zero=False, **kwargs):
        r"""State for network reconstruction based on the equilibrium configurations of
        the continuous Ising model, using the Pseudolikelihood approximation and
        the stochastic block model as a prior.

        See documentation for :class:`IsingBlockStateBase` for details.

        Note that in this case the ``s`` parameter should contain property maps
        of type ``vector<double>``, with values in the range :math:`[-1,1]`.

        Notes
        -----

        This is a equilibrium model with where the states :math:`\boldsymbol s`
        are sampled with probability

        .. math::

           P(\boldsymbol s | \boldsymbol A,
                      \boldsymbol x, \boldsymbol \theta) =
           \frac{\exp(\sum_jA_{ij}x_{ij}s_is_j + \sum_i\theta_is_i)}
           {Z(\boldsymbol A, \boldsymbol x, \boldsymbol \theta)},

        where :math:`Z(\boldsymbol A, \boldsymbol x, \boldsymbol \theta)` is an
        intractable normalization constant.

        Instead of computing this likelihood exactly, this model makes use of
        the pseudo-likelihood approximation [pseudo]_:

        .. math::

           P(\boldsymbol s | \boldsymbol A,
                      \boldsymbol x, \boldsymbol \theta) =
           \prod_{i<j}\frac{\exp(s_i\sum_{j\ne i}A_{ij}x_{ij}s_j + \theta_is_i)}
           {Z(\sum_{j\ne i}A_{ij}x_{ij}s_j + \theta_i)},

        with :math:`Z(x) = 2\sinh(x)/x`.

        References
        ----------
        .. [pseudo] https://en.wikipedia.org/wiki/Pseudolikelihood

        """
        DynamicsBlockStateBase.__init__(self, s, g=g, has_zero=has_zero,
                                        directed=False, **kwargs)

    def _make_state(self):
        return libinference.make_pseudo_cising_state(self._state, self.ot,
                                                     self.os, self.params)

    def get_dyn_state(self, s=None):
        """Return an :class:`~graph_tool.dynamics.CIsingGlauberState` instance
        corresponding to the inferred model, optionally with initial state given
        by ``s``."""
        return CIsingGlauberState(self.u, w=self.x, h=self.theta, s=s)

def zero_mean(s, g):
    if isinstance(s, np.ndarray):
        s = s.copy()
        for i in range(s.shape[0]):
            s[i,:] -= s[i,:].mean()
    else:
        if isinstance(s, list):
            s = [sx.copy("vector<double>") for sx in s]
        else:
            s = [s.copy("vector<double>")]
        for sx in s:
            for v in g.vertices():
                sx[v].a -= sx[v].a.mean()
    return s


class PseudoNormalBlockState(BPBlockStateBase):
    def __init__(self, s, g=None, fix_mean=True, positive=True, pslack=1e-6,
                 **kwargs):
        r"""State for network reconstruction based on the multivariate normal
        distribution, using the Pseudolikelihood approximation and the
        stochastic block model as a prior.

        ``fix_mean == True`` means that ``s`` will be changed to become zero-mean.

        ``positive == True`` ensures that the result is positive-semidefinite,
        according to slack given by ``pslack``.

        See documentation for :class:`DynamicsBlockStateBase` for more details.
        """

        if fix_mean:
            s = zero_mean(s, g)
        BPBlockStateBase.__init__(self, s, g=g, positive=positive, pslack=pslack,
                                  directed=False, **kwargs)

    def _make_state(self):
        return libinference.make_pseudo_normal_state(self._state, self.ot,
                                                     self.os, self.params)

    def get_dcov(self):
        """Return data covariance matrix."""
        N = self.g.num_vertices()
        S = np.zeros((N, N))
        M = 0
        for m in range(len(self.s)):
            M += len(self.s[m][0])
            for i in range(N):
                for j in range(N):
                    S[i,j] += np.dot(self.s[0][i].a,
                                     self.s[0][j].a)
        S /= M
        return S

    def get_precision(self):
        """Return precision matrix."""
        theta = self.theta.copy()
        w = self.x.copy()
        w.fa = abs(w.fa)
        pslack = self.params["pslack"]
        for v in self.u.vertices():
            k = v.out_degree(w)
            theta[v] =  min(theta[v], -log(k)/2-pslack) if k > 0 else theta[v]
        W = adjacency(self.u, self.x).tolil()
        for v in self.u.vertices():
            W[int(v), int(v)] = 1/exp(theta[v] * 2)
        return W

    def get_theta_shifted(self):
        """Return shifted node values to ensure positive semi-definiteness."""
        h = self.theta.copy()
        w = self.x.copy()
        w.fa = np.abs(w.fa)
        k = self.u.degree_property_map("out", w)
        idx = k.a > 0
        pslack = self.params["pslack"]
        h.a[idx] = np.minimum(-np.log(k.a[idx])/2 - pslack, h.a[idx])
        return h

    def log_P_exact(self):
        """Return exact log-likelihood."""
        S = self.get_dcov()
        W = self.get_precision().todense()
        N = W.shape[0]
        D = np.linalg.slogdet(W)[1]
        M = 0
        for m in range(len(self.s)):
            M += len(self.s[m][0])
        return float(-M*(S@W).trace()/2 + M/2*D - M*N*np.log(2*np.pi)/2)

    def log_Z_exact(self):
        """Return exact log-likelihood normalization constant."""
        W = self.get_precision().todense()
        N = W.shape[0]
        D = np.linalg.slogdet(W)[1]
        return -float(D/2 - N * np.log(2 * np.pi)/2)

    def get_dyn_state(self, s=None):
        """Return an :class:`~graph_tool.dynamics.NormalState` instance
        corresponding to the inferred model, optionally with initial state given
        by ``s``."""
        h = self.get_theta_shifted()
        h.a = np.exp(h.a)
        return NormalState(self.u, w=self.x, h=h, s=s)

    def get_bp_state(self, **kwargs):
        """Return an :class:`~graph_tool.dynamics.NormalBPState` instance
        corresponding to the inferred model."""
        theta = self.get_theta_shifted()
        theta.fa = np.exp(-2 * theta.fa)
        return NormalBPState(self.u, x=self.x, theta=theta, **kwargs)


class NormalGlauberBlockState(DynamicsBlockStateBase):
    def __init__(self, s, g=None, self_loops=False, fix_mean=True, **kwargs):
        r"""State for network reconstruction based on the dynamical multivariate
        normal distribution, using the Pseudolikelihood approximation and the
        stochastic block model as a prior.

        ``fix_mean == True`` means that ``s`` will be changed to become zero-mean.

        ``positive == True`` ensures that the result is positive-semidefinite,
        according to slack given by ``pslack``.

        See documentation for :class:`DynamicsBlockStateBase` for more details.
        """

        if fix_mean:
            s = zero_mean(s, g)
        DynamicsBlockStateBase.__init__(self, s, g=g, self_loops=self_loops,
                                        **kwargs)

    def _make_state(self):
        return libinference.make_normal_glauber_state(self._state, self.ot,
                                                      self.os, self.params)

    def get_dyn_state(self, s=None):
        """Return an :class:`~graph_tool.dynamics.NormalState` instance
        corresponding to the inferred model, optionally with initial state given
        by ``s``."""
        h = self.get_theta_shifted()
        h.a = np.exp(h.a)
        return NormalState(self.u, w=self.x, h=h, s=s)

class LinearNormalBlockState(DynamicsBlockStateBase):
    def __init__(self, s, g=None, self_loops=True, **kwargs):
        r"""State for network reconstruction based on a linear dynamical model.

        ``self_loops == True`` means self-loops will be allowed in the
        reconstruction.

        See documentation for :class:`DynamicsBlockStateBase` for more details.

        """
        DynamicsBlockStateBase.__init__(self, s, g=g, self_loops=self_loops,
                                        **kwargs)

    def _make_state(self):
        return libinference.make_linear_normal_state(self._state, self.ot,
                                                     self.os, self.params)

    def get_dyn_state(self, s=None):
        """Return an :class:`~graph_tool.dynamics.LinearNormalState` instance
        corresponding to the inferred model, optionally with initial state given
        by ``s``."""
        h = self.theta.copy()
        h.a = np.exp(h.a)
        return LinearNormalState(self.u, w=self.x, sigma=h, s=s)

class LVBlockState(DynamicsBlockStateBase):
    def __init__(self, s, g=None, self_loops=True, sigma=1, **kwargs):
        DynamicsBlockStateBase.__init__(self, s, g=g, self_loops=self_loops,
                                        sigma=sigma, **kwargs)

    def _make_state(self):
        return libinference.make_lotka_volterra_state(self._state, self.ot,
                                                      self.os, self.params)

    def get_dyn_state(self, s=None):
        """Return an :class:`~graph_tool.dynamics.LVState` instance
        corresponding to the inferred model, optionally with initial state given
        by ``s``."""
        r = self.theta.copy()
        return LVState(self.u, w=self.x, r=r, sigma=self.params["sigma"], s=s)
