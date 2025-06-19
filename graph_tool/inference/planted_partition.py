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

from .. import Graph, GraphView, _get_rng, Vector_size_t, PropertyMap, \
    group_vector_property
from . blockmodel import DictState, init_q_cache
from . base_states import *
from . util import *

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

import numpy as np
import math

@entropy_state_signature
class PPBlockState(MCMCState, MultiflipMCMCState, MultilevelMCMCState,
                   GibbsMCMCState, DrawBlockState):
    r"""Obtain the partition of a network according to the Bayesian planted partition
    model.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modelled.
    b : :class:`~graph_tool.PropertyMap` (optional, default: ``None``)
        Initial partition. If not supplied, a partition into a single group will
        be used.
    entropy_args: ``dict`` (optional, default: ``{}``)
        Override default arguments for :meth:`~PPBlockState.entropy()`
        method and releated operations.

    References
    ----------
    .. [lizhi-statistical-2020] Lizhi Zhang, Tiago P. Peixoto, "Statistical
       inference of assortative community structures", Phys. Rev. Research 2
       043271 (2020), :doi:`10.1103/PhysRevResearch.2.043271`, :arxiv:`2006.14493`
    """

    def __init__(self, g, b=None, entropy_args={}):
        EntropyState.__init__(self, entropy_args=entropy_args)
        self.g = GraphView(g, directed=False)
        if b is None:
            self.b = self.g.new_vp("int32_t")
        elif isinstance(g, PropertyMap):
            self.b = b.copy("int32_t")
        else:
            self.b = self.g.new_vp("int32_t", vals=b)

        self.wr = Vector_size_t()
        self.er = Vector_size_t()
        self.err = Vector_size_t()
        self.eio = Vector_size_t()

        init_q_cache(max((2 * max((self.g.num_edges(),
                                   self.g.num_vertices())), 100)))

        self.bg = GraphView(self.g)
        self.bg.clear_filters()
        self._abg = self.bg._get_any()
        self._state = libinference.make_pp_state(self)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        g = copy.deepcopy(self.g, memo)
        b = copy.deepcopy(self.b, memo)
        return self.copy(g=g, b=b)

    def copy(self, g=None, b=None):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""
        return PPBlockState(g=g if g is not None else self.g,
                            b=b if b is not None else self.b)

    def __getstate__(self):
        state = EntropyState.__getstate__(self)
        return dict(state, g=self.g, b=self.b)

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        return "<PPBlockState object with %d blocks, for graph %s, at 0x%x>" % \
            (self.get_B(), str(self.g), id(self))

    def get_blocks(self):
        r"""Returns the property map which contains the block labels for each vertex."""
        return self.b

    def get_state(self):
        """Alias to :meth:`~PPBlockState.get_blocks`."""
        return self.get_blocks()

    def get_B(self):
        r"Returns the total number of blocks."
        return len(np.unique(self.b.fa))

    def get_nonempty_B(self):
        r"Alias to :meth:`~PPBlockState.get_B`."
        return self.get_B()

    def get_Be(self):
        r"""Returns the effective number of blocks, defined as :math:`e^{H}`, with
        :math:`H=-\sum_r\frac{n_r}{N}\ln \frac{n_r}{N}`, where :math:`n_r` is
        the number of nodes in group r.
        """
        w = np.array(np.bincount(self.b.fa), dtype="double")
        w = w[w>0]
        w /= w.sum()
        return np.exp(-(w*log(w)).sum())

    def virtual_vertex_move(self, v, s, **kwargs):
        r"""Computes the entropy difference if vertex ``v`` is moved to block ``s``. The
        remaining parameters are the same as in
        :meth:`~graph_tool.inference.PPBlockState.entropy`."""
        return self._state.virtual_move(int(v), self.b[v], s,
                                        get_pp_entropy_args(dict(self._entropy_args,
                                                                 **kwargs)))

    def move_vertex(self, v, s):
        r"""Move vertex ``v`` to block ``s``."""
        self._state.move_vertex(int(v), int(s))

    @copy_state_wrap
    def _entropy(self, uniform=False, degree_dl_kind="distributed"):
        r"""Return the model entropy (negative log-likelihood).

        Parameters
        ----------
        uniform : ``bool`` (optional, default: ``False``)
            If ``True``, the uniform planted partition model is used, otherwise
            a non-uniform version is used.
        degree_dl_kind : ``str`` (optional, default: ``"distributed"``)
            This specifies the prior used for the degree sequence. It must be
            one of: ``"uniform"`` or ``"distributed"`` (default).

        Notes
        -----

        The "entropy" of the state is the negative log-likelihood of the
        microcanonical SBM, that includes the generated graph
        :math:`\boldsymbol{A}` and the model parameters :math:`e_{\text{in}}`,
        :math:`e_{\text{out}}`, :math:`\boldsymbol{k}` and
        :math:`\boldsymbol{b}`,

        .. math::

           \Sigma &= - \ln P(\boldsymbol{A},e_{\text{in}},e_{\text{out}},\boldsymbol{k},\boldsymbol{b}) \\
                  &= - \ln P(\boldsymbol{A}|e_{\text{in}},e_{\text{out}},\boldsymbol{k},\boldsymbol{b}) - \ln P(e_{\text{in}},e_{\text{out}},\boldsymbol{k},\boldsymbol{b}).

        This value is also called the `description length
        <https://en.wikipedia.org/wiki/Minimum_description_length>`_ of the data,
        and it corresponds to the amount of information required to describe it
        (in `nats <https://en.wikipedia.org/wiki/Nat_(unit)>`_).

        For the uniform version of the model, the likelihood is

        .. math::

            P(\boldsymbol{A}|\boldsymbol{k},\boldsymbol{b}) = \frac{e_{\text{in}}!e_{\text{out}}!}
            {\left(\frac{B}{2}\right)^{e_{\text{in}}}{B\choose 2}^{e_{\text{out}}}(E+1)^{1-\delta_{B,1}}\prod_re_r!}\times
            \frac{\prod_ik_i!}{\prod_{i<j}A_{ij}!\prod_i A_{ii}!!}.

        where :math:`e_{\text{in}}` and :math:`e_{\text{out}}` are the number of
        edges inside and outside communities, respectively, and :math:`e_r` is
        the sum of degrees in group :math:`r`.

        For the non-uniform model we have instead:

        .. math::

            P(\boldsymbol{A}|\boldsymbol{k},\boldsymbol{b}) = \frac{e_{\text{out}}!\prod_re_{rr}!!}
            {{B\choose 2}^{e_{\text{out}}}(E+1)^{1-\delta_{B,1}}\prod_re_r!}\times{B + e_{\text{in}} - 1 \choose e_{\text{in}}}^{-1}\times
            \frac{\prod_ik_i!}{\prod_{i<j}A_{ij}!\prod_i A_{ii}!!}.


        Here there are two options for the prior on the degrees:

        1. ``degree_dl_kind == "uniform"``

            .. math::

                P(\boldsymbol{k}|\boldsymbol{e},\boldsymbol{b}) = \prod_r\left(\!\!{n_r\choose e_r}\!\!\right)^{-1}.

            This corresponds to a noninformative prior, where the degrees are
            sampled from a uniform distribution.

        2. ``degree_dl_kind == "distributed"`` (default)

            .. math::

                P(\boldsymbol{k}|\boldsymbol{e},\boldsymbol{b}) = \prod_r\frac{\prod_k\eta_k^r!}{n_r!} \prod_r q(e_r, n_r)^{-1}

            with :math:`\eta_k^r` being the number of nodes with degree
            :math:`k` in group :math:`r`, and :math:`q(n,m)` being the number of
            `partitions
            <https://en.wikipedia.org/wiki/Partition_(number_theory)>`_ of
            integer :math:`n` into at most :math:`m` parts.

            This corresponds to a prior for the degree sequence conditioned on
            the degree frequencies, which are themselves sampled from a uniform
            hyperprior. This option should be preferred in most cases.


        For the partition prior :math:`P(\boldsymbol{b})` please refer to
        :meth:`~graph_tool.inference.BlockState.entropy`.

        References
        ----------
        .. [lizhi-statistical-2020] Lizhi Zhang, Tiago P. Peixoto, "Statistical
           inference of assortative community structures", Phys. Rev. Research 2
           043271 (2020), :doi:`10.1103/PhysRevResearch.2.043271`, :arxiv:`2006.14493`
        """

        eargs = self._get_entropy_args(locals())

        return self._state.entropy(eargs)

    def _gen_eargs(self, args):
        return libinference.pp_entropy_args()

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
        return super()._get_entropy_args(kwargs, consume)

    def _mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.pp_mcmc_sweep(mcmc_state, self._state,
                                          _get_rng())

    def _multiflip_mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.pp_multiflip_mcmc_sweep(mcmc_state, self._state,
                                                    _get_rng())

    def _multilevel_mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.pp_multilevel_mcmc_sweep(mcmc_state, self._state,
                                                     _get_rng())

    def _gibbs_sweep_dispatch(self, gibbs_state):
        return libinference.pp_gibbs_sweep(gibbs_state, self._state,
                                           _get_rng())
