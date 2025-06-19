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

from .. import Graph, PropertyMap, _get_rng, Vector_int32_t, Vector_size_t, \
    _parallel
from . blockmodel import DictState
from . base_states import *

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

import numpy as np
import math
from functools import wraps

class PartitionModeState(object):
    r"""The random label model state for a set of labelled partitions, which attempts
    to align them with a common group labelling.

    Parameters
    ----------
    bs : list of iterables
        List of partitions to be aligned. If ``nested=True``, these should be
        hierarchical partitions, composed each as a list of partitions.
    relabel : ``bool`` (optional, default: ``True``)
        If ``True``, an initial alignment of the partitions will be attempted during
        instantiation, otherwise they will be incorporated as they are.
    nested : ``bool`` (optional, default: ``False``)
        If ``True``, the partitions will be assumed to be hierarchical.
    converge : ``bool`` (optional, default: ``False``)

        If ``True``, the label alignment will be iterated until convergence upon
        initialization (otherwise
        :meth:`~graph_tool.inference.PartitionModeState.replace_partitions`
        needs to be called repeatedly).

    References
    ----------
    .. [peixoto-revealing-2021] Tiago P. Peixoto, "Revealing consensus and
       dissensus between network partitions", Phys. Rev. X 11 021003 (2021)
       :doi:`10.1103/PhysRevX.11.021003`, :arxiv:`2005.13977`

    """
    def __init__(self, bs, relabel=True, nested=False, converge=False, **kwargs):
        self.bs = {}
        self.nested = nested

        if len(kwargs) > 0:
            self._base = kwargs["base"]
        else:
            self._base = libinference.PartitionModeState()
            for b in bs:
                self.add_partition(b, relabel)

        if converge:
            delta = 1
            while np.abs(delta) > 1e-6:
                delta = self.replace_partitions()

    def __copy__(self):
        return self.copy()

    def copy(self, bs=None):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""
        if bs is None:
            bs = list(self.get_nested_partitions().values())
            if not self.nested:
                bs = [bv[0] for bv in bs]
        return PartitionModeState(bs=bs, relabel=False, nested=self.nested)

    def __getstate__(self):
        bs = list(self.get_nested_partitions().values())
        if not self.nested:
            bs = [bv[0] for bv in bs]
        return dict(bs=bs, nested=self.nested)

    def __setstate__(self, state):
        self.__init__(**state, relabel=False)

    def add_partition(self, b, relabel=True):
        r"""Adds partition ``b`` to the ensemble, after relabelling it if
        ``relabel=True``, and returns its index in the population."""
        def c(x):
            if isinstance(x, PropertyMap):
                return x.fa
            return x
        if self.nested:
            b = [Vector_int32_t(init=c(x)) for x in b]
        else:
            b = [Vector_int32_t(init=c(b))]
        i = self._base.add_partition(b, relabel)
        self.bs[i] = b
        return i

    def remove_partition(self, i):
        r"""Removes partition  with index ``i`` from the ensemble."""
        self._base.remove_partition(i)
        if i in self.bs:
            del self.bs[i]

    def virtual_add_partition(self, b, relabel=True):
        r"""Computes the entropy difference (negative log probability) if partition ``b``
        were inserted the ensemble, after relabelling it if ``relabel=True``.
        """
        if self.nested:
            b = [Vector_int32_t(init=x) for x in b]
        else:
            b = [Vector_int32_t(init=b)]
        return self._base.virtual_add_partition(b, relabel)

    def virtual_remove_partition(self, b, relabel=True):
        r"""Computes the entropy difference (negative log probability) if partition ``b``
        were removed from the ensemble, after relabelling it if ``relabel=True``.
        """
        if self.nested:
            b = [Vector_int32_t(init=x) for x in b]
        else:
            b = [Vector_int32_t(init=b)]
        return self._base.virtual_add_partition(b, relabel)

    def replace_partitions(self):
        r"""Removes and re-adds every partition, after relabelling, and returns
        the entropy difference (negative log probability)."""
        return self._base.replace_partitions(_get_rng())

    def relabel_partition(self, b):
        r"""Returns a relabelled copy of partition ``b``, according to its alignment with
        the ensemble."""
        if self.nested:
            b = [Vector_int32_t(init=x) for x in b]
        else:
            b = [Vector_int32_t(init=b)]
        self._base.relabel_partition(b)
        if not self.nested:
            return b[0].a.copy()
        else:
            return [x.a.copy() for x in b]

    def align_mode(self, mode):
        r"""Relabel entire ensemble to align with another ensemble given by ``mode``,
        which should be an instance of
        :class:`~graph_tool.inference.PartitionModeState`."""
        self._base.align_mode(mode._base)

    def get_partition(self, i):
        r"""Returns partition with index ``i``."""
        return self._base.get_partition(i)

    def get_nested_partition(self, i):
        r"""Returns nested partition with index ``i``."""
        return self._base.get_nested_partition(i)

    def get_partitions(self):
        r"""Returns all partitions."""
        return self._base.get_partitions()

    def get_nested_partitions(self):
        r"""Returns all nested partitions."""
        return self._base.get_nested_partitions()

    def relabel(self):
        r"""Re-order group labels according to group sizes."""
        return self._base.relabel()

    @copy_state_wrap
    def _entropy(self):
        r"""Return the description length (negative joint log-likelihood)."""
        return self._base.entropy()

    def posterior_entropy(self, MLE=True):
        r"""Return the entropy of the random label model, using maximum likelihood
        estimates for the marginal node probabilities if ```MLE=True```,
        otherwise using posterior mean estimates.
        """
        return self._base.posterior_entropy(MLE)

    def posterior_cdev(self, MLE=True):
        r"""Return the uncertainty of the mode in the range :math:`[0,1]`, using maximum
        likelihood estimates for the marginal node probabilities if
        ```MLE=True```, otherwise using posterior mean estimates.
        """
        return self._base.posterior_cdev(MLE)

    def posterior_lprob(self, b, MLE=True):
        r"""Return the log-probability of partition ``b``, using maximum
        likelihood estimates for the marginal node probabilities if
        ```MLE=True```, otherwise using posterior mean estimates.
        """
        if self.nested:
            b = [Vector_int32_t(init=x) for x in b]
        else:
            b = [Vector_int32_t(init=b)]
        return self._base.posterior_lprob(b, MLE)

    def get_coupled_state(self):
        r"""Return the instance of :class:`~graph_tool.inference.PartitionModeState`
        representing the model at the upper hierarchical level.
        """
        base = self._base.get_coupled_state()
        if base is None:
            return None
        return PartitionModeState(bs=None, base=base, nested=self.nested)

    def get_marginal(self, g):
        r"""Return a :class:`~graph_tool.VertexPropertyMap` for :class:`~graph_tool.Graph`
        ``g``, with ``vector<int>`` values containing the marginal group
        membership counts for each node. """
        bm = g.new_vp("vector<int>")
        self._base.get_marginal(g._Graph__graph, bm._get_any())
        return bm

    def get_max(self, g):
        r"""Return a :class:`~graph_tool.VertexPropertyMap` for
        :class:`~graph_tool.Graph` ``g``, with ``int`` values containing the
        maximum marginal group membership for each node."""
        b = g.new_vp("int")
        self._base.get_map(g._Graph__graph, b._get_any())
        return b

    def get_max_nested(self):
        r"""Return a hierarchical partition as a list of :class:`numpy.ndarray` objects,
        containing the maximum marginal group membership for each node in every
        level."""
        return self._base.get_map_bs()

    def get_B(self):
        r"""Return the total number of labels used."""
        return self._base.get_B()

    def get_M(self):
        r"""Return the number of partitions"""
        return self._base.get_M()

    def sample_partition(self, MLE=True):
        r"""Sampled a partition from the inferred model, using maximum likelihood
        estimates for the marginal node probabilities if ```MLE=True```,
        otherwise using posterior mean estimates."""
        return self._base.sample_partition(MLE, _get_rng())

    def sample_nested_partition(self, MLE=True, fix_empty=True):
        r"""Sampled a nested partition from the inferred model, using maximum likelihood
        estimates for the marginal node probabilities if ```MLE=True```,
        otherwise using posterior mean estimates."""
        return self._base.sample_nested_partition(MLE, fix_empty, _get_rng())

@entropy_state_signature
class ModeClusterState(MCMCState, MultiflipMCMCState, MultilevelMCMCState):
    r"""The mixed random label model state for a set of labelled partitions, which
    attempts to align them inside clusters with a common group labelling.

    Parameters
    ----------
    bs : list of iterables
        List of partitions to be aligned. If ``nested=True``, these should be
        hierarchical partitions, composed each as a list of partitions.
    b : iterable (optional, default: ``None``)
        Initial cluster membership for every partition. If ``None`` a random
        division into ``B`` groups will be used.
    B : ``int`` (optional, default: ``1``)
        Number of groups for initial division.
    relabel : ``bool`` (optional, default: ``True``)
        If ``True``, an initial alignment of the partitions will be attempted during
        instantiation, otherwise they will be incorporated as they are.
    nested : ``bool`` (optional, default: ``False``)
        If ``True``, the partitions will be assumed to be hierarchical.

    References
    ----------
    .. [peixoto-revealing-2021] Tiago P. Peixoto, "Revealing consensus and
       dissensus between network partitions", Phys. Rev. X 11 021003 (2021)
       :doi:`10.1103/PhysRevX.11.021003`, :arxiv:`2005.13977`

    """
    def __init__(self, bs, b=None, B=1, nested=False, relabel=True):
        EntropyState.__init__(self)
        self.bs = []
        self.nested = nested

        for bv in bs:
            if self.nested:
                bv = [Vector_int32_t(init=x) for x in bv]
            else:
                bv = [Vector_int32_t(init=bv)]
            self.bs.append(bv)

        if b is None:
            self.b = np.random.randint(0, B, len(self.bs), dtype="int32")
        else:
            self.b = np.asarray(b, dtype="int32")
            if self.b.max() >= len(self.bs):
                raise ValueError("supplied clustering 'b' has maximum label larger than number of partitions")
        self.b = Vector_int32_t(init=self.b)

        self.relabel_init = relabel
        self.g = Graph()
        self.g.add_vertex(len(self.b))
        self.bg = self.g
        self._abg = self.bg._get_any()
        self.obs = self.bs
        self._state = libinference.make_mode_cluster_state(self)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        b = copy.deepcopy(self.b, memo)
        bs = copy.deepcopy(self.bs, memo)
        return self.copy(bs=bs, b=b)

    def copy(self, bs=None, b=None):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""
        if bs is None:
            if not self.nested:
                bs = [x[0] for x in self.bs]
            else:
                bs = self.bs
        return ModeClusterState(bs=bs, b=b if b is not None else self.b,
                                relabel=False, nested=self.nested)


    def __getstate__(self):
        state = EntropyState.__getstate__(self)
        return dict(state, bs=self.bs, b=self.b, nested=self.nested)

    def __setstate__(self, state):
        if not state["nested"]:
            state["bs"] = [x[0] for x in state["bs"]]
        self.__init__(**state, relabel=False)

    def get_mode(self, r):
        r"""Return the mode in cluster ``r`` as an instance of
        :class:`~graph_tool.inference.PartitionModeState`. """
        base = self._state.get_mode(r);
        return PartitionModeState(None, base=base, nested=self.nested)

    def get_modes(self, sort=True):
        r"""Return the list of nonempty modes, as instances of
        :class:`~graph_tool.inference.PartitionModeState`. If `sorted == True`,
        the modes are retured in decreasing order with respect to their size.
        """
        modes = [self.get_mode(r) for r in np.unique(self.b)]
        if sort:
            modes = list(sorted(modes, key=lambda m: -m.get_M()))
        return modes

    def get_wr(self):
        r"""Return cluster sizes. """
        return np.array(np.bincount(self.b))

    def get_B(self):
        r"Returns the total number of clusters."
        return len(np.unique(self.b))

    def get_Be(self):
        r"""Returns the effective number of clusters, defined as :math:`e^{H}`, with
        :math:`H=-\sum_r\frac{n_r}{N}\ln \frac{n_r}{N}`, where :math:`n_r` is
        the number of partitions in cluster :math:`r`.
        """
        w = np.array(np.bincount(self.b), dtype="double")
        w = w[w>0]
        w /= w.sum()
        return np.exp(-(w*np.log(w)).sum())

    def virtual_add_partition(self, b, r, relabel=True):
        r"""Computes the entropy difference (negative log probability) if partition ``b``
        were inserted in cluster ``r``, after relabelling it if ``relabel=True``.
        """
        if self.nested:
            b = [Vector_int32_t(init=x) for x in b]
        else:
            b = [Vector_int32_t(init=b)]
        return self._state.virtual_add_partition(b, r, relabel)

    def add_partition(self, b, r, relabel=True):
        r"""Add partition ``b`` in cluster ``r``, after relabelling it if ``relabel=True``."""
        if self.nested:
            b = [Vector_int32_t(init=x) for x in b]
        else:
            b = [Vector_int32_t(init=b)]
        self._state.add_partition(b, r, relabel)

    def classify_partition(self, b, relabel=True, new_group=True, sample=False):
        r"""Returns the cluster ``r`` to which partition ``b`` would belong, after
        relabelling it if ``relabel=True``, according to the most probable
        assignment, or randomly sampled according to the relative probabilities
        if ``sample==True``. If ``new_group==True``, a new previously unoccupied
        group is also considered for the classification.  """
        rs = list(np.unique(self.b))
        if new_group:
            rs.append(max(rs) + 1)
        dS = [self.virtual_add_partition(b, r, relabel) for r in rs]
        if not sample:
            r = np.argmin(dS)
        else:
            Ps = -np.array(dS)
            Ps -= Ps.max()
            Ps = np.exp(Ps)
            Ps /= Ps.sum()
            r = np.random.choice(np.arange(len(Ps)), p=Ps)
        return rs[r]

    def relabel(self, epsilon=1e-6, maxiter=100):
        r"""Attempt to align group labels between clusters via a greedy algorithm. The
        algorithm stops after ``maxiter`` iterations or when the entropy
        improvement lies below ``epsilon``."""
        return self._state.relabel_modes(epsilon, maxiter)

    @copy_state_wrap
    def _entropy(self):
        r"""Return the description length (negative joint log-likelihood)."""
        return self._state.entropy()

    def _gen_eargs(self, eargs):
        return ""

    def posterior_entropy(self, MLE=True):
        r"""Return the entropy of the random label model, using maximum likelihood
        estimates for the marginal node probabilities if ```MLE=True```,
        otherwise using posterior mean estimates.
        """
        return self._state.posterior_entropy(MLE)

    def posterior_lprob(self, r, b, MLE=True):
        r"""Return the log-probability of partition ``b`` belonging to mode ``r``, using
        maximum likelihood estimates for the marginal node probabilities if
        ```MLE=True```, otherwise using posterior mean estimates."""
        if self.nested:
            b = [Vector_int32_t(init=x) for x in b]
        else:
            b = [Vector_int32_t(init=b)]
        return self._state.posterior_lprob(r, b, MLE)

    def replace_partitions(self):
        r"""For every cluster, removes and re-adds every partition, after relabelling,
        and returns the entropy difference (negative log probability)."""
        return self._state.replace_partitions(_get_rng())

    def sample_partition(self, MLE=True):
        r"""Sampled a cluster label and partition from the inferred model, using maximum
        likelihood estimates for the marginal node probabilities if
        ```MLE=True```, otherwise using posterior mean estimates."""
        return self._state.sample_partition(MLE, _get_rng())

    def sample_nested_partition(self, MLE=True, fix_empty=True):
        r"""Sampled a cluster label and nested partition from the inferred model, using
        maximum likelihood estimates for the marginal node probabilities if
        ```MLE=True```, otherwise using posterior mean estimates.
        """
        return self._state.sample_nested_partition(MLE, fix_empty, _get_rng())

    def _mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.mode_clustering_mcmc_sweep(mcmc_state, self._state,
                                                       _get_rng())

    def _multiflip_mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.mode_clustering_multiflip_mcmc_sweep(mcmc_state,
                                                                 self._state,
                                                                 _get_rng())

    def _multilevel_mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.mode_clustering_multilevel_mcmc_sweep(mcmc_state,
                                                                  self._state,
                                                                  _get_rng())

def _get_array(x):
    if isinstance(x, PropertyMap):
        return x.fa
    return x

def _handle_props(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        nargs = (_get_array(x) for x in args)
        ret = func(*nargs, **kwargs)
        if isinstance(ret, np.ndarray) and isinstance(args[0], PropertyMap):
            nret = args[0].copy()
            nret.fa = ret
            return nret
        return ret
    return wrapper

def _handle_nprops(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        nargs = ([_get_array(xl) for xl in x] for x in args)
        ret = func(*nargs, **kwargs)
        if isinstance(ret, list):
            nret = []
            for i in range(len(ret)):
                val = ret[i]
                if isinstance(val, np.ndarray) and isinstance(args[0][i], PropertyMap):
                    nval = args[0][i].copy()
                    nval.fa = val
                    nret.append(nval)
                else:
                    nret.append(val)
            return nret
        return ret
    return wrapper

@_handle_props
def partition_overlap(x, y, norm=True):
    r"""Returns the maximum overlap between partitions, according to an optimal
    label alignment.

    Parameters
    ----------
    x : iterable of ``int`` values or :class:`~graph_tool.PropertyMap`
        First partition.
    y : iterable of ``int`` values or :class:`~graph_tool.PropertyMap`
        Second partition.
    norm : (optional, default: ``True``)
        If ``True``, the result will be normalized in the range :math:`[0,1]`.

    Returns
    -------
    w : ``float`` or ``int``
        Maximum overlap value.

    Notes
    -----

    The maximum overlap between partitions :math:`\boldsymbol x` and :math:`\boldsymbol y` is defined as

    .. math::

        \omega(\boldsymbol x,\boldsymbol y) = \underset{\boldsymbol\mu}{\max}\sum_i\delta_{x_i,\mu(y_i)},

    where :math:`\boldsymbol\mu` is a bijective mapping between group labels. It
    corresponds to solving an instance of the maximum weighted bipartite
    matching problem, which is done with the Kuhn-Munkres algorithm
    [kuhn_hungarian_1955]_ [munkres_algorithms_1957]_.

    If ``norm == True``, the normalized value is returned:

    .. math::

        \frac{\omega(\boldsymbol x,\boldsymbol y)}{N}

    which lies in the unit interval :math:`[0,1]`.

    This algorithm runs in time :math:`O[N + (B_x+B_y)E_m]` where :math:`N` is
    the length of :math:`\boldsymbol x` and :math:`\boldsymbol y`, :math:`B_x`
    and :math:`B_y` are the number of labels in partitions :math:`\boldsymbol x`
    and :math:`\boldsymbol y`, respectively, and :math:`E_m \le B_xB_y` is the
    number of nonzero entries in the contingency table between both partitions.

    Examples
    --------
    >>> x = np.random.randint(0, 10, 1000)
    >>> y = np.random.randint(0, 10, 1000)
    >>> gt.partition_overlap(x, y)
    0.143

    References
    ----------
    .. [peixoto-revealing-2021] Tiago P. Peixoto, "Revealing consensus and
       dissensus between network partitions", Phys. Rev. X 11 021003 (2021)
       :doi:`10.1103/PhysRevX.11.021003`, :arxiv:`2005.13977`
    .. [kuhn_hungarian_1955] H. W. Kuhn, "The Hungarian method for the
       assignment problem," Naval Research Logistics Quarterly 2, 83–97 (1955)
       :doi:`10.1002/nav.3800020109`
    .. [munkres_algorithms_1957] James Munkres, "Algorithms for the Assignment
       and Transportation Problems," Journal of the Society for Industrial and
       Applied Mathematics 5, 32–38 (1957).
       :doi:`10.1137/0105003`

    """
    x = np.asarray(x, dtype="int32")
    y = np.asarray(y, dtype="int32")
    if len(x) != len(y):
        raise ValueError("Partitions need to have the same number of elements")
    m = libinference.partition_overlap(x, y)
    if norm:
        m /= max((len(x), len(y)))
    return m

@_handle_nprops
def nested_partition_overlap(x, y, norm=True):
    r"""Returns the hierarchical maximum overlap between nested partitions, according
    to an optimal recursive label alignment.

    Parameters
    ----------
    x : iterable of iterables of ``int`` values or iterable of :class:`~graph_tool.PropertyMap`
        First partition.
    y : iterable of iterables of ``int`` values or iterable of :class:`~graph_tool.PropertyMap`
        Second partition.
    norm : (optional, default: ``True``)
        If ``True``, the result will be normalized in the range :math:`[0,1]`.

    Returns
    -------
    w : ``float`` or ``int``
        Maximum hierarchical overlap value.

    Notes
    -----

    The maximum overlap between partitions :math:`\bar{\boldsymbol x}` and
    :math:`\bar{\boldsymbol y}` is defined as

    .. math::

        \omega(\bar{\boldsymbol x},\bar{\boldsymbol y}) = \sum_l\underset{\boldsymbol\mu_l}{\max}\sum_i\delta_{x_i^l,\mu_l(\tilde y_i^l)},

    where :math:`\boldsymbol\mu_l` is a bijective mapping between group labels
    at level :math:`l`, and :math:`\tilde y_i^l = y^i_{\mu_{l-1}(i)}` are the
    nodes reordered according to the lower level. It corresponds to solving an
    instance of the maximum weighted bipartite matching problem for every
    hierarchical level, which is done with the Kuhn-Munkres algorithm
    [kuhn_hungarian_1955]_ [munkres_algorithms_1957]_.

    If ``norm == True``, the normalized value is returned:

    .. math::

        1 - \frac{\left(\sum_lN_l\right) - \omega(\bar{\boldsymbol x}, \bar{\boldsymbol y})}{\sum_l\left(N_l - 1\right)}

    which lies in the unit interval :math:`[0,1]`, where
    :math:`N_l=\max(N_{{\boldsymbol x}^l}, N_{{\boldsymbol y}^l})` is the number of
    nodes in level `l`.

    This algorithm runs in time :math:`O[\sum_l N_l + (B_x^l+B_y^l)E_m^l]` where
    :math:`B_x^l` and :math:`B_y^l` are the number of labels in partitions
    :math:`\bar{\boldsymbol x}` and :math:`\bar{\boldsymbol y}` at level
    :math:`l`, respectively, and :math:`E_m^l \le B_x^lB_y^l` is the number of
    nonzero entries in the contingency table between both partitions.

    Examples
    --------

    >>> x = [np.random.randint(0, 100, 1000), np.random.randint(0, 10, 100), np.random.randint(0, 3, 10)]
    >>> y = [np.random.randint(0, 100, 1000), np.random.randint(0, 10, 100), np.random.randint(0, 3, 10)]
    >>> gt.nested_partition_overlap(x, y)
    0.140018...


    References
    ----------
    .. [peixoto-revealing-2021] Tiago P. Peixoto, "Revealing consensus and
       dissensus between network partitions", Phys. Rev. X 11 021003 (2021)
       :doi:`10.1103/PhysRevX.11.021003`, :arxiv:`2005.13977`
    .. [kuhn_hungarian_1955] H. W. Kuhn, "The Hungarian method for the
       assignment problem," Naval Research Logistics Quarterly 2, 83–97 (1955)
       :doi:`10.1002/nav.3800020109`
    .. [munkres_algorithms_1957] James Munkres, "Algorithms for the Assignment
       and Transportation Problems," Journal of the Society for Industrial and
       Applied Mathematics 5, 32–38 (1957).
       :doi:`10.1137/0105003`

    """
    y = order_nested_partition_labels(y)
    x = align_nested_partition_labels(x, y)
    L = min((len(x), len(y)))
    m = 0
    N = 0
    for l in range(L):
        xl = x[l]
        yl = y[l]
        Nl = min((len(xl), len(yl)))
        null = np.logical_and(xl[:Nl] == -1, y[:Nl] == -1).sum()
        ml = (xl[:Nl] == yl[:Nl]).sum() - null
        Nl = max(((xl != -1).sum(), (yl != -1).sum()))
        N += Nl
        m += ml
    if norm:
        return 1 - (N - m) / (N - L)
    return m

@_handle_props
def contingency_graph(x, y):
    r"""Returns the contingency graph between both partitions.

    Parameters
    ----------
    x : iterable of ``int`` values or :class:`~graph_tool.PropertyMap`
        First partition.
    y : iterable of ``int`` values or :class:`~graph_tool.PropertyMap`
        Second partition.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Contingency graph, containing an internal edge property map ``mrs`` with
        the weights, an internal vertex property map ``label`` with the label
        values, and an internal boolean vertex property map ``partition``
        indicating the partition membership.

    Notes
    -----

    The contingency graph is a bipartite graph with the labels of
    :math:`\boldsymbol x` and :math:`\boldsymbol y` as vertices, and edge
    weights given by

    .. math::

        m_{rs} = \sum_i\delta_{x_i,r}\delta_{y_i,s}.

    This algorithm runs in time :math:`O(N)` where :math:`N` is
    the length of :math:`\boldsymbol x` and :math:`\boldsymbol y`.

    Examples
    --------
    >>> x = np.random.randint(0, 10, 1000)
    >>> y = np.random.randint(0, 10, 1000)
    >>> g = gt.contingency_graph(x, y)
    >>> g.ep.mrs.a
    PropertyArray([ 8, 10,  4, 11,  8, 11, 15, 12, 14, 11,  8, 12,  9,  9, 11,
                   14, 10,  9, 16,  8, 12, 16, 15, 13, 12,  7, 11, 13, 18,  9,
                    9,  9, 14, 10, 11,  8,  6,  7, 11, 11,  8, 11, 14, 12,  8,
                    7,  7,  8,  8, 12, 10, 11,  8, 15,  6, 13, 14, 14,  8, 10,
                    8, 11,  7,  6, 10, 13, 10, 13,  6, 11, 15,  5,  4, 10, 13,
                    8,  8,  9, 14,  8,  6, 11,  7,  8, 15, 10,  9,  7,  8, 11,
                   10, 11,  9,  8,  8, 10,  7,  8,  3,  9], dtype=int32)
    """
    x = np.asarray(x, dtype="int32")
    y = np.asarray(y, dtype="int32")
    g = Graph(directed=False)
    g.ep.mrs = g.new_ep("int32_t")
    g.vp.label = g.new_vp("int32_t")
    g.vp.partition = g.new_vp("bool")
    libinference.get_contingency_graph(g._Graph__graph,
                                       g.vp.label._get_any(),
                                       g.ep.mrs._get_any(),
                                       g.vp.partition._get_any(),
                                       x, y)
    return g

@_handle_props
def shuffle_partition_labels(x):
    r"""Returns a copy of partition ``x``, with the group labels randomly shuffled.

    Parameters
    ----------
    x : iterable of ``int`` values or :class:`~graph_tool.PropertyMap`
        Partition.

    Returns
    -------
    y : :class:`numpy.ndarray` or :class:`~graph_tool.PropertyMap`
        Partition with shuffled labels.

    Examples
    --------
    >>> x = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    >>> gt.shuffle_partition_labels(x)
    array([0, 0, 0, 2, 2, 2, 1, 1, 1], dtype=int32)
    """

    x = np.asarray(x, dtype="int32").copy()
    libinference.partition_shuffle_labels(x, _get_rng())
    return x

@_handle_nprops
def shuffle_nested_partition_labels(x):
    r"""Returns a copy of nested partition ``x``, with the group labels randomly shuffled.

    Parameters
    ----------
    x : iterable iterable of ``int`` values or iterable of :class:`~graph_tool.PropertyMap`
        Partition.

    Returns
    -------
    y : list of :class:`numpy.ndarray` or list of :class:`~graph_tool.PropertyMap`
        Nested partition with shuffled labels.

    Examples
    --------
    >>> x = [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 1], [1, 0]]
    >>> gt.shuffle_nested_partition_labels(x)
    [array([0, 0, 0, 2, 2, 2, 1, 1, 1], dtype=int32), array([1, 0, 1], dtype=int32), array([1, 0], dtype=int32)]
    """

    x = [np.asarray(xl, dtype="int32") for xl in x]
    x = libinference.nested_partition_shuffle_labels(x, _get_rng())
    return x

@_handle_props
def order_partition_labels(x):
    r"""Returns a copy of partition ``x``, with the group labels ordered decreasingly
    according to group size.

    Parameters
    ----------
    x : iterable of ``int`` values or :class:`~graph_tool.PropertyMap`
        Partition.

    Returns
    -------
    y : :class:`numpy.ndarray` or :class:`~graph_tool.PropertyMap`
        Partition with ordered labels.

    Examples
    --------
    >>> x = [0, 2, 2, 1, 1, 1, 2, 2, 2]
    >>> gt.order_partition_labels(x)
    array([2, 0, 0, 1, 1, 1, 0, 0, 0], dtype=int32)
    """

    x = np.asarray(x, dtype="int32").copy()
    libinference.partition_order_labels(x)
    return x

@_handle_nprops
def order_nested_partition_labels(x):
    r"""Returns a copy of nested partition ``x``, with the group labels ordered
    decreasingly according to group size at each level.

    Parameters
    ----------
    x : iterable of iterables of ``int`` values or iterable of :class:`~graph_tool.PropertyMap`
        Partition.

    Returns
    -------
    y : list of :class:`numpy.ndarray` or list of :class:`~graph_tool.PropertyMap`
        Nested partition with ordered labels.

    Examples
    --------
    >>> x = [[0, 2, 2, 1, 1, 1, 2, 2, 2], [1, 1, 0], [1, 1]]
    >>> gt.order_nested_partition_labels(x)
    [array([2, 0, 0, 1, 1, 1, 0, 0, 0], dtype=int32), array([1, 0, 0], dtype=int32), array([0, 0], dtype=int32)]
    """

    x = [np.asarray(xl, dtype="int32") for xl in x]
    x = libinference.nested_partition_order_labels(x)
    return x

@_handle_props
def align_partition_labels(x, y):
    r"""Returns a copy of partition ``x``, with the group labels aligned as to
    maximize the overlap with ``y``.

    Parameters
    ----------
    x : iterable of ``int`` values or :class:`~graph_tool.PropertyMap`
        Partition.
    y : iterables of ``int`` values or :class:`~graph_tool.PropertyMap`
        Reference partition.

    Returns
    -------
    z : :class:`numpy.ndarray` or :class:`~graph_tool.PropertyMap`
        Partition ``x`` with labels aligned to ``y``.

    Notes
    -----
    This algorithm runs in time :math:`O[N + (B_x+B_y)E_m]` where :math:`N` is
    the length of :math:`\boldsymbol x` and :math:`\boldsymbol y`, :math:`B_x`
    and :math:`B_y` are the number of labels in partitions :math:`\boldsymbol x`
    and :math:`\boldsymbol y`, respectively, and :math:`E_m \le B_xB_y` is the
    number of nonzero entries in the contingency table between both partitions.

    Examples
    --------
    >>> x = [0, 2, 2, 1, 1, 1, 2, 3, 2]
    >>> y = gt.shuffle_partition_labels(x)
    >>> print(y)
    [3 1 1 2 2 2 1 0 1]
    >>> gt.align_partition_labels(y, x)
    array([0, 2, 2, 1, 1, 1, 2, 3, 2], dtype=int32)

    References
    ----------
    .. [peixoto-revealing-2021] Tiago P. Peixoto, "Revealing consensus and
       dissensus between network partitions", Phys. Rev. X 11 021003 (2021)
       :doi:`10.1103/PhysRevX.11.021003`, :arxiv:`2005.13977`
    """

    x = np.asarray(x, dtype="int32").copy()
    y = np.asarray(y, dtype="int32")
    libinference.align_partition_labels(x, y)
    return x

@_handle_nprops
def align_nested_partition_labels(x, y):
    r"""Returns a copy of nested partition ``x``, with the group labels aligned as to
    maximize the overlap with ``y``.

    Parameters
    ----------
    x : iterable of iterables of ``int`` values or iterable of :class:`~graph_tool.PropertyMap`
        Nested partition.
    y : iterable of iterables of ``int`` values or iterable of :class:`~graph_tool.PropertyMap`
        Reference nested partition.

    Returns
    -------
    z : list of :class:`numpy.ndarray` or list of :class:`~graph_tool.PropertyMap`
        Nested partition ``x`` with labels aligned to ``y``.

    Notes
    -----
    This algorithm runs in time :math:`O[\sum_l N_l + (B_x^l+B_y^l)E_m^l]` where
    :math:`B_x^l` and :math:`B_y^l` are the number of labels in partitions
    :math:`\bar{\boldsymbol x}` and :math:`\bar{\boldsymbol y}` at level
    :math:`l`, respectively, and :math:`E_m^l \le B_x^lB_y^l` is the number of
    nonzero entries in the contingency table between both partitions.

    Examples
    --------
    >>> x = [[0, 2, 2, 1, 1, 1, 2, 3, 2], [1, 0, 1, 0], [0,0]]
    >>> y = gt.shuffle_nested_partition_labels(x)
    >>> print(y)
    [array([3, 1, 1, 2, 2, 2, 1, 0, 1], dtype=int32), array([0, 1, 0, 1], dtype=int32), array([0, 0], dtype=int32)]
    >>> gt.align_nested_partition_labels(y, x)
    [array([0, 2, 2, 1, 1, 1, 2, 3, 2], dtype=int32), array([1, 0, 1, 0], dtype=int32), array([0, 0], dtype=int32)]
    """

    x = [np.asarray(xl, dtype="int32") for xl in x]
    y = [np.asarray(yl, dtype="int32") for yl in y]
    x = libinference.align_nested_partition_labels(x, y)
    return x

@_parallel
def partition_overlap_center(bs, init=None, relabel_bs=False):
    r"""Find a partition with a maximal overlap to all items of the list of
    partitions given.

    Parameters
    ----------
    bs : list of iterables of ``int`` values or list of :class:`~graph_tool.PropertyMap`
        List of partitions.
    init : iterable of ``int`` values (optional, default: ``None``)
        If given, it will determine the initial partition.
    relabel_bs : ``bool`` (optional, default: ``False``)
        If ``True`` the given list of partitions will be updated with
        relabelled values.

    Returns
    -------
    c : :class:`numpy.ndarray`
        Partition containing the overlap consensus.
    r : ``float``
        Uncertainty in range :math:`[0,1]`.

    Notes
    -----

    This algorithm obtains a partition :math:`\hat{\boldsymbol b}` that has a
    maximal sum of overlaps with all partitions given in ``bs``. It is obtained
    by performing the double maximization:

    .. math::

        \begin{aligned}
        \hat b_i &= \underset{r}{\operatorname{argmax}}\;\sum_m \delta_{\mu_m(b^m_i), r}\\
        \boldsymbol\mu_m &= \underset{\boldsymbol\mu}{\operatorname{argmax}} \sum_rm_{r,\mu(r)}^{(m)},
        \end{aligned}

    where :math:`\boldsymbol\mu` is a bijective mapping between group labels,
    and :math:`m_{rs}^{(m)}` is the contingency table between
    :math:`\hat{\boldsymbol b}` and :math:`\boldsymbol b ^{(m)}`. This algorithm
    simply iterates the above equations, until no further improvement is
    possible.

    The uncertainty is given by:

    .. math::

        r = 1 - \frac{1}{NM} \sum_i \sum_m \delta_{\mu_m(b^m_i), \hat b_i}

    This algorithm runs in time :math:`O[M(N + B^3)]` where :math:`M` is the
    number of partitions, :math:`N` is the length of the partitions and
    :math:`B` is the number of labels used.

    @parallel@

    Examples
    --------
    >>> x = [5, 5, 2, 0, 1, 0, 1, 0, 0, 0, 0]
    >>> bs = []
    >>> for m in range(100):
    ...     y = np.array(x)
    ...     y[np.random.randint(len(y))] = np.random.randint(5)
    ...     bs.append(y)
    >>> bs[:3]
    [array([5, 5, 2, 0, 1, 0, 1, 0, 0, 0, 3]), array([5, 5, 2, 0, 4, 0, 1, 0, 0, 0, 0]), array([5, 5, 1, 0, 1, 0, 1, 0, 0, 0, 0])]
    >>> c, r = gt.partition_overlap_center(bs)
    >>> print(c, r)
    [1 1 2 0 3 0 3 0 0 0 0] 0.06818181...
    >>> gt.align_partition_labels(c, x)
    array([5, 5, 2, 0, 1, 0, 1, 0, 0, 0, 0], dtype=int32)

    References
    ----------
    .. [peixoto-revealing-2021] Tiago P. Peixoto, "Revealing consensus and
       dissensus between network partitions", Phys. Rev. X 11 021003 (2021)
       :doi:`10.1103/PhysRevX.11.021003`, :arxiv:`2005.13977`
    """

    bs = _get_array(bs)
    if not relabel_bs:
        bs = np.array(bs, dtype="int32")
    if init is None:
        c = np.zeros(len(bs[0]), dtype="int32")
    else:
        c = np.asarray(init, dtype="int32")

    r = libinference.partition_overlap_center(bs, c)
    return c, r

@_parallel
def nested_partition_overlap_center(bs, init=None, return_bs=False):
    r"""Find a nested partition with a maximal overlap to all items of the list of
    nested partitions given.

    Parameters
    ----------
    bs : list of list of iterables of ``int`` values or list of list of :class:`~graph_tool.PropertyMap`
        List of nested partitions.
    init : iterable of iterables of ``int`` values (optional, default: ``None``)
        If given, it will determine the initial nested partition.
    return_bs : ``bool`` (optional, default: ``False``)
        If ``True`` the an update list of nested partitions will be return with
        relabelled values.

    Returns
    -------
    c : List of :class:`numpy.ndarray`
        Nested partition containing the overlap consensus.
    r : ``float``
        Uncertainty in range :math:`[0,1]`.
    bs : List of lists of :class:`numpy.ndarray`
        List of relabelled nested partitions.

    Notes
    -----

    This algorithm obtains a nested partition :math:`\hat{\bar{\boldsymbol b}}`
    that has a maximal sum of overlaps with all nested partitions given in
    ``bs``. It is obtained by performing the double maximization:

    .. math::

        \begin{aligned}
        \hat b_i^l &= \underset{r}{\operatorname{argmax}}\;\sum_m \delta_{\mu_m^l(b^{l,m}_i), r}\\
        \boldsymbol\mu_m^l &= \underset{\boldsymbol\mu}{\operatorname{argmax}} \sum_rm_{r,\mu(r)}^{(l,m)},
        \end{aligned}

    where :math:`\boldsymbol\mu` is a bijective mapping between group labels,
    and :math:`m_{rs}^{(l,m)}` is the contingency table between
    :math:`\hat{\boldsymbol b}_l` and :math:`\boldsymbol b ^{(m)}_l`. This algorithm
    simply iterates the above equations, until no further improvement is
    possible.

    The uncertainty is given by:

    .. math::

        r = 1 - \frac{1}{N-L}\sum_l\frac{N_l-1}{N_l}\sum_i\frac{1}{M}\sum_m \delta_{\mu_m(b^{l,m}_i), \hat b_i^l}.

    This algorithm runs in time :math:`O[M\sum_l(N_l + B_l^3)]` where :math:`M`
    is the number of partitions, :math:`N_l` is the length of the partitions and
    :math:`B_l` is the number of labels used, in level :math:`l`.

    @parallel@

    Examples
    --------
    >>> x = [[5, 5, 2, 0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 1, 1]]
    >>> bs = []
    >>> for m in range(100):
    ...     y = [np.array(xl) for xl in x]
    ...     y[0][np.random.randint(len(y[0]))] = np.random.randint(5)
    ...     y[1][np.random.randint(len(y[1]))] = np.random.randint(2)
    ...     bs.append(y)
    >>> bs[:3]
    [[array([5, 5, 2, 0, 1, 0, 1, 0, 0, 0, 3]), array([0, 1, 0, 1, 1, 1])], [array([5, 5, 2, 0, 1, 0, 4, 0, 0, 0, 0]), array([0, 1, 1, 1, 1, 1])], [array([5, 5, 2, 0, 1, 0, 1, 0, 0, 0, 2]), array([0, 1, 0, 1, 1, 1])]]
    >>> c, r = gt.nested_partition_overlap_center(bs)
    >>> print(c, r)
    [array([1, 1, 2, 0, 3, 0, 3, 0, 0, 0, 0], dtype=int32), array([0, 1, 0, 1, 1], dtype=int32)] 0.069385...
    >>> gt.align_nested_partition_labels(c, x)
    [array([5, 5, 2, 0, 1, 0, 1, 0, 0, 0, 0], dtype=int32), array([ 0,  1,  0, -1, -1,  1], dtype=int32)]

    References
    ----------
    .. [peixoto-revealing-2021] Tiago P. Peixoto, "Revealing consensus and
       dissensus between network partitions", Phys. Rev. X 11 021003 (2021)
       :doi:`10.1103/PhysRevX.11.021003`, :arxiv:`2005.13977`
    """

    bs = [[np.asarray(_get_array(bs[m][l]), dtype="int32") for l in range(len(bs[m]))] for m in range(len(bs))]
    if init is None:
        c = [np.zeros(len(bs[0][l]), dtype="int32") for l in range(len(bs[0]))]
    else:
        c = [np.asarray(init[l], dtype="int32") for l in range(len(init))]
    c, bs, r = libinference.nested_partition_overlap_center(bs, c)
    if return_bs:
        return c, r, bs
    else:
        return c, r

@_handle_nprops
def nested_partition_clear_null(x):
    r"""Returns a copy of nested partition ``x`` where the null values ``-1`` are
    replaced with ``0``.

    Parameters
    ----------
    x : iterable of iterables of ``int`` values or list of :class:`~graph_tool.PropertyMap`
        Partition.

    Returns
    -------
    y : list of :class:`numpy.ndarray`
        Nested partition with null values removed.

    Notes
    -----
    This is useful to pass hierarchical partitions to
    :class:`~graph_tool.inference.NestedBlockState`.

    Examples
    --------
    >>> x = [[5, 5, 2, 0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 0, -1, -1, 1]]
    >>> gt.nested_partition_clear_null(x)
    [array([5, 5, 2, 0, 1, 0, 1, 0, 0, 0, 0], dtype=int32), array([0, 1, 0, 0, 0, 1], dtype=int32)]

    """
    x = [np.asarray(x[l], dtype="int32") for l in range(len(x))]
    return libinference.nested_partition_clear_null(x)
