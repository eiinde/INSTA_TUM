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

from .. import Graph, _get_rng, Vector_size_t, _parallel
from .. decorators import _limit_args
from . blockmodel import DictState
from . base_states import *
from . partition_modes import contingency_graph

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

import numpy as np
import math
from scipy.special import gammaln

@entropy_state_signature
class PartitionCentroidState(MCMCState, MultiflipMCMCState, MultilevelMCMCState):
    r"""Obtain the center of a set of partitions, according to the variation of
    information metric or reduced mutual information.

    Parameters
    ----------
    bs : iterable of iterable of ``int``
        List of partitions.
    b : ``list`` or :class:`numpy.ndarray` (optional, default: ``None``)
        Initial partition. If not supplied, a partition into a single group will
        be used.
    RMI : ``bool`` (optional, default: ``False``)
         If ``True``, the reduced mutual information will be used, otherwise the
         variation of information metric will be used instead.
    """

    def __init__(self, bs, b=None, RMI=False):
        EntropyState.__init__(self)
        self.bs = bs = np.asarray(bs, dtype="int32")
        if b is None:
            b = np.zeros(bs.shape[1], dtype="int32")
        self.b = np.array(b, dtype="int32")

        self.g = Graph()
        self.g.add_vertex(bs.shape[1])
        self.bg = self.g
        self._abg = self.bg._get_any()
        self.RMI = RMI
        if self.RMI:
            self._state = libinference.make_rmi_center_state(self)
        else:
            self._state = libinference.make_vi_center_state(self)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        b = copy.deepcopy(self.b, memo)
        bs = copy.deepcopy(self.bs, memo)
        return self.copy(bs=bs, b=b, RMI=self.rmi)

    def copy(self, bs=None, b=None, RMI=None):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""

        return VICentroidState(bs=bs if bs is not None else self.bs,
                               b=b if b is not None else self.b,
                               RMI=RMI if RMI is not None else self.self.RMI)


    def __getstate__(self):
        state = EntropyState.__getstate__(self)
        return dict(state, bs=self.bs, b=self.b, RMI=self.RMI)

    def __setstate__(self, state):
        self.__init__(**state)

    def get_B(self):
        r"Returns the total number of blocks."
        return len(np.unique(self.b))

    def get_Be(self):
        r"""Returns the effective number of blocks, defined as :math:`e^{H}`, with
        :math:`H=-\sum_r\frac{n_r}{N}\ln \frac{n_r}{N}`, where :math:`n_r` is
        the number of nodes in group r.
        """
        w = np.array(np.bincount(self.b), dtype="double")
        w = w[w>0]
        w /= w.sum()
        return np.exp(-(w*log(w)).sum())

    @copy_state_wrap
    def _entropy(self):
        r"""Return the description length (negative joint log-likelihood)."""
        return self._state.entropy()

    def _gen_eargs(self, args):
        return "f"

    def _mcmc_sweep_dispatch(self, mcmc_state):
        if self.RMI:
            return libinference.rmi_mcmc_sweep(mcmc_state, self._state,
                                               _get_rng())
        else:
            return libinference.vi_mcmc_sweep(mcmc_state, self._state,
                                              _get_rng())

    def _multiflip_mcmc_sweep_dispatch(self, mcmc_state):
        if self.RMI:
            return libinference.rmi_multiflip_mcmc_sweep(mcmc_state,
                                                         self._state,
                                                         _get_rng())
        else:
            return libinference.vi_multiflip_mcmc_sweep(mcmc_state, self._state,
                                                        _get_rng())

    def _multilevel_mcmc_sweep_dispatch(self, mcmc_state):
        if self.RMI:
            return libinference.rmi_multilevel_mcmc_sweep(mcmc_state,
                                                          self._state,
                                                          _get_rng())
        else:
            return libinference.vi_multilevel_mcmc_sweep(mcmc_state,
                                                         self._state,
                                                         _get_rng())


def variation_information(x, y, norm=False):
    r"""Returns the variation of information between two partitions.

    Parameters
    ----------
    x : iterable of ``int`` values
        First partition.
    y : iterable of ``int`` values
        Second partition.
    norm : (optional, default: ``False``)
        If ``True``, the result will be normalized in the range :math:`[0,1]`.

    Returns
    -------
    VI : ``float``
        Variation of information value.

    Notes
    -----

    The variation of information [meila_comparing_2003]_ is defined as

    .. math::

        \text{VI}(\boldsymbol x,\boldsymbol y) = -\frac{1}{N}\sum_{rs}m_{rs}\left[\ln\frac{m_{rs}}{n_r} + \ln\frac{m_{rs}}{n_s'}\right],

    with :math:`m_{rs}=\sum_i\delta_{x_i,r}\delta_{y_i,s}` being the contingency
    table between :math:`\boldsymbol x` and :math:`\boldsymbol y`, and
    :math:`n_r=\sum_sm_{rs}` and :math:`n'_s=\sum_rm_{rs}` are the group sizes
    in both partitions.

    If ``norm == True``, the normalized value is returned:

    .. math::

        \frac{\text{VI}(\boldsymbol x,\boldsymbol y)}{\ln N}

    which lies in the unit interval :math:`[0,1]`.

    This algorithm runs in time :math:`O(N)` where :math:`N` is
    the length of :math:`\boldsymbol x` and :math:`\boldsymbol y`.

    Examples
    --------
    >>> x = np.random.randint(0, 10, 1000)
    >>> y = np.random.randint(0, 10, 1000)
    >>> gt.variation_information(x, y)
    4.525389...

    References
    ----------
    .. [meila_comparing_2003] Marina Meilă, "Comparing Clusterings by the
       Variation of Information," in Learning Theory and Kernel Machines,
       Lecture Notes in Computer Science No. 2777, edited by Bernhard Schölkopf
       and Manfred K. Warmuth (Springer Berlin Heidelberg, 2003) pp. 173–187.
       :doi:`10.1007/978-3-540-45167-9_14`

    """
    g = contingency_graph(x, y)
    nr = g.degree_property_map("out", g.ep.mrs)
    nr = nr.fa[nr.fa > 0]
    mrs = g.ep.mrs.fa
    mrs = mrs[mrs > 0]
    N = mrs.sum()
    VI = -2 * (mrs * np.log(mrs)).sum() + (nr * np.log(nr)).sum()
    VI /= N
    if norm:
        VI /= np.log(N)
    return float(VI)

@_limit_args({"avg_method": ["min", "max", "arithmetic", "geometric"]})
def mutual_information(x, y, norm=False, adjusted=False, avg_method="arithmetic"):
    r"""Returns the mutual information between two partitions.

    Parameters
    ----------
    x : iterable of ``int`` values
        First partition.
    y : iterable of ``int`` values
        Second partition.
    norm : (optional, default: ``True``)
        If ``True``, the result will be normalized in the range :math:`[0,1]`.
    adjusted : (optional, default: ``False``)
        If ``True``, the adjusted mutual information will be computed. Note that
        if ``adjusted == True``, the parameter ``norm`` is ignored.
    avg_method : (optional, default: ``"arithmetic"``)
        Determines how the adjusted normalization average should be
        computed. Must be one of: ``"min"``, ``"max"``, ``"arithmethic"``, or
        ``"geometric"``. This option has an effect only if ``adjusted == True``.

    Returns
    -------
    MI : ``float``
        Mutual information value

    Notes
    -----

    The mutual information is defined as

    .. math::

        \text{MI}(\boldsymbol x,\boldsymbol y) = \frac{1}{N}\sum_{rs}m_{rs}\ln\frac{N m_{rs}}{n_rn'_s},

    with :math:`m_{rs}=\sum_i\delta_{x_i,r}\delta_{y_i,s}` being the contingency
    table between :math:`\boldsymbol x` and :math:`\boldsymbol y`, and
    :math:`n_r=\sum_sm_{rs}` and :math:`n'_s=\sum_rm_{rs}` are the group sizes
    in both partitions.

    If ``norm == True``, the normalized value is returned:

    .. math::

        2\frac{\text{MI}(\boldsymbol x,\boldsymbol y)}{H_x + H_y}

    which lies in the unit interval :math:`[0,1]`, and where :math:`H_x =
    -\frac{1}{N}\sum_rn_r\ln\frac{n_r}{N}` and :math:`H_x =
    -\frac{1}{N}\sum_rn'_r\ln\frac{n'_r}{N}`.

    If ``adjusted == True``, the adjusted mutual information is returned, defined
    as [vinh_information_2009]_:

    .. math::

       \text{AMI}(\boldsymbol x,\boldsymbol y) =
       \frac{\text{MI}(\boldsymbol x,\boldsymbol y) - E[\text{MI}(\boldsymbol x,\boldsymbol y)]}
       {\text{avg}(H_x, H_y) - E[\text{MI}(\boldsymbol x,\boldsymbol y)]},

    where

    .. math::

        E[\text{MI}(\boldsymbol x,\boldsymbol y)] =
        \sum_r\sum_s\sum_{m=\max(1,n_r+n'_s-N)}^{\min(n_r, n'_s)}
        \frac{m}{N} \ln \left( \frac{Nm}{n_rn'_s}\right) \times \\
        \frac{n_r!n'_s!(N-n_r)!(N-n'_s)!}{N!m!(n_r-m)!(n'_s-m)!(N-n_r-n'_s+m)!}

    is the expected value of :math:`\text{MI}(\boldsymbol x,\boldsymbol y)` if
    either partition has its labels randomly permuted, and :math:`\text{avg}` is the
    function corresponding to the parameter ``avg_methdod``.

    This algorithm runs in time :math:`O(N)` where :math:`N` is
    the length of :math:`\boldsymbol x` and :math:`\boldsymbol y`.

    @parallel@

        If ``adjusted == False``, the algorith will not run in parallel.

    Examples
    --------
    >>> x = np.random.randint(0, 10, 1000)
    >>> y = np.random.randint(0, 10, 1000)
    >>> gt.mutual_information(x, y)
    0.036350...

    References
    ----------
    .. [vinh_information_2009] Nguyen Xuan Vinh, Julien Epps, and James
       Bailey. 2009. “Information theoretic measures for clusterings comparison:
       is a correction for chance necessary?”, In Proceedings of the 26th Annual
       International Conference on Machine Learning (ICML '09). Association for
       Computing Machinery, New York, NY, USA,
       1073–1080. :doi:`10.1145/1553374.1553511`.

    """
    g = contingency_graph(x, y)
    nr = g.degree_property_map("out", g.ep.mrs)
    part = g.vp.partition.fa
    part = part[nr.fa > 0]
    nr = nr.fa[nr.fa > 0]
    mrs = g.ep.mrs.fa
    mrs = mrs[mrs > 0]
    N = mrs.sum()
    MI = (mrs * np.log(mrs)).sum() - (nr * np.log(nr)).sum()
    MI /= N
    MI += np.log(N)
    if adjusted:
        n1 = nr[part == 0].copy()
        n2 = nr[part == 1].copy()
        EMI = libinference.expected_MI(n1, n2)
        H1 = -(n1 * np.log(n1)).sum() / N + np.log(N)
        H2 = -(n2 * np.log(n2)).sum() / N + np.log(N)
        if avg_method == "max":
            avg = max((H1, H2))
        elif avg_method == "min":
            avg = min((H1, H2))
        elif avg_method == "arithmetic":
            avg = (H1 + H2) / 2
        elif avg_method == "geometric":
            avg = sqrt(H1 * H2)
        R = avg - EMI
        if R < 0:
            R = min((R, -np.finfo("float64").eps))
        else:
            R = max((R, np.finfo("float64").eps))
        MI = (MI - EMI)/ R
    elif norm:
        Hx = -((nr * np.log(nr))[part == 0]).sum() / N + np.log(N)
        Hy = -((nr * np.log(nr))[part == 1]).sum() / N + np.log(N)
        MI /= (Hx + Hy)/2
    return float(MI)

def reduced_mutual_information(x, y, norm=False):
    r"""Returns the reduced mutual information between two partitions.

    Parameters
    ----------
    x : iterable of ``int`` values
        First partition.
    y : iterable of ``int`` values
        Second partition.
    norm : (optional, default: ``True``)
        If ``True``, the result will be normalized such that the maximum
        possible value is one.

    Returns
    -------
    RMI : ``float``
        Reduced mutual information value.

    Notes
    -----

    The reduced mutual information (RMI) [newman_improved_2020]_ is defined as

    .. math::

        \text{RMI}(\boldsymbol x,\boldsymbol y) =
        \frac{1}{N}\left[\ln \frac{N!\prod_{rs}m_{rs}!}{\prod_rn_r!\prod_sn_s'!}
        -\ln\Omega(\boldsymbol n, \boldsymbol n')\right],

    with :math:`m_{rs}=\sum_i\delta_{x_i,r}\delta_{y_i,s}` being the contingency
    table between :math:`\boldsymbol x` and :math:`\boldsymbol y`, and
    :math:`n_r=\sum_sm_{rs}` and :math:`n'_s=\sum_rm_{rs}` are the group sizes
    in both partitions, and :math:`\Omega(\boldsymbol n, \boldsymbol n')` is the
    total number of contingency tables with fixed row and column sums.

    If ``norm == True``, the normalized value is returned:

    .. math::

        \frac{2\ln \frac{N!\prod_{rs}m_{rs}!}{\prod_rn_r!\prod_sn_s'!}
        -2\ln\Omega(\boldsymbol n, \boldsymbol n')}
        {\ln\frac{N!}{\prod_rn_r!} + \ln\frac{N!}{\prod_rn'_r!}
         -\ln\Omega(\boldsymbol n, \boldsymbol n)
         -\ln\Omega(\boldsymbol n', \boldsymbol n')}

    which can take a maximum value of one.

    Note that both the normalized and unormalized RMI values can be negative.

    This algorithm runs in time :math:`O(N)` where :math:`N` is
    the length of :math:`\boldsymbol x` and :math:`\boldsymbol y`.

    Examples
    --------
    >>> x = np.random.randint(0, 10, 1000)
    >>> y = np.random.randint(0, 10, 1000)
    >>> gt.reduced_mutual_information(x, y)
    -0.069938...

    References
    ----------
    .. [newman_improved_2020] M. E. J. Newman, G. T. Cantwell and J.-G. Young,
       "Improved mutual information measure for classification and community
       detection", Phys. Rev. E, 101, 042304 (2020),
       :doi:`10.1103/PhysRevE.101.042304`, :arxiv:`1907.12581`

    """
    g = contingency_graph(x, y)
    nr = g.degree_property_map("out", g.ep.mrs)
    part = g.vp.partition.fa
    part = part[nr.fa > 0]
    nr = nr.fa[nr.fa > 0]
    mrs = g.ep.mrs.fa
    mrs = mrs[mrs > 0]
    N = mrs.sum()
    RMI = gammaln(N + 1) + gammaln(mrs + 1).sum() - gammaln(nr + 1).sum()
    RMI -= libinference.log_omega(nr[part == 0], nr[part == 1])
    if norm:
        RMI *= 2
        aRMI = 2 * gammaln(N + 1)
        aRMI -= gammaln(nr + 1).sum()
        aRMI -= libinference.log_omega(nr[part == 0], nr[part == 0])
        aRMI -= libinference.log_omega(nr[part == 1], nr[part == 1])
        RMI /= aRMI
    else:
        RMI /= N
    return float(RMI)
