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

from .. import _prop, _parallel

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

from numpy import sqrt

@_parallel
def latent_multigraph(g, epsilon=1e-8, max_niter=0, verbose=False):
    r"""Infer latent Poisson multigraph model given an "erased" simple graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used. This is expected to be a simple graph.
    epsilon : ``float`` (optional, default: ``1e-8``)
        Convergence criterion.
    max_niter : ``int`` (optional, default: ``0``)
        Maximum number of iterations allowed (if ``0``, no maximum is assumed).
    verbose : ``boolean`` (optional, default: ``False``)
        If ``True``, display verbose information.

    Returns
    -------
    u : :class:`~graph_tool.Graph`
        Latent graph.
    w : :class:`~graph_tool.EdgePropertyMap`
        Edge property map with inferred edge multiplicities.

    Notes
    -----
    This implements the expectation maximization algorithm described in
    [peixoto-latent-2020]_ which consists in iterating the following steps
    until convergence:

    1. In the "expectation" step we obtain the marginal mean multiedge
       multiplicities via:

       .. math::

           w_{ij} =
           \begin{cases}
               \frac{\theta_i\theta_j}{1-\mathrm{e}^{-\theta_i\theta_j}} & \text{ if } G_{ij} = 1,\\
               \theta_i^2 & \text{ if } i = j,\\
               0 & \text{ otherwise.}
           \end{cases}

    2. In the "maximization" step we use the current values of
       :math:`\boldsymbol w` to update the values of :math:`\boldsymbol \theta`:

       .. math::

            \theta_i = \frac{d_i}{\sqrt{\sum_jd_j}}, \quad\text{ with } d_i = \sum_jw_{ji}. 

    The equations above are adapted accordingly if the supplied graph is
    directed, where we have :math:`\theta_i\theta_j\to\theta_i^-\theta_j^+`,
    :math:`\theta_i^2\to\theta_i^-\theta_i^+`, and
    :math:`\theta_i^{\pm}=\frac{d_i^{\pm}}{\sqrt{\sum_jd_j^{\pm}}}`, with
    :math:`d^+_i = \sum_jw_{ji}` and :math:`d^-_i = \sum_jw_{ij}`.

    A single EM iteration takes time :math:`O(V + E)`.

    @parallel@

    Examples
    --------
    >>> g = gt.collection.data["as-22july06"]
    >>> gt.scalar_assortativity(g, "out")
    (-0.198384..., 0.001338...)
    >>> u, w = gt.latent_multigraph(g)
    >>> gt.scalar_assortativity(u, "out", eweight=w)
    (-0.048426..., 0.034526...)

    References
    ----------
    .. [peixoto-latent-2020] Tiago P. Peixoto, "Latent Poisson models for
       networks with heterogeneous density", Phys. Rev. E 102 012309 (2020)
       :doi:`10.1103/PhysRevE.102.012309`, :arxiv:`2002.07803`
    """

    g = g.copy()
    theta_out = g.degree_property_map("out").copy("double")
    theta_out.fa /= sqrt(theta_out.fa.sum())
    if g.is_directed():
        theta_in = g.degree_property_map("in").copy("double")
        theta_in.fa /= sqrt(theta_in.fa.sum())
    else:
        theta_in = theta_out

    w = g.new_ep("double", 1)

    libinference.latent_multigraph(g._Graph__graph,
                                   _prop("e", g, w),
                                   _prop("v", g, theta_out),
                                   _prop("v", g, theta_in),
                                   epsilon, max_niter, verbose)
    return g, w
