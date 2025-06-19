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

"""
``graph_tool.spectral``
-----------------------

This module contains several linear operators based on network structure, which
useful for spectral analysis.

Sparse matrices
+++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   adjacency
   laplacian
   incidence
   transition
   modularity_matrix
   hashimoto

Operator objects
++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary
   :template: class.rst

   AdjacencyOperator
   LaplacianOperator
   IncidenceOperator
   TransitionOperator
   HashimotoOperator
   CompactHashimotoOperator

"""

from .. import _degree, _prop, Graph, GraphView, _limit_args, Vector_int64_t, \
    Vector_double, _parallel
from .. generation import label_self_loops
import numpy
import scipy.sparse
import scipy.sparse.linalg

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_spectral")

__all__ = ["adjacency", "AdjacencyOperator", "laplacian", "LaplacianOperator",
           "incidence", "IncidenceOperator", "transition", "TransitionOperator",
           "modularity_matrix", "hashimoto", "HashimotoOperator",
           "CompactHashimotoOperator"]

def _operator(f):
    text = """.. note:: :class:`~scipy.sparse.linalg.LinearOperator` vs. sparse matrices

        For many linear algebra computations it is more efficient to pass
        ``operator=True`` to this function. In this case, it will return a
        :class:`scipy.sparse.linalg.LinearOperator` subclass, which implements
        matrix-vector and matrix-matrix multiplication, and is sufficient for
        the sparse linear algebra operations available in the scipy module
        :mod:`scipy.sparse.linalg`. This avoids copying the whole graph as a
        sparse matrix, and performs the multiplication operations in parallel
        (if enabled during compilation) --- see note below.

    @parallel@

        (The above is only applicable if ``operator == True``, and when the
        object returned is used to perform matrix-vector or matrix-matrix
        multiplications.)
    """
    f.__doc__ = f.__doc__.replace("@operator@", text)
    return f

@_parallel
@_operator
def adjacency(g, weight=None, vindex=None, operator=False, csr=True):
    r"""Return the adjacency matrix of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    weight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge property map with the edge weights.
    vindex : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map specifying the row/column indices. If not provided, the
        internal vertex index is used.
    operator : ``bool`` (optional, default: ``False``)
        If ``True``, a :class:`scipy.sparse.linalg.LinearOperator` subclass is
        returned, instead of a sparse matrix.
    csr : ``bool`` (optional, default: ``True``)
        If ``True``, and ``operator`` is ``False``, a
        :class:`scipy.sparse.csr_matrix` sparse matrix is returned, otherwise a
        :class:`scipy.sparse.coo_matrix` is returned instead.

    Returns
    -------
    A : :class:`~scipy.sparse.csr_matrix` or :class:`AdjacencyOperator`
        The (sparse) adjacency matrix.

    Notes
    -----
    For undirected graphs, the adjacency matrix is defined as

    .. math::

        A_{ij} =
        \begin{cases}
            1 & \text{if } (j, i) \in E, \\
            2 & \text{if } i = j \text{ and } (i, i) \in E, \\
            0 & \text{otherwise},
        \end{cases}

    where :math:`E` is the edge set.

    For directed graphs, we have instead simply

    .. math::

        A_{ij} =
        \begin{cases}
            1 & \text{if } (j, i) \in E, \\
            0 & \text{otherwise}.
        \end{cases}

    In the case of weighted edges, the entry values are multiplied by the weight
    of the respective edge.

    In the case of networks with parallel edges, the entries in the matrix
    become simply the edge multiplicities (or twice them for the diagonal, for
    undirected graphs).

    .. note::

        For directed graphs the definition above means that the entry
        :math:`A_{ij}` corresponds to the directed edge :math:`j\to
        i`. Although this is a typical definition in network and graph theory
        literature, many also use the transpose of this matrix.

    @operator@

    Examples
    --------
    .. testsetup::

       import scipy.linalg
       from pylab import *

    >>> g = gt.collection.data["polblogs"]
    >>> A = gt.adjacency(g, operator=True)
    >>> N = g.num_vertices()
    >>> ew1 = scipy.sparse.linalg.eigs(A, k=N//2, which="LR", return_eigenvectors=False)
    >>> ew2 = scipy.sparse.linalg.eigs(A, k=N-N//2, which="SR", return_eigenvectors=False)
    >>> ew = np.concatenate((ew1, ew2))

    >>> figure(figsize=(8, 2))
    <...>
    >>> scatter(real(ew), imag(ew), c=sqrt(abs(ew)), linewidths=0, alpha=0.6)
    <...>
    >>> xlabel(r"$\operatorname{Re}(\lambda)$")
    Text(...)
    >>> ylabel(r"$\operatorname{Im}(\lambda)$")
    Text(...)
    >>> tight_layout()
    >>> savefig("adjacency-spectrum.svg")

    .. figure:: adjacency-spectrum.*
        :align: center

        Adjacency matrix spectrum for the political blogs network.

    References
    ----------
    .. [wikipedia-adjacency] http://en.wikipedia.org/wiki/Adjacency_matrix

    """

    if operator:
        return AdjacencyOperator(g, weight=weight, vindex=vindex)

    if vindex is None:
        if g.get_vertex_filter()[0] is not None:
            vindex = g.new_vertex_property("int64_t")
            vindex.fa = numpy.arange(g.num_vertices())
        else:
            vindex = g.vertex_index

    E = g.num_edges() if g.is_directed() else 2 * g.num_edges()

    data = numpy.zeros(E, dtype="double")
    i = numpy.zeros(E, dtype="int32")
    j = numpy.zeros(E, dtype="int32")

    libgraph_tool_spectral.adjacency(g._Graph__graph, _prop("v", g, vindex),
                                     _prop("e", g, weight), data, i, j)

    if E > 0:
        V = max(g.num_vertices(), max(i.max() + 1, j.max() + 1))
    else:
        V = g.num_vertices()

    m = scipy.sparse.coo_matrix((data, (i,j)), shape=(V, V))
    if csr:
        m = m.tocsr()
    return m

class AdjacencyOperator(scipy.sparse.linalg.LinearOperator):
    def __init__(self, g, weight=None, vindex=None):
        r"""A :class:`scipy.sparse.linalg.LinearOperator` representing the adjacency
        matrix of a graph. See :func:`adjacency` for details."""
        self.g = g
        self.weight = weight

        if vindex is None:
            if g.get_vertex_filter()[0] is not None:
                self.vindex = g.new_vertex_property("int64_t")
                self.vindex.fa = numpy.arange(g.num_vertices())
                N = g.num_vertices()
            else:
                self.vindex = g.vertex_index
                N = g.num_vertices()
        else:
            self.vindex = vindex
            if vindex is vindex.get_graph().vertex_index:
                N = g.num_vertices()
            else:
                N = vindex.fa.max() + 1

        self.shape = (N, N)
        self.dtype = numpy.dtype("float")

    def _matvec(self, x):
        y = numpy.zeros(self.shape[0])
        x = numpy.asarray(x, dtype="float")
        libgraph_tool_spectral.adjacency_matvec(self.g._Graph__graph,
                                                _prop("v", self.g, self.vindex),
                                                _prop("e", self.g, self.weight),
                                                x, y)
        return y

    def _matmat(self, x):
        y = numpy.zeros((self.shape[0], x.shape[1]))
        x = numpy.asarray(x, dtype="float")
        libgraph_tool_spectral.adjacency_matvec(self.g._Graph__graph,
                                                _prop("v", self.g, self.vindex),
                                                _prop("e", self.g, self.weight),
                                                x, y)
        return y

    def _adjoint(self):
        if self.g.is_directed():
            return AdjacencyOperator(GraphView(self.g, reversed=True),
                                     self.weight, self.vindex)
        else:
            return self

@_limit_args({"deg": ["total", "in", "out"]})
@_parallel
@_operator
def laplacian(g, deg="out", norm=False, weight=None, r=1, vindex=None, operator=False,
              csr=True):
    r"""Return the Laplacian (or Bethe Hessian if :math:`r > 1`) matrix of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    deg : str (optional, default: "total")
        Degree to be used, in case of a directed graph.
    norm : bool (optional, default: False)
        Whether to compute the normalized Laplacian.
    weight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge property map with the edge weights.
    r : ``double`` (optional, default: ``1.``)
        Regularization parameter. If :math:`r > 1`, and ``norm`` is ``False``,
        then this corresponds to the Bethe Hessian. (This parameter has an
        effect only for ``norm == False``.)
    vindex : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map specifying the row/column indices. If not provided, the
        internal vertex index is used.
    operator : ``bool`` (optional, default: ``False``)
        If ``True``, a :class:`scipy.sparse.linalg.LinearOperator` subclass is
        returned, instead of a sparse matrix.
    csr : ``bool`` (optional, default: ``True``)
        If ``True``, and ``operator`` is ``False``, a
        :class:`scipy.sparse.csr_matrix` sparse matrix is returned, otherwise a
        :class:`scipy.sparse.coo_matrix` is returned instead.

    Returns
    -------
    L : :class:`~scipy.sparse.csr_matrix` or :class:`LaplacianOperator`
        The (sparse) Laplacian matrix.

    Notes
    -----
    The weighted Laplacian matrix is defined as

    .. math::

        \ell_{ij} =
        \begin{cases}
        \Gamma(v_i) & \text{if } i = j \\
        -w_{ij}     & \text{if } i \neq j \text{ and } (j, i) \in E \\
        0           & \text{otherwise}.
        \end{cases}

    Where :math:`\Gamma(v_i)=\sum_j A_{ij}w_{ij}` is sum of the weights of the
    edges incident on vertex :math:`v_i`.

    In case of :math:`r > 1`, the matrix returned is the Bethe Hessian
    [bethe-hessian]_:

    .. math::

        \ell_{ij} =
        \begin{cases}
        \Gamma(v_i) + (r^2 - 1) & \text{if } i = j \\
        -r w_{ij}     & \text{if } i \neq j \text{ and } (j, i) \in E \\
        0           & \text{otherwise}.
        \end{cases}

    The normalized version is

    .. math::

        \ell_{ij} =
        \begin{cases}
        1         & \text{ if } i = j \text{ and } \Gamma(v_i) \neq 0 \\
        -\frac{w_{ij}}{\sqrt{\Gamma(v_i)\Gamma(v_j)}} & \text{ if } i \neq j \text{ and } (j, i) \in E \\
        0         & \text{otherwise}.
        \end{cases}

    In the case of unweighted edges, it is assumed :math:`w_{ij} = 1`.

    For directed graphs, it is assumed :math:`\Gamma(v_i)=\sum_j A_{ij}w_{ij} +
    \sum_j A_{ji}w_{ji}` if ``deg=="total"``, :math:`\Gamma(v_i)=\sum_j A_{ji}w_{ji}`
    if ``deg=="out"`` or :math:`\Gamma(v_i)=\sum_j A_{ij}w_{ij}` if ``deg=="in"``.

    .. note::

        For directed graphs the definition above means that the entry
        :math:`\ell_{i,j}` corresponds to the directed edge :math:`j\to
        i`. Although this is a typical definition in network and graph theory
        literature, many also use the transpose of this matrix.

    @operator@

    Examples
    --------

    .. testsetup::

       import scipy.linalg
       from pylab import *

    >>> g = gt.collection.data["polblogs"]
    >>> L = gt.laplacian(g, operator=True)
    >>> N = g.num_vertices()
    >>> ew1 = scipy.sparse.linalg.eigs(L, k=N//2, which="LR", return_eigenvectors=False)
    >>> ew2 = scipy.sparse.linalg.eigs(L, k=N-N//2, which="SR", return_eigenvectors=False)
    >>> ew = np.concatenate((ew1, ew2))

    >>> figure(figsize=(8, 2))
    <...>
    >>> scatter(real(ew), imag(ew), c=sqrt(abs(ew)), linewidths=0, alpha=0.6)
    <...>
    >>> xlabel(r"$\operatorname{Re}(\lambda)$")
    Text(...)
    >>> ylabel(r"$\operatorname{Im}(\lambda)$")
    Text(...)
    >>> tight_layout()
    >>> savefig("laplacian-spectrum.svg")

    .. figure:: laplacian-spectrum.*
        :align: center

        Laplacian matrix spectrum for the political blogs network.

    >>> L = gt.laplacian(g, norm=True, operator=True)
    >>> ew1 = scipy.sparse.linalg.eigs(L, k=N//2, which="LR", return_eigenvectors=False)
    >>> ew2 = scipy.sparse.linalg.eigs(L, k=N-N//2, which="SR", return_eigenvectors=False)
    >>> ew = np.concatenate((ew1, ew2))

    >>> figure(figsize=(8, 2))
    <...>
    >>> scatter(real(ew), imag(ew), c=sqrt(abs(ew)), linewidths=0, alpha=0.6)
    <...>
    >>> xlabel(r"$\operatorname{Re}(\lambda)$")
    Text(...)
    >>> ylabel(r"$\operatorname{Im}(\lambda)$")
    Text(...)
    >>> tight_layout()
    >>> savefig("norm-laplacian-spectrum.svg")

    .. figure:: norm-laplacian-spectrum.*
        :align: center

        Normalized Laplacian matrix spectrum for the political blogs network.

    References
    ----------
    .. [wikipedia-laplacian] http://en.wikipedia.org/wiki/Laplacian_matrix
    .. [bethe-hessian] Saade, Alaa, Florent Krzakala, and Lenka
       Zdeborová. "Spectral clustering of graphs with the bethe hessian." Advances
       in Neural Information Processing Systems 27 (2014): 406-414, :arxiv:`1406.1880`,
       https://proceedings.neurips.cc/paper/2014/hash/63923f49e5241343aa7acb6a06a751e7-Abstract.html

    """

    if operator:
        return LaplacianOperator(g, deg=deg, norm=norm, weight=weight, r=r,
                                 vindex=vindex)

    if vindex is None:
        if g.get_vertex_filter()[0] is not None:
            vindex = g.new_vertex_property("int64_t")
            vindex.fa = numpy.arange(g.num_vertices())
        else:
            vindex = g.vertex_index

    V = g.num_vertices()
    nself = int(label_self_loops(g, mark_only=True).a.sum())
    E = g.num_edges() - nself
    if not g.is_directed():
        E *= 2

    N = E + g.num_vertices()
    data = numpy.zeros(N, dtype="double")
    i = numpy.zeros(N, dtype="int32")
    j = numpy.zeros(N, dtype="int32")

    if norm:
        libgraph_tool_spectral.norm_laplacian(g._Graph__graph, _prop("v", g, vindex),
                                              _prop("e", g, weight), deg, data, i, j)
    else:
        libgraph_tool_spectral.laplacian(g._Graph__graph, _prop("v", g, vindex),
                                         _prop("e", g, weight), deg, r, data, i, j)
    if E > 0:
        V = max(g.num_vertices(), max(i.max() + 1, j.max() + 1))
    else:
        V = g.num_vertices()

    m = scipy.sparse.coo_matrix((data, (i, j)), shape=(V, V))
    if csr:
        m = m.tocsr()
    return m

class LaplacianOperator(scipy.sparse.linalg.LinearOperator):

    @_limit_args({"deg": ["total", "in", "out"]})
    def __init__(self, g, weight=None, deg="out", r=1, norm=False, vindex=None):
        r"""A :class:`scipy.sparse.linalg.LinearOperator` representing the laplacian
        matrix of a graph. See :func:`laplacian` for details."""

        self.g = g
        self.weight = weight
        self.r = r

        if vindex is None:
            if g.get_vertex_filter()[0] is not None:
                self.vindex = g.new_vertex_property("int64_t")
                self.vindex.fa = numpy.arange(g.num_vertices())
                N = g.num_vertices()
            else:
                self.vindex = g.vertex_index
                N = g.num_vertices()
        else:
            self.vindex = vindex
            if vindex is vindex.get_graph().vertex_index:
                N = g.num_vertices()
            else:
                N = vindex.fa.max() + 1

        self.shape = (N, N)
        self.deg = deg
        self.d = self.g.degree_property_map(deg, weight)
        if norm:
            idx = self.d.a > 0
            d = g.new_vp("double")
            d.a[idx] = 1./numpy.sqrt(self.d.a[idx])
            self.d = d
        else:
            self.d = self.d.copy("double")
        self.norm = norm
        self.dtype = numpy.dtype("float")

    def _matvec(self, x):
        y = numpy.zeros(self.shape[0])
        x = numpy.asarray(x, dtype="float")
        if self.norm:
            matvec = libgraph_tool_spectral.norm_laplacian_matvec
            matvec(self.g._Graph__graph,
                   _prop("v", self.g, self.vindex),
                   _prop("e", self.g, self.weight),
                   _prop("v", self.g, self.d), x, y)
        else:
            matvec = libgraph_tool_spectral.laplacian_matvec
            matvec(self.g._Graph__graph,
                   _prop("v", self.g, self.vindex),
                   _prop("e", self.g, self.weight),
                   _prop("v", self.g, self.d), self.r, x, y)
        return y

    def _matmat(self, x):
        y = numpy.zeros((self.shape[0], x.shape[1]))
        x = numpy.asarray(x, dtype="float")
        if self.norm:
            matmat = libgraph_tool_spectral.norm_laplacian_matmat
            matmat(self.g._Graph__graph,
                   _prop("v", self.g, self.vindex),
                   _prop("e", self.g, self.weight),
                   _prop("v", self.g, self.d), x, y)
        else:
            matmat = libgraph_tool_spectral.laplacian_matmat
            matmat(self.g._Graph__graph,
                   _prop("v", self.g, self.vindex),
                   _prop("e", self.g, self.weight),
                   _prop("v", self.g, self.d), self.r, x, y)
        return y

    def _adjoint(self):
        if self.g.is_directed():
            deg = self.deg
            if deg == "in":
                deg = "out"
            elif deg == "out":
                deg = "in"
            return LaplacianOperator(GraphView(self.g, reversed=True),
                                     self.weight, deg=deg, norm=self.norm,
                                     vindex=self.vindex)
        else:
            return self


@_parallel
@_operator
def incidence(g, vindex=None, eindex=None, operator=False, csr=True):
    r"""Return the incidence matrix of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    vindex : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map specifying the row indices. If not provided, the
        internal vertex index is used.
    eindex : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge property map specifying the column indices. If not provided, the
        internal edge index is used.
    operator : ``bool`` (optional, default: ``False``)
        If ``True``, a :class:`scipy.sparse.linalg.LinearOperator` subclass is
        returned, instead of a sparse matrix.
    csr : ``bool`` (optional, default: ``True``)
        If ``True``, and ``operator`` is ``False``, a
        :class:`scipy.sparse.csr_matrix` sparse matrix is returned, otherwise a
        :class:`scipy.sparse.coo_matrix` is returned instead.

    Returns
    -------
    a : :class:`~scipy.sparse.csr_matrix` or :class:`IncidenceOperator`
        The (sparse) incidence matrix.

    Notes
    -----
    For undirected graphs, the incidence matrix is defined as

    .. math::

        b_{i,j} =
        \begin{cases}
            1 & \text{if vertex } v_i \text{and edge } e_j \text{ are incident}, \\
            0 & \text{otherwise}
        \end{cases}

    For directed graphs, the definition is

    .. math::

        b_{i,j} =
        \begin{cases}
            1 & \text{if edge } e_j \text{ enters vertex } v_i, \\
            -1 & \text{if edge } e_j \text{ leaves vertex } v_i, \\
            0 & \text{otherwise}
        \end{cases}

    @operator@

    Examples
    --------

    .. testsetup::

       import scipy.linalg
       from pylab import *

    >>> g = gt.collection.data["polblogs"]
    >>> B = gt.incidence(g, operator=True)
    >>> N = g.num_vertices()
    >>> s1 = scipy.sparse.linalg.svds(B, k=N//2, which="LM", return_singular_vectors=False)
    >>> s2 = scipy.sparse.linalg.svds(B, k=N-N//2, which="SM", return_singular_vectors=False)
    >>> s = np.concatenate((s1, s2))
    >>> s.sort()

    >>> figure(figsize=(8, 2))
    <...>
    >>> plot(s, "s")
    [...]
    >>> xlabel(r"$i$")
    Text(...)
    >>> ylabel(r"$\lambda_i$")
    Text(...)
    >>> tight_layout()
    >>> savefig("polblogs-indidence-svd.svg")

    .. figure:: polblogs-indidence-svd.*
        :align: center

        Incidence singular values for the political blogs network.

    References
    ----------
    .. [wikipedia-incidence] http://en.wikipedia.org/wiki/Incidence_matrix
    """

    if operator:
        return IncidenceOperator(g, vindex=vindex, eindex=eindex)

    if vindex is None:
        if g.get_edge_filter()[0] is not None:
            vindex = g.new_vertex_property("int64_t")
            vindex.fa = numpy.arange(g.num_vertices())
        else:
            vindex = g.vertex_index

    if eindex is None:
        if g.get_edge_filter()[0] is not None:
            eindex = g.new_edge_property("int64_t")
            eindex.fa = numpy.arange(g.num_edges())
        else:
            eindex = g.edge_index

    E = g.num_edges()

    if E == 0:
        raise ValueError("Cannot construct incidence matrix for a graph with no edges.")

    data = numpy.zeros(2 * E, dtype="double")
    i = numpy.zeros(2 * E, dtype="int32")
    j = numpy.zeros(2 * E, dtype="int32")

    libgraph_tool_spectral.incidence(g._Graph__graph, _prop("v", g, vindex),
                                     _prop("e", g, eindex), data, i, j)
    m = scipy.sparse.coo_matrix((data, (i,j)))
    if csr:
        m = m.tocsr()
    return m


class IncidenceOperator(scipy.sparse.linalg.LinearOperator):

    def __init__(self, g, vindex=None, eindex=None, transpose=False):
        r"""A :class:`scipy.sparse.linalg.LinearOperator` representing the incidence
        matrix of a graph. See :func:`incidence` for details.
        """

        self.g = g
        self.transpose = transpose
        self.dtype = numpy.dtype("float")

        if vindex is None:
            if g.get_vertex_filter()[0] is not None:
                self.vindex = g.new_vertex_property("int64_t")
                self.vindex.fa = numpy.arange(g.num_vertices())
                N = g.num_vertices()
            else:
                self.vindex = g.vertex_index
                N = g.num_vertices()
        else:
            self.vindex = vindex
            if vindex is vindex.get_graph().vertex_index:
                N = g.num_vertices()
            else:
                N = vindex.fa.max() + 1

        if eindex is None:
            if g.get_edge_filter()[0] is not None:
                self.eindex = g.new_edge_property("int64_t")
                self.eindex.fa = numpy.arange(g.num_edges())
                E = g.num_edges()
            else:
                self.eindex = g.edge_index
                E = g.edge_index_range
        else:
            self.eindex = eindex
            if eindex is g.edge_index:
                E = g.edge_index_range
            else:
                E = self.eindex.fa.max() + 1

        if not transpose:
            self.shape = (N, E)
        else:
            self.shape = (E, N)

    def _matvec(self, x):
        y = numpy.zeros(self.shape[0])
        x = numpy.asarray(x, dtype="float")
        libgraph_tool_spectral.incidence_matvec(self.g._Graph__graph,
                                                _prop("v", self.g, self.vindex),
                                                _prop("e", self.g, self.eindex),
                                                x, y, self.transpose)
        return y

    def _matmat(self, x):
        y = numpy.zeros((self.shape[0], x.shape[1]))
        x = numpy.asarray(x, dtype="float")
        libgraph_tool_spectral.incidence_matmat(self.g._Graph__graph,
                                                _prop("v", self.g, self.vindex),
                                                _prop("e", self.g, self.eindex),
                                                x, y, self.transpose)
        return y

    def _adjoint(self):
        return IncidenceOperator(self.g, vindex=self.vindex, eindex=self.eindex,
                                 transpose=not self.transpose)

@_parallel
@_operator
def transition(g, weight=None, vindex=None, operator=False, csr=True):
    r"""Return the transition matrix of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    weight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge property map with the edge weights.
    vindex : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map specifying the row/column indices. If not provided,
        the internal vertex index is used.
    operator : ``bool`` (optional, default: ``False``)
        If ``True``, a :class:`scipy.sparse.linalg.LinearOperator` subclass is
        returned, instead of a sparse matrix.
    csr : ``bool`` (optional, default: ``True``)
        If ``True``, and ``operator`` is ``False``, a
        :class:`scipy.sparse.csr_matrix` sparse matrix is returned, otherwise a
        :class:`scipy.sparse.coo_matrix` is returned instead.

    Returns
    -------
    T : :class:`~scipy.sparse.csr_matrix` or :class:`TransitionOperator`
        The (sparse) transition matrix.

    Notes
    -----
    The transition matrix is defined as

    .. math::

        T_{ij} = \frac{A_{ij}}{k_j}

    where :math:`k_i = \sum_j A_{ji}`, and :math:`A_{ij}` is the adjacency
    matrix.

    In the case of weighted edges, the values of the adjacency matrix are
    multiplied by the edge weights.

    .. note::

        For directed graphs the definition above means that the entry
        :math:`T_{ij}` corresponds to the directed edge :math:`j\to
        i`. Although this is a typical definition in network and graph theory
        literature, many also use the transpose of this matrix.

    @operator@

    Examples
    --------
    .. testsetup::

       import scipy.linalg
       from pylab import *

    >>> g = gt.collection.data["polblogs"]
    >>> T = gt.transition(g, operator=True)
    >>> N = g.num_vertices()
    >>> ew1 = scipy.sparse.linalg.eigs(T, k=N//2, which="LR", return_eigenvectors=False)
    >>> ew2 = scipy.sparse.linalg.eigs(T, k=N-N//2, which="SR", return_eigenvectors=False)
    >>> ew = np.concatenate((ew1, ew2))

    >>> figure(figsize=(8, 2))
    <...>
    >>> scatter(real(ew), imag(ew), c=sqrt(abs(ew)), linewidths=0, alpha=0.6)
    <...>
    >>> xlabel(r"$\operatorname{Re}(\lambda)$")
    Text(...)
    >>> ylabel(r"$\operatorname{Im}(\lambda)$")
    Text(...)
    >>> tight_layout()
    >>> savefig("transition-spectrum.svg")

    .. figure:: transition-spectrum.*
        :align: center

        Transition matrix spectrum for the political blogs network.

    References
    ----------
    .. [wikipedia-transition] https://en.wikipedia.org/wiki/Stochastic_matrix

    """

    if operator:
        return TransitionOperator(g, weight=weight, vindex=vindex)

    if vindex is None:
        if g.get_vertex_filter()[0] is not None:
            vindex = g.new_vertex_property("int64_t")
            vindex.fa = numpy.arange(g.num_vertices())
        else:
            vindex = g.vertex_index

    E = g.num_edges() if g.is_directed() else 2 * g.num_edges()
    data = numpy.zeros(E, dtype="double")
    i = numpy.zeros(E, dtype="int32")
    j = numpy.zeros(E, dtype="int32")

    libgraph_tool_spectral.transition(g._Graph__graph, _prop("v", g, vindex),
                                      _prop("e", g, weight), data, i, j)

    if E > 0:
        V = max(g.num_vertices(), max(i.max() + 1, j.max() + 1))
    else:
        V = g.num_vertices()
    m = scipy.sparse.coo_matrix((data, (i,j)), shape=(V, V))
    if csr:
        m = m.tocsr()
    return m

class TransitionOperator(scipy.sparse.linalg.LinearOperator):

    def __init__(self, g, weight=None, transpose=False, inv_d=None, vindex=None):
        r"""A :class:`scipy.sparse.linalg.LinearOperator` representing the transition
        matrix of a graph. See :func:`transition` for details.
        """

        self.g = g
        self.weight = weight

        if vindex is None:
            if g.get_vertex_filter()[0] is not None:
                self.vindex = g.new_vertex_property("int64_t")
                self.vindex.fa = numpy.arange(g.num_vertices())
                N = g.num_vertices()
            else:
                self.vindex = g.vertex_index
                N = g.num_vertices()
        else:
            self.vindex = vindex
            if vindex is vindex.get_graph().vertex_index:
                N = g.num_vertices()
            else:
                N = vindex.fa.max() + 1

        self.shape = (N, N)
        if inv_d is None:
            d = self.g.degree_property_map("out", weight)
            nd = g.new_vp("double")
            idx = d.a > 0
            nd.a[idx] = 1/d.a[idx]
            self.d = nd
        else:
            self.d = inv_d.copy("double")
        self.dtype = numpy.dtype("float")
        self.transpose = transpose

    def _matvec(self, x):
        y = numpy.zeros(self.shape[0])
        x = numpy.asarray(x, dtype="float")
        libgraph_tool_spectral.transition_matvec(self.g._Graph__graph,
                                                 _prop("v", self.g, self.vindex),
                                                 _prop("e", self.g, self.weight),
                                                 _prop("v", self.g, self.d), x,
                                                 y, self.transpose)
        return y

    def _matmat(self, x):
        y = numpy.zeros((self.shape[0], x.shape[1]))
        x = numpy.asarray(x, dtype="float")
        libgraph_tool_spectral.transition_matmat(self.g._Graph__graph,
                                                 _prop("v", self.g, self.vindex),
                                                 _prop("e", self.g, self.weight),
                                                 _prop("v", self.g, self.d), x,
                                                 y, self.transpose)
        return y

    def _adjoint(self):
        if self.g.is_directed():
            g = GraphView(self.g, reversed=True)
        else:
            g = self.g
        return TransitionOperator(g, self.weight, inv_d=self.d,
                                  transpose=not self.transpose,
                                  vindex=self.vindex)


@_parallel
@_operator
def modularity_matrix(g, weight=None, vindex=None):
    r"""Return the modularity matrix of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    weight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge property map with the edge weights.
    index : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map specifying the row/column indices. If not provided, the
        internal vertex index is used.

    Returns
    -------
    B : :class:`~scipy.sparse.linalg.LinearOperator`
        The (sparse) modularity matrix, represented as a
        :class:`~scipy.sparse.linalg.LinearOperator`.

    Notes
    -----
    The modularity matrix is defined as

    .. math::

        B_{ij} =  A_{ij} - \frac{k^+_i k^-_j}{2E}

    where :math:`k^+_i = \sum_j A_{ji}`, :math:`k^-_i = \sum_j A_{ij}`,
    :math:`2E=\sum_{ij}A_{ij}` and :math:`A_{ij}` is the adjacency matrix.

    In the case of weighted edges, the values of the adjacency matrix are
    multiplied by the edge weights.

    Examples
    --------

    .. testsetup::

       import scipy.linalg
       from pylab import *

    >>> g = gt.collection.data["polblogs"]
    >>> B = gt.modularity_matrix(g)
    >>> N = g.num_vertices()
    >>> ew1 = scipy.sparse.linalg.eigs(B, k=N//2, which="LR", return_eigenvectors=False)
    >>> ew2 = scipy.sparse.linalg.eigs(B, k=N-N//2, which="SR", return_eigenvectors=False)
    >>> ew = np.concatenate((ew1, ew2))

    >>> figure(figsize=(8, 2))
    <...>
    >>> scatter(real(ew), imag(ew), c=sqrt(abs(ew)), linewidths=0, alpha=0.6)
    <...>
    >>> xlabel(r"$\operatorname{Re}(\lambda)$")
    Text(...)
    >>> ylabel(r"$\operatorname{Im}(\lambda)$")
    Text(...)
    >>> autoscale()
    >>> tight_layout()
    >>> savefig("modularity-spectrum.svg")

    .. figure:: modularity-spectrum.*
        :align: center

        Modularity matrix spectrum for the political blogs network.

    References
    ----------
    .. [newman-modularity]  M. E. J. Newman, M. Girvan, "Finding and evaluating
       community structure in networks", Phys. Rev. E 69, 026113 (2004).
       :doi:`10.1103/PhysRevE.69.026113`
    """

    A = adjacency(g, weight=weight, vindex=vindex, operator=True)
    A_T = A.adjoint()
    if g.is_directed():
        k_in = g.degree_property_map("in", weight=weight).fa
    else:
        k_in = g.degree_property_map("out", weight=weight).fa
    k_out = g.degree_property_map("out", weight=weight).fa

    N = g.num_vertices()
    E2 = float(k_out.sum())

    def matvec(x):
        M = x.shape[0]
        if len(x.shape) > 1:
            x = x.reshape(M)
        nx = A.matvec(x) - k_out * numpy.dot(k_in, x) / E2
        return nx

    def rmatvec(x):
        M = x.shape[0]
        if len(x.shape) > 1:
            x = x.reshape(M)
        nx = A_T.matvec(x) - k_in * numpy.dot(k_out, x) / E2
        return nx

    B = scipy.sparse.linalg.LinearOperator((g.num_vertices(), g.num_vertices()),
                                           matvec=matvec, rmatvec=rmatvec,
                                           dtype="float")

    return B

@_parallel
@_operator
def hashimoto(g, index=None, compact=False, operator=False, csr=True):
    r"""Return the Hashimoto (or non-backtracking) matrix of a graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    index : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Edge property map specifying the row/column indices. If not provided, the
        internal edge index is used.
    compact : ``boolean`` (optional, default: ``False``)
        If ``True``, a compact :math:`2|V|\times 2|V|` version of the matrix is
        returned.
    operator : ``bool`` (optional, default: ``False``)
        If ``True``, a :class:`scipy.sparse.linalg.LinearOperator` subclass is
        returned, instead of a sparse matrix.
    csr : ``bool`` (optional, default: ``True``)
        If ``True``, and ``operator`` is ``False``, a
        :class:`scipy.sparse.csr_matrix` sparse matrix is returned, otherwise a
        :class:`scipy.sparse.coo_matrix` is returned instead.

    Returns
    -------
    H : :class:`~scipy.sparse.csr_matrix` or :class:`HashimotoOperator` or :class:`CompactHashimotoOperator`
        The (sparse) Hashimoto matrix.

    Notes
    -----
    The Hashimoto (a.k.a. non-backtracking) matrix [hashimoto]_ is defined as

    .. math::

        h_{k\to l,i\to j} =
        \begin{cases}
            1 & \text{if } (k,l) \in E, (i,j) \in E, l=i, k\ne j,\\
            0 & \text{otherwise},
        \end{cases}

    where :math:`E` is the edge set. It is therefore a :math:`2|E|\times 2|E|`
    asymmetric square matrix (or :math:`|E|\times |E|` for directed graphs),
    indexed over edge directions.

    If the option ``compact=True`` is passed, the matrix returned has shape
    :math:`2|V|\times 2|V|`, where :math:`|V|` is the number of vertices, and is
    given by

    .. math::

        \boldsymbol h =
        \left(\begin{array}{c|c}
            \boldsymbol A & -\boldsymbol 1 \\ \hline
            \boldsymbol D-\boldsymbol 1 & \boldsymbol 0
        \end{array}\right)

    where :math:`\boldsymbol A` is the adjacency matrix, and :math:`\boldsymbol
    D` is the diagonal matrix with the node degrees [krzakala_spectral]_.

    @operator@

    Examples
    --------
    .. testsetup::

       import scipy.linalg
       from pylab import *

    >>> g = gt.collection.data["football"]
    >>> H = gt.hashimoto(g, operator=True)
    >>> N = 2 * g.num_edges()
    >>> ew1 = scipy.sparse.linalg.eigs(H, k=N//2, which="LR", return_eigenvectors=False)
    >>> ew2 = scipy.sparse.linalg.eigs(H, k=N-N//2, which="SR", return_eigenvectors=False)
    >>> ew = np.concatenate((ew1, ew2))

    >>> figure(figsize=(8, 4))
    <...>
    >>> scatter(real(ew), imag(ew), c=sqrt(abs(ew)), linewidths=0, alpha=0.6)
    <...>
    >>> xlabel(r"$\operatorname{Re}(\lambda)$")
    Text(...)
    >>> ylabel(r"$\operatorname{Im}(\lambda)$")
    Text(...)
    >>> tight_layout()
    >>> savefig("hashimoto-spectrum.svg")

    .. figure:: hashimoto-spectrum.*
        :align: center

        Hashimoto matrix spectrum for the network of American football teams.

    References
    ----------
    .. [hashimoto] Hashimoto, Ki-ichiro. "Zeta functions of finite graphs and
       representations of p-adic groups." Automorphic forms and geometry of
       arithmetic varieties. 1989. 211-280. :DOI:`10.1016/B978-0-12-330580-0.50015-X`
    .. [krzakala_spectral] Florent Krzakala, Cristopher Moore, Elchanan Mossel,
       Joe Neeman, Allan Sly, Lenka Zdeborová, and Pan Zhang, "Spectral redemption
       in clustering sparse networks", PNAS 110 (52) 20935-20940, 2013.
       :doi:`10.1073/pnas.1312486110`, :arxiv:`1306.5550`

    """

    if compact:
        if operator:
            return CompactHashimotoOperator(g)

        i = Vector_int64_t()
        j = Vector_int64_t()
        x = Vector_double()

        libgraph_tool_spectral.compact_nonbacktracking(g._Graph__graph,
                                                       i, j, x)

        N = g.num_vertices(ignore_filter=True)
        m = scipy.sparse.coo_matrix((x, (i.a,j.a)), shape=(2 * N, 2 * N))
    else:
        if operator:
            return HashimotoOperator(g, eindex=index)

        if index is None:
            if g.get_edge_filter()[0] is not None:
                index = g.new_edge_property("int64_t")
                index.fa = numpy.arange(g.num_edges())
                E = index.fa.max() + 1
            else:
                index = g.edge_index
                E = g.edge_index_range

        if not g.is_directed():
            E *= 2

        i = Vector_int64_t()
        j = Vector_int64_t()

        libgraph_tool_spectral.nonbacktracking(g._Graph__graph, _prop("e", g, index),
                                               i, j)

        data = numpy.ones(i.a.shape)
        m = scipy.sparse.coo_matrix((data, (i.a,j.a)), shape=(E, E))
    if csr:
        m = m.tocsr()
    return m


class HashimotoOperator(scipy.sparse.linalg.LinearOperator):

    def __init__(self, g, eindex=None, transpose=False):
        r"""A :class:`scipy.sparse.linalg.LinearOperator` representing the hashimoto
        matrix of a graph. See :func:`hashimoto` for details.
        """

        self.g = g
        if eindex is None:
            if g.get_edge_filter()[0] is not None:
                self.eindex = g.new_edge_property("int64_t")
                self.eindex.fa = numpy.arange(g.num_edges())
                E = g.num_edges()
            else:
                self.eindex = g.edge_index
                E = g.edge_index_range
        else:
            self.eindex = eindex
            if eindex is g.edge_index:
                E = g.edge_index_range
            else:
                E = self.eindex.fa.max() + 1
        if g.is_directed():
            self.shape = (E, E)
        else:
            self.shape = (2 * E, 2 * E)

        self.dtype = numpy.dtype("float")
        self.transpose = transpose

    def _matvec(self, x):
        y = numpy.zeros(self.shape[0])
        x = numpy.asarray(x, dtype="float")
        libgraph_tool_spectral.nonbacktracking_matvec(self.g._Graph__graph,
                                                      _prop("e", self.g, self.eindex),
                                                      x, y, self.transpose)
        return y

    def _matmat(self, x):
        y = numpy.zeros((self.shape[0], x.shape[1]))
        x = numpy.asarray(x, dtype="float")
        libgraph_tool_spectral.nonbacktracking_matmat(self.g._Graph__graph,
                                                      _prop("e", self.g, self.eindex),
                                                      x, y, self.transpose)
        return y

    def _adjoint(self):
        if self.g.is_directed():
            g = GraphView(self.g, reversed=True)
        else:
            g = self.g
        return HashimotoOperator(g, self.eindex, transpose=not self.transpose)

class CompactHashimotoOperator(scipy.sparse.linalg.LinearOperator):

    def __init__(self, g, vindex=None, transpose=False):
        r"""A :class:`scipy.sparse.linalg.LinearOperator` representing the compact
        hashimoto matrix of a graph. See :func:`hashimoto` for details.
        """

        self.g = g
        if vindex is None:
            if g.get_vertex_filter()[0] is not None:
                self.vindex = g.new_vertex_property("int64_t")
                self.vindex.fa = numpy.arange(g.num_vertices())
                N = g.num_vertices()
            else:
                self.vindex = g.vertex_index
                N = g.num_vertices()
        else:
            self.vindex = vindex
            if vindex is vindex.get_graph().vertex_index:
                N = g.num_vertices()
            else:
                N = vindex.fa.max() + 1

        self.shape = (2 * N, 2 * N)
        self.dtype = numpy.dtype("float")
        self.transpose = transpose

    def _matvec(self, x):
        y = numpy.zeros(self.shape[0])
        x = numpy.asarray(x, dtype="float")
        libgraph_tool_spectral.compact_nonbacktracking_matvec(self.g._Graph__graph,
                                                              _prop("v", self.g, self.vindex),
                                                              x, y, self.transpose)
        return y

    def _matmat(self, x):
        y = numpy.zeros((self.shape[0], x.shape[1]))
        x = numpy.asarray(x, dtype="float")
        libgraph_tool_spectral.compact_nonbacktracking_matmat(self.g._Graph__graph,
                                                              _prop("v", self.g, self.vindex),
                                                              x, y, self.transpose)
        return y

    def _adjoint(self):
        if self.g.is_directed():
            g = GraphView(self.g, reversed=True)
        else:
            g = self.g
        return CompactHashimotoOperator(g, self.vindex, transpose=not self.transpose)
