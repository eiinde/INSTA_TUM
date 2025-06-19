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
``graph_tool.stats``
--------------------

This module contains miscellaneous statistical functions.

Summary
+++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   vertex_hist
   edge_hist
   vertex_average
   edge_average
   distance_histogram

"""

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_stats")

from .. import _degree, _prop, _get_rng, GraphView, PropertyMap, _parallel
from .. generation import libgraph_tool_generation
from numpy import *

__all__ = ["vertex_hist", "edge_hist", "vertex_average", "edge_average",
           "distance_histogram"]


@_parallel
def vertex_hist(g, deg, bins=[0, 1], float_count=True):
    """
    Return the vertex histogram of the given degree type or property.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    deg : string or :class:`~graph_tool.VertexPropertyMap`
        Degree or property to be used for the histogram. It can be either "in",
        "out" or "total", for in-, out-, or total degree of the vertices. It can
        also be a vertex property map.
    bins : list of bins (optional, default: [0, 1])
        List of bins to be used for the histogram. The values given represent
        the edges of the bins (i.e. lower and upper bounds). If the list
        contains two values, this will be used to automatically create an
        appropriate bin range, with a constant width given by the second value,
        and starting from the first value.
    float_count : bool (optional, default: True)
        If True, the counts in each histogram bin will be returned as floats. If
        False, they will be returned as integers.

    Returns
    -------
    counts : :class:`numpy.ndarray`
        The bin counts.
    bins : :class:`numpy.ndarray`
        The bin edges.

    See Also
    --------
    edge_hist: Edge histograms.
    vertex_average: Average of vertex properties, degrees.
    edge_average: Average of edge properties.
    distance_histogram : Shortest-distance histogram.

    Notes
    -----
    The algorithm runs in :math:`O(|V|)` time.

    @parallel@

    Examples
    --------
    .. testsetup::

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> from numpy.random import poisson
    >>> g = gt.random_graph(1000, lambda: (poisson(5), poisson(5)))
    >>> print(gt.vertex_hist(g, "out"))
    [array([ 11.,  28.,  88., 141., 169., 165., 151., 111.,  60.,  42.,  20.,
            10.,   3.,   1.]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
          dtype=uint64)]
    """

    ret = libgraph_tool_stats.\
          get_vertex_histogram(g._Graph__graph, _degree(g, deg),
                               [float(x) for x in bins])
    return [array(ret[0], dtype="float64") if float_count else ret[0], ret[1]]


@_parallel
def edge_hist(g, eprop, bins=[0, 1], float_count=True):
    """
    Return the edge histogram of the given property.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    eprop : :class:`~graph_tool.EdgePropertyMap`
        Edge property to be used for the histogram.
    bins : list of bins (optional, default: [0, 1])
        List of bins to be used for the histogram. The values given represent
        the edges of the bins (i.e. lower and upper bounds). If the list
        contains two values, this will be used to automatically create an
        appropriate bin range, with a constant width given by the second value,
        and starting from the first value.
    float_count : bool (optional, default: True)
        If True, the counts in each histogram bin will be returned as floats. If
        False, they will be returned as integers.

    Returns
    -------
    counts : :class:`numpy.ndarray`
        The bin counts.
    bins : :class:`numpy.ndarray`
        The bin edges.

    See Also
    --------
    vertex_hist : Vertex histograms.
    vertex_average : Average of vertex properties, degrees.
    edge_average : Average of edge properties.
    distance_histogram : Shortest-distance histogram.

    Notes
    -----
    The algorithm runs in :math:`O(|E|)` time.

    @parallel@

    Examples
    --------
    .. testsetup::

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> from numpy import arange
    >>> from numpy.random import random
    >>> g = gt.random_graph(1000, lambda: (5, 5))
    >>> eprop = g.new_edge_property("double")
    >>> eprop.get_array()[:] = random(g.num_edges())
    >>> print(gt.edge_hist(g, eprop, linspace(0, 1, 11)))
    [array([525., 504., 502., 502., 467., 499., 531., 471., 520., 479.]), array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])]
    """

    ret = libgraph_tool_stats.\
          get_edge_histogram(g._Graph__graph, _prop("e", g, eprop),
                             [float(x) for x in bins])
    return [array(ret[0], dtype="float64") if float_count else ret[0], ret[1]]


@_parallel
def vertex_average(g, deg):
    """
    Return the average of the given degree or vertex property.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    deg : string or :class:`~graph_tool.VertexPropertyMap`
        Degree or property to be used for the histogram. It can be either "in",
        "out" or "total", for in-, out-, or total degree of the vertices. It can
        also be a vertex property map.

    Returns
    -------
    average : float
        The average of the given degree or property.
    std : float
        The standard deviation of the average.

    See Also
    --------
    vertex_hist : Vertex histograms.
    edge_hist : Edge histograms.
    edge_average : Average of edge properties.
    distance_histogram : Shortest-distance histogram.

    Notes
    -----
    The algorithm runs in :math:`O(|V|)` time.

    @parallel@

    Examples
    --------
    .. testsetup::

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> from numpy.random import poisson
    >>> g = gt.random_graph(1000, lambda: (poisson(5), poisson(5)))
    >>> print(gt.vertex_average(g, "in"))
    (5.028, 0.071240...)
    """

    if isinstance(deg, PropertyMap) and "string" in deg.value_type():
        raise ValueError("Cannot calculate average of property type: " + deg.value_type())
    a, aa, count  = libgraph_tool_stats.\
          get_vertex_average(g._Graph__graph, _degree(g, deg))
    try:
        a = array(a.a)
        aa = array(aa.a)
    except AttributeError:
        pass
    a /= count
    aa = sqrt((aa / count - a ** 2) / count)
    return a, aa


@_parallel
def edge_average(g, eprop):
    """
    Return the average of the given degree or vertex property.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    eprop : :class:`~graph_tool.EdgePropertyMap`
        Edge property to be used for the histogram.

    Returns
    -------
    average : float
        The average of the given property.
    std : float
        The standard deviation of the average.

    See Also
    --------
    vertex_hist : Vertex histograms.
    edge_hist : Edge histograms.
    vertex_average : Average of vertex degree, properties.
    distance_histogram : Shortest-distance histogram.

    Notes
    -----
    The algorithm runs in :math:`O(|E|)` time.

    @parallel@

    Examples
    --------
    .. testsetup::

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> from numpy import arange
    >>> from numpy.random import random
    >>> g = gt.random_graph(1000, lambda: (5, 5))
    >>> eprop = g.new_edge_property("double")
    >>> eprop.get_array()[:] = random(g.num_edges())
    >>> print(gt.edge_average(g, eprop))
    (0.49683199099042286, 0.004095628750806015)
    """

    if "string" in eprop.value_type():
        raise ValueError("Cannot calculate average of property type: " + eprop.value_type())
    g = GraphView(g, directed=True)
    a, aa, count = libgraph_tool_stats.\
          get_edge_average(g._Graph__graph, _prop("e", g, eprop))
    try:
        a = array(a.a)
        aa = array(aa.a)
    except AttributeError:
        pass
    a /= count
    aa = sqrt((aa / count - a ** 2) / count)
    return a, aa

@_parallel
def distance_histogram(g, weight=None, bins=[0, 1], samples=None,
                       float_count=True):
    r"""
    Return the shortest-distance histogram for each vertex pair in the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    weight : :class:`~graph_tool.EdgePropertyMap` (optional, default: None)
        Edge weights.
    bins : list of bins (optional, default: `[0, 1]`)
        List of bins to be used for the histogram. The values given represent
        the edges of the bins (i.e. lower and upper bounds). If the list
        contains two values, this will be used to automatically create an
        appropriate bin range, with a constant width given by the second value,
        and starting from the first value.
    samples : int (optional, default: `None`)
        If supplied, the distances will be randomly sampled from a number of
        source vertices given by this parameter. If `samples is None` (default),
        all pairs are used.
    float_count : bool (optional, default: `True`)
        If True, the counts in each histogram bin will be returned as floats. If
        False, they will be returned as integers.

    Returns
    -------
    counts : :class:`numpy.ndarray`
        The bin counts.
    bins : :class:`numpy.ndarray`
        The bin edges.

    See Also
    --------
    vertex_hist : Vertex histograms.
    edge_hist : Edge histograms.
    vertex_average : Average of vertex degree, properties.
    distance_histogram : Shortest-distance histogram.

    Notes
    -----
    The algorithm runs in :math:`O(V^2)` time, or :math:`O(V^2\log V)` if
    `weight is not None`. If `samples` is supplied, the complexities are
    :math:`O(\text{samples}\times V)`  and
    :math:`O(\text{samples}\times V\log V)`, respectively.

    @parallel@

    Examples
    --------
    .. testsetup::

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(100, lambda: (3, 3))
    >>> hist = gt.distance_histogram(g)
    >>> print(hist)
    [array([   0.,  300.,  862., 2205., 3787., 2532.,  214.]), array([0, 1, 2, 3, 4, 5, 6, 7], dtype=uint64)]
    >>> hist = gt.distance_histogram(g, samples=10)
    >>> print(hist)
    [array([  0.,  30.,  87., 224., 377., 253.,  19.]), array([0, 1, 2, 3, 4, 5, 6, 7], dtype=uint64)]

    """

    if samples is not None:
        ret = libgraph_tool_stats.\
              sampled_distance_histogram(g._Graph__graph,
                                         _prop("e", g, weight),
                                         [float(x) for x in bins],
                                         samples, _get_rng())
    else:
        ret = libgraph_tool_stats.\
              distance_histogram(g._Graph__graph, _prop("e", g, weight), bins)
    return [array(ret[0], dtype="float64") if float_count else ret[0], ret[1]]
