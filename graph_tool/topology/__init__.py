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
``graph_tool.topology``
-----------------------

This module contains various functions that assess the graph topology.

Distance and paths
++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   shortest_distance
   shortest_path
   random_shortest_path
   count_shortest_paths
   all_shortest_paths
   all_predecessors
   all_paths
   all_circuits
   pseudo_diameter

Graph comparison
++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   similarity
   vertex_similarity
   isomorphism
   subgraph_isomorphism
   mark_subgraph
   max_cliques

Matching and independent sets
+++++++++++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   max_cardinality_matching
   max_independent_vertex_set

Spanning tree
+++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   min_spanning_tree
   random_spanning_tree

Sorting and closure
+++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   dominator_tree
   topological_sort
   transitive_closure

Components and connectivity
+++++++++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   label_components
   label_biconnected_components
   label_largest_component
   extract_largest_component
   label_out_component
   vertex_percolation
   edge_percolation
   kcore_decomposition

Graph classification
++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   is_bipartite
   is_DAG
   is_planar
   make_maximal_planar

Directionality
++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   edge_reciprocity

Combinatorial optimizaton
+++++++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   tsp_tour
   sequential_vertex_coloring

"""

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_topology")

from .. import _prop, Vector_int32_t, Vector_size_t, _check_prop_writable, \
     _check_prop_scalar, _check_prop_vector, Graph, VertexPropertyMap, \
     PropertyMap, GraphView, libcore, _get_rng, perfect_prop_hash, \
     _limit_args, _parallel
from .. generation import label_self_loops
import numpy, collections.abc

__all__ = ["isomorphism", "subgraph_isomorphism", "mark_subgraph",
           "max_cliques", "max_cardinality_matching",
           "max_independent_vertex_set", "min_spanning_tree",
           "random_spanning_tree", "dominator_tree", "topological_sort",
           "transitive_closure", "tsp_tour", "sequential_vertex_coloring",
           "label_components", "label_largest_component",
           "extract_largest_component", "label_biconnected_components",
           "label_out_component", "vertex_percolation", "edge_percolation",
           "kcore_decomposition", "shortest_distance", "shortest_path",
           "random_shortest_path", "count_shortest_paths", "all_shortest_paths",
           "all_predecessors", "all_paths", "all_circuits", "pseudo_diameter",
           "is_bipartite", "is_DAG", "is_planar", "make_maximal_planar",
           "similarity", "vertex_similarity", "edge_reciprocity"]

@_parallel
def similarity(g1, g2, eweight1=None, eweight2=None, label1=None, label2=None,
               norm=True, p=1., distance=False, asymmetric=False):
    r"""Return the Jaccard adjacency similarity between two graphs.

    Parameters
    ----------
    g1 : :class:`~graph_tool.Graph`
        First graph to be compared.
    g2 : :class:`~graph_tool.Graph`
        Second graph to be compared.
    eweight1 : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge weights for the first graph to be used in comparison.
    eweight2 : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge weights for the second graph to be used in comparison.
    label1 : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex labels for the first graph to be used in comparison. If not
        supplied, the vertex indices are used.
    label2 : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex labels for the second graph to be used in comparison. If not
        supplied, the vertex indices are used.
    norm : bool (optional, default: ``True``)
        If ``True``, the returned value is normalized by the total number of
        edges.
    p : float (optional, default: ``1.``)
        Exponent of the :math:`L^p` distance function.
    distance : bool (optional, default: ``False``)
        If ``True``, the complementary value is returned, i.e. the distance
        between the two graphs.
    asymmetric : bool (optional, default: ``False``)
        If ``True``, the asymmetric similarity of ``g1`` to ``g2`` will be
        computed.

    Returns
    -------
    similarity : float
        Adjacency similarity value.

    Notes
    -----
    In its default parametrization, the Jaccard adjacency similarity is the sum
    of equal non-zero entries in the adjacency matrices of both graphs, given a
    vertex ordering determined by the vertex labels. In other words, it counts
    the number of edges which have the same source and target labels in both
    graphs.

    If ``norm == True`` (the default) the value returned is the total fraction
    of edges of both networks that match.

    This function also allows for generalized similarities according to an
    :math:`L^p` norm, for arbitrary exponent :math:`p`.

    More specifically, the (unormalized) adjacency similarity is defined as:

    .. math::

       S(\boldsymbol A_1, \boldsymbol A_2) = E - d(\boldsymbol A_1, \boldsymbol A_2)

    where

    .. math::

       d(\boldsymbol A_1, \boldsymbol A_2) = \left(\sum_{i\le j} \left|A_{ij}^{(1)} - A_{ij}^{(2)}\right|^p\right)^{1/p}

    is the corresponding distance between graphs, and :math:`E=(\sum_{i\le
    j}|A_{ij}^{(1)}|^p + |A_{ij}^{(2)}|^p)^{1/p}`. Unless otherwise stated via
    the parameter ``p``, the exponent used is :math:`p=1`. This definition holds
    for undirected graphs, otherwise the sums go over all directed pairs. If
    edge weights are provided, the weighted adjacency matrix is used.

    If a multigraph is passed, the matrix entries :math:`A^{(1)}_{ij}` and
    :math:`A^{(2)}_{ij}` correspond to the edge multiplicities between nodes
    :math:`i` and :math:`j` in each graph.

    If ``norm == True`` the value returned is :math:`S(\boldsymbol A_1,
    \boldsymbol A_2) / E`, which lies in the interval :math:`[0,1]`.

    If ``asymmetric == True``, the above is changed so that the comparison is
    made only for entries in :math:`\boldsymbol A_1` that are larger than in :math:`\boldsymbol A_2`, i.e.

    .. math::

       d(\boldsymbol A_1, \boldsymbol A_2) = \left(\sum_{i\le j} \left|A_{ij}^{(1)} - A_{ij}^{(2)}\right|^p H(A_{ij}^{(1)} - A_{ij}^{(2)})\right)^{1/p},

    where :math:`H(x) = \{1 \text{ if } x > 0; 0 \text{ otherwise}\}` is the
    unit step function, and the total sum is changed accordingly to
    :math:`E=\left(\sum_{i\le j}|A_{ij}^{(1)}|^p\right)^{1/p}`.

    **Relation to set operations**

    If the graph is unweighted, :math:`p=1`, and ``norm == False`` the
    algorithm is equivalent to the following set comparisons:

    1. If ``distance == True`` the returned value is equal to

       .. math::

            2 |\boldsymbol A_1 \cup \boldsymbol A_2|

       where :math:`\boldsymbol A_1 \cup \boldsymbol A_2` is the union of the
       edges in :math:`\boldsymbol A_1` and :math:`\boldsymbol A_2`.

    2. If ``distance == False`` the returned value is equal to

       .. math::

            2 |\boldsymbol A_1 \cap \boldsymbol A_2|

       where :math:`\boldsymbol A_1 \cap \boldsymbol A_2` is the intersection of
       the edges in :math:`\boldsymbol A_1` and :math:`\boldsymbol A_2`.

    3. If ``distance == True`` and ``asymmetric == True`` the returned value is
       equal to

       .. math::

            |\boldsymbol A_1 \setminus \boldsymbol A_2|

       where :math:`\boldsymbol A_1 \setminus \boldsymbol A_2` is the set of
       edges in :math:`\boldsymbol A_1` that are not also in :math:`\boldsymbol
       A_2`.

    The algorithm runs with complexity :math:`O(E_1 + V_1 + E_2 + V_2)`.

    @parallel@

        (The above is applicable only if the vertex labels are integers bounded
        by the sizes of the graphs.)

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(100, lambda: (3,3))
    >>> u = g.copy()
    >>> gt.similarity(u, g)
    1.0
    >>> gt.random_rewire(u)
    190
    >>> gt.similarity(u, g)
    0.03

    """

    if label1 is None:
        label1 = g1.vertex_index
    if label2 is None:
        label2 = g2.vertex_index

    _check_prop_scalar(label1, name="label1")
    _check_prop_scalar(label2, name="label2")

    if label1.value_type() != label2.value_type():
        try:
            label2 = label2.copy(label1.value_type())
        except ValueError:
            label1 = label1.copy(label2.value_type())

    if eweight1 is None and eweight2 is None:
        ew1 = ew2 = libcore.any()
    else:
        if eweight1 is None:
            eweight1 = g1.new_ep(eweight2.value_type(), 1)
        if eweight2 is None:
            eweight2 = g2.new_ep(eweight1.value_type(), 1)

        _check_prop_scalar(eweight1, name="eweight1")
        _check_prop_scalar(eweight2, name="eweight2")

        if eweight1.value_type() != eweight2.value_type():
            try:
                eweight2 = eweight2.copy(eweight1.value_type())
            except ValueError:
                eweight1 = eweight1.copy(eweight2.value_type())

        ew1 = _prop("e", g1, eweight1)
        ew2 = _prop("e", g2, eweight2)

    if ((label1.is_writable() and label1.fa.max() > g1.num_vertices()) or
        (label2.is_writable() and label2.fa.max() > g2.num_vertices())):
        s = libgraph_tool_topology.\
               similarity(g1._Graph__graph, g2._Graph__graph,
                          ew1, ew2, _prop("v", g1, label1),
                          _prop("v", g2, label2), p, asymmetric)
    else:
        s = libgraph_tool_topology.\
               similarity_fast(g1._Graph__graph, g2._Graph__graph,
                               ew1, ew2, _prop("v", g1, label1),
                               _prop("v", g2, label2), p, asymmetric)

    if not g1.is_directed() or not g2.is_directed():
        s /= 2

    s **= 1./p

    if eweight1 is None and eweight2 is None:
        if asymmetric:
            E = g1.num_edges()
        else:
            E = g1.num_edges() + g2.num_edges()
    else:
        if asymmetric:
            E = float((abs(eweight1.fa)**p).sum()) ** (1./p)
        else:
            E = float((abs(eweight1.fa)**p).sum() +
                      (abs(eweight2.fa)**p).sum()) ** (1./p)
    if not distance:
        s = E - s
    if norm:
        return s / E if E > 0 else numpy.nan
    return s

@_parallel
@_limit_args({"sim_type": ["dice", "salton", "hub-promoted", "hub-suppressed",
                           "jaccard", "inv-log-weight", "resource-allocation",
                           "leicht-holme-newman"]})
def vertex_similarity(g, sim_type="jaccard", vertex_pairs=None, eweight=None,
                      sim_map=None):
    r"""Return the similarity between pairs of vertices.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        The graph to be used.
    sim_type : ``str`` (optional, default: ``"jaccard"``)
        Type of similarity to use. This must be one of ``"dice"``, ``"salton"``,
        ``"hub-promoted"``, ``"hub-suppressed"``, ``"jaccard"``,
        ``"inv-log-weight"``, ``"resource-allocation"`` or ``"leicht-holme-newman"``.
    vertex_pairs : iterable of pairs of integers (optional, default: ``None``)
        Pairs of vertices to compute the similarity. If omitted, all pairs will
        be considered.
    eweight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge weights.
    sim_map : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        If provided, and ``vertex_pairs is None``, the vertex similarities will
        be stored in this vector-valued property. Otherwise, a new one will be
        created.

    Returns
    -------
    similarities : :class:`numpy.ndarray` or :class:`~graph_tool.VertexPropertyMap`
        If ``vertex_pairs`` was supplied, this will be a :class:`numpy.ndarray`
        with the corresponding similarities, otherwise it will be a
        vector-valued vertex :class:`~graph_tool.VertexPropertyMap`, with the
        similarities to all other vertices.

    Notes
    -----
    According to ``sim_type``, this function computes one of the following
    similarities:

    ``sim_type == "dice"``

       The Sørensen–Dice similarity [sorensen-dice]_ of vertices :math:`u` and
       :math:`v` is defined as

       .. math::

           \frac{2|\Gamma(u)\cap\Gamma(v)|}{|\Gamma(u)|+|\Gamma(v)|},

       where :math:`\Gamma(u)` is the set of neighbors of vertex :math:`u`.

    ``sim_type == "salton"``

       The Salton (or cosine) similarity [salton]_ of vertices :math:`u` and
       :math:`v` is defined as

       .. math::

           \frac{|\Gamma(u)\cap\Gamma(v)|}{\sqrt{|\Gamma(u)||\Gamma(v)|}},

       where :math:`\Gamma(u)` is the set of neighbors of vertex :math:`u`.

    ``sim_type == "hub-promoted"``

       The "hub promoted" similarity [ravasz_hierarchical_2002]_ of vertices
       :math:`u` and :math:`v` is defined as

       .. math::

           \frac{|\Gamma(u)\cap\Gamma(v)|}{\max(|\Gamma(u)|,|\Gamma(v)|)},

       where :math:`\Gamma(u)` is the set of neighbors of vertex :math:`u`.

    ``sim_type == "hub-suppressed"``

       The "hub suppressed" similarity of vertices :math:`u` and
       :math:`v` is defined as

       .. math::

           \frac{|\Gamma(u)\cap\Gamma(v)|}{\min(|\Gamma(u)|,|\Gamma(v)|)},

       where :math:`\Gamma(u)` is the set of neighbors of vertex :math:`u`.

    ``sim_type == "jaccard"``

       The Jaccard similarity [jaccard]_ of vertices :math:`u` and
       :math:`v` is defined as

       .. math::

           \frac{|\Gamma(u)\cap\Gamma(v)|}{|\Gamma(u)\cup\Gamma(v)|},

       where :math:`\Gamma(u)` is the set of neighbors of vertex :math:`u`.

    ``sim_type == "inv-log-weight"``

       The inverse log weighted similarity [adamic-friends-2003]_ of vertices
       :math:`u` and :math:`v` is defined as

       .. math::

           \sum_{w \in \Gamma(u)\cap\Gamma(v)}\frac{1}{\log |\Gamma(w)|},

       where :math:`\Gamma(u)` is the set of neighbors of vertex :math:`u`.

    ``sim_type == "resource-allocation"``

       The resource allocation similarity [zhou-predicting-2009]_ of vertices
       :math:`u` and :math:`v` is defined as

       .. math::

           \sum_{w \in \Gamma(u)\cap\Gamma(v)}\frac{1}{|\Gamma(w)|},

       where :math:`\Gamma(u)` is the set of neighbors of vertex :math:`u`.

    ``sim_type == "leicht-holme-newman"``

       The Leicht-Holme-Newman similarity [leicht_vertex_2006]_ of vertices
       :math:`u` and :math:`v` is defined as

       .. math::

           \frac{|\Gamma(u)\cap\Gamma(v)|}{|\Gamma(u)||\Gamma(v)|},

       where :math:`\Gamma(u)` is the set of neighbors of vertex :math:`u`.

    For directed graphs, only out-neighbors are considered in the above
    algorthms (for "inv-log-weight", the in-degrees are used to compute the
    weights). To use the in-neighbors instead, a :class:`~graph_tool.GraphView`
    should be used to reverse the graph, e.g. ``vertex_similarity(GraphView(g,
    reversed=True))``.

    For weighted or multigraphs, in the above equations it is assumed the
    following:

    .. math::

        |\Gamma(u)\cap\Gamma(v)| &= \sum_w \min(A_{wv}, A_{wu}),\\
        |\Gamma(u)\cup\Gamma(v)| &= \sum_w \max(A_{wv}, A_{wu}),\\
        |\Gamma(u)| &= \sum_w A_{wu},

    where :math:`A_{wu}` is the weighted adjacency matrix.

    See [liben-nowell-link-prediction-2007]_ for a review of the above.

    The algorithm runs with complexity :math:`O(\left<k\right>N^2)` if
    ``vertex_pairs is None``, otherwise with :math:`O(\left<k\right>P)` where
    :math:`P` is the length of ``vertex_pairs``.

    @parallel@

    Examples
    --------
    .. testcode::
       :hide:

       import matplotlib

    >>> g = gt.collection.data["polbooks"]
    >>> s = gt.vertex_similarity(g, "jaccard")
    >>> color = g.new_vp("double")
    >>> color.a = s[0].a
    >>> gt.graph_draw(g, pos=g.vp.pos, vertex_text=g.vertex_index,
    ...               vertex_color=color, vertex_fill_color=color,
    ...               vcmap=matplotlib.cm.inferno,
    ...               output="polbooks-jaccard.svg")
    <...>

    .. figure:: polbooks-jaccard.*
       :align: center

       Jaccard similarities to vertex ``0`` in a political books network.

    References
    ----------
    .. [sorensen-dice] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    .. [salton] G. Salton, M. J. McGill, "Introduction to Modern Informa-tion Retrieval",
       (MuGraw-Hill, Auckland, 1983).
    .. [ravasz_hierarchical_2002] Ravasz, E., Somera, A. L., Mongru, D. A.,
       Oltvai, Z. N., & Barabási, A. L., "Hierarchical organization of
       modularity in metabolic networks", Science, 297(5586), 1551-1555,
       (2002). :doi:`10.1126/science.1073374`
    .. [jaccard] https://en.wikipedia.org/wiki/Jaccard_index
    .. [leicht_vertex_2006] E. A. Leicht, Petter Holme, and M. E. J. Newman,
       "Vertex similarity in networks", Phys. Rev. E 73, 026120 (2006),
       :doi:`10.1103/PhysRevE.73.026120`, :arxiv:`physics/0510143`
    .. [adamic-friends-2003] Lada A. Adamic and Eytan Adar, "Friends and neighbors
       on the Web", Social Networks Volume 25, Issue 3, Pages 211–230 (2003)
       :doi:`10.1016/S0378-8733(03)00009-1`
    .. [liben-nowell-link-prediction-2007] David Liben-Nowell and Jon Kleinberg,
       "The link-prediction problem for social networks", Journal of the
       American Society for Information Science and Technology, Volume 58, Issue
       7, pages 1019–1031 (2007), :doi:`10.1002/asi.20591`
    .. [zhou-predicting-2009] Zhou, Tao, Linyuan Lü, and Yi-Cheng Zhang,
       "Predicting missing links via local information", The European Physical
       Journal B 71, no. 4: 623-630 (2009), :doi:`10.1140/epjb/e2009-00335-8`,
       :arxiv:`0901.0553`

    """

    if eweight is None:
        eweight = libcore.any()
    else:
        eweight = _prop("e", g, eweight)

    if vertex_pairs is None:
        if sim_map is None:
            s = g.new_vp("vector<double>")
        else:
            s = sim_map
        if sim_type == "dice":
            libgraph_tool_topology.dice_similarity(g._Graph__graph,
                                                   _prop("v", g, s),
                                                   eweight)
        elif sim_type == "salton":
            libgraph_tool_topology.salton_similarity(g._Graph__graph,
                                                     _prop("v", g, s),
                                                     eweight)
        elif sim_type == "hub-promoted":
            libgraph_tool_topology.hub_promoted_similarity(g._Graph__graph,
                                                           _prop("v", g, s),
                                                           eweight)
        elif sim_type == "hub-suppressed":
            libgraph_tool_topology.hub_suppressed_similarity(g._Graph__graph,
                                                             _prop("v", g, s),
            eweight)
        elif sim_type == "jaccard":
            libgraph_tool_topology.jaccard_similarity(g._Graph__graph,
                                                      _prop("v", g, s),
                                                      eweight)
        elif sim_type == "inv-log-weight":
            libgraph_tool_topology.inv_log_weight_similarity(g._Graph__graph,
                                                             _prop("v", g, s),
                                                             eweight)
        elif sim_type == "resource-allocation":
            libgraph_tool_topology.r_allocation_similarity(g._Graph__graph,
                                                           _prop("v", g, s),
                                                           eweight)
        elif sim_type == "leicht-holme-newman":
            libgraph_tool_topology.leicht_holme_newman_similarity(g._Graph__graph,
                                                           _prop("v", g, s),
                                                           eweight)
    else:
        vertex_pairs = numpy.asarray(vertex_pairs, dtype="int64")
        s = numpy.zeros(vertex_pairs.shape[0], dtype="double")
        if sim_type == "dice":
            libgraph_tool_topology.dice_similarity_pairs(g._Graph__graph,
                                                         vertex_pairs,
                                                         s, eweight)
        elif sim_type == "salton":
            libgraph_tool_topology.salton_similarity_pairs(g._Graph__graph,
                                                           vertex_pairs,
                                                           s, eweight)
        elif sim_type == "hub-promoted":
            libgraph_tool_topology.hub_promoted_similarity_pairs(g._Graph__graph,
                                                                 vertex_pairs,
                                                                 s, eweight)
        elif sim_type == "hub-suppressed":
            libgraph_tool_topology.hub_suppressed_similarity_pairs(g._Graph__graph,
                                                                   vertex_pairs,
                                                                   s, eweight)
        elif sim_type == "jaccard":
            libgraph_tool_topology.jaccard_similarity_pairs(g._Graph__graph,
                                                            vertex_pairs,
                                                            s, eweight)
        elif sim_type == "inv-log-weight":
            libgraph_tool_topology.\
                inv_log_weight_similarity_pairs(g._Graph__graph, vertex_pairs,
                                                s, eweight)
        elif sim_type == "resource-allocation":
            libgraph_tool_topology.\
                r_allocation_similarity_pairs(g._Graph__graph, vertex_pairs,
                                              s, eweight)
        elif sim_type == "leicht-holme-newman":
            libgraph_tool_topology.\
                leicht_holme_newman_similarity_pairs(g._Graph__graph, vertex_pairs,
                                              s, eweight)
    return s


def isomorphism(g1, g2, vertex_inv1=None, vertex_inv2=None, isomap=False):
    r"""Check whether two graphs are isomorphic.

    Parameters
    ----------
    g1 : :class:`~graph_tool.Graph`
        First graph.
    g2 : :class:`~graph_tool.Graph`
        Second graph.
    vertex_inv1 : :class:`~graph_tool.VertexPropertyMap` (optional, default: `None`)
        Vertex invariant of the first graph. Only vertices with with the same
        invariants are considered in the isomorphism.
    vertex_inv2 : :class:`~graph_tool.VertexPropertyMap` (optional, default: `None`)
        Vertex invariant of the second graph. Only vertices with with the same
        invariants are considered in the isomorphism.
    isomap : ``bool`` (optional, default: ``False``)
        If ``True``, a :class:`~graph_tool.VertexPropertyMap` with the
        isomorphism mapping is returned as well.

    Returns
    -------
    is_isomorphism : ``bool``
        ``True`` if both graphs are isomorphic, otherwise ``False``.
    isomap : :class:`~graph_tool.VertexPropertyMap`
         Isomorphism mapping corresponding to a property map belonging to the
         first graph which maps its vertices to their corresponding vertices of
         the second graph.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(100, lambda: (3,3))
    >>> g2 = gt.Graph(g)
    >>> gt.isomorphism(g, g2)
    True
    >>> g.add_edge(g.vertex(0), g.vertex(1))
    <...>
    >>> gt.isomorphism(g, g2)
    False

    """
    imap = g1.new_vertex_property("int32_t")
    if vertex_inv1 is None:
        vertex_inv1 = g1.degree_property_map("total").copy("int64_t")
    else:
        vertex_inv1 = perfect_prop_hash([vertex_inv1])[0].copy("int64_t")
    if vertex_inv2 is None:
        vertex_inv2 = g2.degree_property_map("total").copy("int64_t")
    else:
        vertex_inv2 = perfect_prop_hash([vertex_inv2])[0].copy("int64_t")

    inv_max = max(vertex_inv1.fa.max(), vertex_inv2.fa.max()) + 1

    l1 = label_self_loops(g1, mark_only=True)
    if l1.fa.max() > 0:
        g1 = GraphView(g1, efilt=1 - l1.fa)

    l2 = label_self_loops(g2, mark_only=True)
    if l2.fa.max() > 0:
        g2 = GraphView(g2, efilt=1 - l2.fa)

    iso = libgraph_tool_topology.\
           check_isomorphism(g1._Graph__graph, g2._Graph__graph,
                             _prop("v", g1, vertex_inv1),
                             _prop("v", g2, vertex_inv2),
                             inv_max,
                             _prop("v", g1, imap))
    if isomap:
        return iso, imap
    else:
        return iso


def subgraph_isomorphism(sub, g, max_n=0, vertex_label=None, edge_label=None,
                         induced=False, subgraph=True, generator=False):
    r"""Obtain all subgraph isomorphisms of `sub` in `g` (or at most `max_n` subgraphs, if `max_n > 0`).


    Parameters
    ----------
    sub : :class:`~graph_tool.Graph`
        Subgraph for which to be searched.
    g : :class:`~graph_tool.Graph`
        Graph in which the search is performed.
    max_n : int (optional, default: ``0``)
        Maximum number of matches to find. If `max_n == 0`, all matches are
        found.
    vertex_label : pair of :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        If provided, this should be a pair of :class:`~graph_tool.VertexPropertyMap`
        objects, belonging to ``sub`` and ``g`` (in this order), which specify
        vertex labels which should match, in addition to the topological
        isomorphism.
    edge_label : pair of :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        If provided, this should be a pair of :class:`~graph_tool.EdgePropertyMap`
        objects, belonging to ``sub`` and ``g`` (in this order), which specify
        edge labels which should match, in addition to the topological
        isomorphism.
    induced : bool (optional, default: ``False``)
        If ``True``, only node-induced subgraphs are found.
    subgraph : bool (optional, default: ``True``)
        If ``False``, all non-subgraph isomorphisms between `sub` and `g` are
        found.
    generator : bool (optional, default: ``False``)
        If ``True``, a generator will be returned, instead of a list. This is
        useful if the number of isomorphisms is too large to store in memory. If
        ``generator == True``, the option ``max_n`` is ignored.

    Returns
    -------
    vertex_maps : list (or generator) of :class:`~graph_tool.VertexPropertyMap` objects
        List (or generator) containing vertex property map objects which
        indicate different isomorphism mappings. The property maps vertices in
        `sub` to the corresponding vertex index in `g`.

    Notes
    -----
    The implementation is based on the VF2 algorithm, introduced by Cordella et
    al. [cordella-improved-2001]_ [cordella-subgraph-2004]_. The spatial
    complexity is of order :math:`O(V)`, where :math:`V` is the (maximum) number
    of vertices of the two graphs. Time complexity is :math:`O(V^2)` in the best
    case and :math:`O(V!\times V)` in the worst case [boost-subgraph-iso]_.

    Examples
    --------
    >>> from numpy.random import poisson
    >>> g = gt.complete_graph(30)
    >>> sub = gt.complete_graph(10)
    >>> vm = gt.subgraph_isomorphism(sub, g, max_n=100)
    >>> print(len(vm))
    100
    >>> for i in range(len(vm)):
    ...   g.set_vertex_filter(None)
    ...   g.set_edge_filter(None)
    ...   vmask, emask = gt.mark_subgraph(g, sub, vm[i])
    ...   g.set_vertex_filter(vmask)
    ...   g.set_edge_filter(emask)
    ...   assert gt.isomorphism(g, sub)
    >>> g.set_vertex_filter(None)
    >>> g.set_edge_filter(None)
    >>> ewidth = g.copy_property(emask, value_type="double")
    >>> ewidth.a += 0.5
    >>> ewidth.a *= 2
    >>> gt.graph_draw(g, vertex_fill_color=vmask, edge_color=emask,
    ...               edge_pen_width=ewidth,
    ...               output="subgraph-iso-embed.pdf")
    <...>
    >>> gt.graph_draw(sub, output="subgraph-iso.pdf")
    <...>

    .. testcleanup::

       conv_png("subgraph-iso-embed.pdf")
       conv_png("subgraph-iso.pdf")


    .. image:: subgraph-iso.png
       :width: 30%
    .. image:: subgraph-iso-embed.png
       :width: 30%


    **Left:** Subgraph searched, **Right:** One isomorphic subgraph found in main graph.

    References
    ----------
    .. [cordella-improved-2001] L. P. Cordella, P. Foggia, C. Sansone, and M. Vento,
       "An improved algorithm for matching large graphs.", 3rd IAPR-TC15 Workshop
       on Graph-based Representations in Pattern Recognition, pp. 149-159, Cuen, 2001.
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.101.5342
    .. [cordella-subgraph-2004] L. P. Cordella, P. Foggia, C. Sansone, and M. Vento,
       "A (Sub)Graph Isomorphism Algorithm for Matching Large Graphs.",
       IEEE Trans. Pattern Anal. Mach. Intell., vol. 26, no. 10, pp. 1367-1372, 2004.
       :doi:`10.1109/TPAMI.2004.75`
    .. [boost-subgraph-iso] http://www.boost.org/libs/graph/doc/vf2_sub_graph_iso.html
    .. [subgraph-isormophism-wikipedia] http://en.wikipedia.org/wiki/Subgraph_isomorphism_problem

    """
    if sub.num_vertices() == 0:
        raise ValueError("Cannot search for an empty subgraph.")
    if vertex_label is None:
        vertex_label = (None, None)
    elif vertex_label[0].value_type() != vertex_label[1].value_type():
        raise ValueError("Both vertex label property maps must be of the same type!")
    elif vertex_label[0].value_type() != "int64_t":
        vertex_label = perfect_prop_hash(vertex_label, htype="int64_t")

    if edge_label is None:
        edge_label = (None, None)
    elif edge_label[0].value_type() != edge_label[1].value_type():
        raise ValueError("Both edge label property maps must be of the same type!")
    elif edge_label[0].value_type() != "int64_t":
        edge_label = perfect_prop_hash(edge_label, htype="int64_t")

    vmaps = libgraph_tool_topology.\
            subgraph_isomorphism(sub._Graph__graph, g._Graph__graph,
                                 _prop("v", sub, vertex_label[0]),
                                 _prop("v", g, vertex_label[1]),
                                 _prop("e", sub, edge_label[0]),
                                 _prop("e", g, edge_label[1]),
                                 max_n, induced, not subgraph,
                                 generator)
    if generator:
        return (VertexPropertyMap(vmap, sub) for vmap in vmaps)
    else:
        return [VertexPropertyMap(vmap, sub) for vmap in vmaps]


def mark_subgraph(g, sub, vmap, vmask=None, emask=None):
    r"""
    Mark a given subgraph `sub` on the graph `g`.

    The mapping must be provided by the `vmap` and `emap` parameters,
    which map vertices/edges of `sub` to indices of the corresponding
    vertices/edges in `g`.

    This returns a vertex and an edge property map, with value type 'bool',
    indicating whether or not a vertex/edge in `g` corresponds to the subgraph
    `sub`.
    """
    if vmask is None:
        vmask = g.new_vertex_property("bool")
    if emask is None:
        emask = g.new_edge_property("bool")

    vmask.a = False
    emask.a = False

    for v in sub.vertices():
        w = g.vertex(vmap[v])
        vmask[w] = True
        us = set([g.vertex(vmap[x]) for x in v.out_neighbors()])

        for ew in w.out_edges():
            if ew.target() in us:
                emask[ew] = True

    return vmask, emask

def max_cliques(g):
    """Return an iterator over the maximal cliques of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.

    Returns
    -------
    max_cliques : iterator over :class:`numpy.ndarray` instances
        Iterator over lists of vertices corresponding to the maximal cliques.

    Notes
    -----

    This implements the Bron-Kerbosh algorithm [bron_algorithm_1973]_
    [bron-kerbosh-wiki]_ with pivoting [tomita_worst-case_2006]_
    [cazals_note_2008]_.

    The worst case complexity of this algorithm is :math:`O(3^{V/3})` for a
    graph of :math:`V` vertices, but for sparse graphs it is typically much
    faster.

    Examples
    --------

    >>> g = gt.collection.data["polblogs"]
    >>> sum(1 for c in gt.max_cliques(g))
    49618

    References
    ----------
    .. [bron_algorithm_1973] Coen Bron and Joep Kerbosch, "Algorithm 457:
       finding all cliques of an undirected graph", Commun. ACM 16, 9, 575-577
       (1973), :doi:`10.1145/362342.362367`
    .. [tomita_worst-case_2006] Etsuji Tomita, Akira Tanaka, and Haruhisa
       Takahashi. "The worst-case time complexity for generating all maximal
       cliques and computational experiments." Theoretical Computer Science 363.1
       28-42 (2006), :doi:`10.1016/j.tcs.2006.06.015`
    .. [cazals_note_2008] Frédéric Cazals, and Chinmay Karande, "A note on the
       problem of reporting maximal cliques." Theoretical Computer Science 407.1-3
       564-568 (2008), :doi:`10.1016/j.tcs.2008.05.010`
    .. [bron-kerbosh-wiki] https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm

    """

    if g.is_directed():
        g = GraphView(g, directed=False)

    for c in libgraph_tool_topology.max_cliques(g._Graph__graph):
        yield c

def min_spanning_tree(g, weights=None, root=None, tree_map=None):
    r"""
    Return the minimum spanning tree of a given graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    weights : :class:`~graph_tool.EdgePropertyMap` (optional, default: `None`)
        The edge weights. If provided, the minimum spanning tree will minimize
        the edge weights.
    root : :class:`~graph_tool.Vertex` (optional, default: `None`)
        Root of the minimum spanning tree. If this is provided, Prim's algorithm
        is used. Otherwise, Kruskal's algorithm is used.
    tree_map : :class:`~graph_tool.EdgePropertyMap` (optional, default: `None`)
        If provided, the edge tree map will be written in this property map.

    Returns
    -------
    tree_map : :class:`~graph_tool.EdgePropertyMap`
        Edge property map with mark the tree edges: 1 for tree edge, 0
        otherwise.

    Notes
    -----
    The algorithm runs with :math:`O(E\log E)` complexity, or :math:`O(E\log V)`
    if `root` is specified.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> from numpy.random import random
    >>> g, pos = gt.triangulation(random((400, 2)) * 10, type="delaunay")
    >>> weight = g.new_edge_property("double")
    >>> for e in g.edges():
    ...    weight[e] = linalg.norm(pos[e.target()].a - pos[e.source()].a)
    >>> tree = gt.min_spanning_tree(g, weights=weight)
    >>> gt.graph_draw(g, pos=pos, output="triang_orig.pdf")
    <...>
    >>> u = gt.GraphView(g, efilt=tree)
    >>> gt.graph_draw(u, pos=pos, output="triang_min_span_tree.pdf")
    <...>

    .. testcleanup::

       conv_png("triang_orig.pdf")
       conv_png("triang_min_span_tree.pdf")

    .. image:: triang_orig.png
        :width: 400px
    .. image:: triang_min_span_tree.png
        :width: 400px

    *Left:* Original graph, *Right:* The minimum spanning tree.

    References
    ----------
    .. [kruskal-shortest-1956] J. B. Kruskal.  "On the shortest spanning subtree
       of a graph and the traveling salesman problem",  In Proceedings of the
       American Mathematical Society, volume 7, pages 48-50, 1956.
       :doi:`10.1090/S0002-9939-1956-0078686-7`
    .. [prim-shortest-1957] R. Prim.  "Shortest connection networks and some
       generalizations",  Bell System Technical Journal, 36:1389-1401, 1957.
    .. [boost-mst] http://www.boost.org/libs/graph/doc/graph_theory_review.html#sec:minimum-spanning-tree
    .. [mst-wiki] http://en.wikipedia.org/wiki/Minimum_spanning_tree
    """
    if tree_map is None:
        tree_map = g.new_edge_property("bool")
    if tree_map.value_type() != "bool":
        raise ValueError("edge property 'tree_map' must be of value type bool.")

    u = GraphView(g, directed=False)
    if root is None:
        libgraph_tool_topology.\
               get_kruskal_spanning_tree(u._Graph__graph,
                                         _prop("e", g, weights),
                                         _prop("e", g, tree_map))
    else:
        libgraph_tool_topology.\
               get_prim_spanning_tree(u._Graph__graph, int(root),
                                      _prop("e", g, weights),
                                      _prop("e", g, tree_map))
    return tree_map


def random_spanning_tree(g, weights=None, root=None, tree_map=None):
    r"""Return a random spanning tree of a given graph, which can be directed or
    undirected.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    weights : :class:`~graph_tool.EdgePropertyMap` (optional, default: `None`)
        The edge weights. If provided, the probability of a particular spanning
        tree being selected is the product of its edge weights.
    root : :class:`~graph_tool.Vertex` (optional, default: `None`)
        Root of the spanning tree. If not provided, it will be selected randomly.
    tree_map : :class:`~graph_tool.EdgePropertyMap` (optional, default: `None`)
        If provided, the edge tree map will be written in this property map.

    Returns
    -------
    tree_map : :class:`~graph_tool.EdgePropertyMap`
        Edge property map with mark the tree edges: 1 for tree edge, 0
        otherwise.

    Notes
    -----

    The running time for this algorithm is :math:`O(\tau)`, with :math:`\tau`
    being the mean hitting time of a random walk on the graph. In the worst case,
    we have :math:`\tau \sim O(V^3)`, with :math:`V` being the number of
    vertices in the graph. However, in much more typical cases (e.g. sparse
    random graphs) the running time is simply :math:`O(V)`.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> from numpy.random import random
    >>> g, pos = gt.triangulation(random((400, 2)), type="delaunay")
    >>> weight = g.new_edge_property("double")
    >>> for e in g.edges():
    ...    weight[e] = linalg.norm(pos[e.target()].a - pos[e.source()].a)
    >>> tree = gt.random_spanning_tree(g, weights=weight)
    >>> tree2 = gt.random_spanning_tree(g, weights=weight)
    >>> gt.graph_draw(g, pos=pos, output="rtriang_orig.pdf")
    <...>
    >>> u = gt.GraphView(g, efilt=tree)
    >>> gt.graph_draw(u, pos=pos, output="triang_random_span_tree.pdf")
    <...>
    >>> u2 = gt.GraphView(g, efilt=tree2)
    >>> gt.graph_draw(u2, pos=pos, output="triang_random_span_tree2.pdf")
    <...>

    .. testcleanup::

       conv_png("rtriang_orig.pdf")
       conv_png("triang_random_span_tree.pdf")
       conv_png("triang_random_span_tree2.pdf")

    .. image:: rtriang_orig.png
        :width: 300px
    .. image:: triang_random_span_tree.png
        :width: 300px
    .. image:: triang_random_span_tree2.png
        :width: 300px

    *Left:* Original graph, *Middle:* A random spanning tree, *Right:* Another
    random spanning tree

    References
    ----------

    .. [wilson-generating-1996] David Bruce Wilson, "Generating random spanning
       trees more quickly than the cover time", Proceedings of the twenty-eighth
       annual ACM symposium on Theory of computing, Pages 296-303, ACM New York,
       1996, :doi:`10.1145/237814.237880`
    .. [boost-rst] http://www.boost.org/libs/graph/doc/random_spanning_tree.html
    """
    if tree_map is None:
        tree_map = g.new_edge_property("bool")
    if tree_map.value_type() != "bool":
        raise ValueError("edge property 'tree_map' must be of value type bool.")

    if root is None:
        root = g.vertex(numpy.random.randint(0, g.num_vertices()),
                        use_index=False)

    # we need to restrict ourselves to the in-component of root
    l = label_out_component(GraphView(g, reversed=True), root)
    u = GraphView(g, vfilt=l)
    if u.num_vertices() != g.num_vertices():
        raise ValueError("There must be a path from all vertices to the root vertex: %d" % int(root) )

    libgraph_tool_topology.\
        random_spanning_tree(g._Graph__graph, int(root),
                             _prop("e", g, weights),
                             _prop("e", g, tree_map), _get_rng())
    return tree_map


def dominator_tree(g, root, dom_map=None):
    r"""Return a vertex property map the dominator vertices for each vertex.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    root : :class:`~graph_tool.Vertex`
        The root vertex.
    dom_map : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        If provided, the dominator map will be written in this property map.

    Returns
    -------
    dom_map : :class:`~graph_tool.VertexPropertyMap`
        The dominator map. It contains for each vertex, the index of its
        dominator vertex.

    Notes
    -----
    A vertex u dominates a vertex v, if every path of directed graph from the
    entry to v must go through u.

    The algorithm runs with :math:`O((V+E)\log (V+E))` complexity.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(100, lambda: (2, 2))
    >>> tree = gt.min_spanning_tree(g)
    >>> g.set_edge_filter(tree)
    >>> root = [v for v in g.vertices() if v.in_degree() == 0]
    >>> dom = gt.dominator_tree(g, root[0])
    >>> print(dom.a)
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

    References
    ----------
    .. [dominator-bgl] http://www.boost.org/libs/graph/doc/lengauer_tarjan_dominator.htm

    """
    if dom_map is None:
        dom_map = g.new_vertex_property("int32_t")
    if dom_map.value_type() != "int32_t":
        raise ValueError("vertex property 'dom_map' must be of value type" +
                         " int32_t.")
    if not g.is_directed():
        raise ValueError("dominator tree requires a directed graph.")
    libgraph_tool_topology.\
               dominator_tree(g._Graph__graph, int(root),
                              _prop("v", g, dom_map))
    return dom_map


def topological_sort(g):
    """
    Return the topological sort of the given graph. It is returned as an array
    of vertex indices, in the sort order.

    Notes
    -----
    The topological sort algorithm creates a linear ordering of the vertices
    such that if edge (u,v) appears in the graph, then u comes before v in the
    ordering. The graph must be a directed acyclic graph (DAG).

    The time complexity is :math:`O(V + E)`.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(30, lambda: (3, 3))
    >>> tree = gt.min_spanning_tree(g)
    >>> g.set_edge_filter(tree)
    >>> sort = gt.topological_sort(g)
    >>> print(sort)
    [28 27 29 26 25 24 22 21 20 19 18 16 15 14 13 11  9  8  7  3  2 10  4 17
      5  1  0 12 23  6]

    References
    ----------
    .. [topological-boost] http://www.boost.org/libs/graph/doc/topological_sort.html
    .. [topological-wiki] http://en.wikipedia.org/wiki/Topological_sorting

    """

    topological_order = Vector_int32_t()
    is_DAG = libgraph_tool_topology.\
        topological_sort(g._Graph__graph, topological_order)
    if not is_DAG:
        raise ValueError("Graph is not a directed acylic graph (DAG).");
    return topological_order.a[::-1].copy()


def transitive_closure(g):
    """Return the transitive closure graph of g.

    Notes
    -----
    The transitive closure of a graph G = (V,E) is a graph G* = (V,E*) such that
    E* contains an edge (u,v) if and only if G contains a path (of at least one
    edge) from u to v. The transitive_closure() function transforms the input
    graph g into the transitive closure graph tc.

    The time complexity (worst-case) is :math:`O(VE)`.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(30, lambda: (3, 3))
    >>> tc = gt.transitive_closure(g)

    References
    ----------
    .. [transitive-boost] http://www.boost.org/libs/graph/doc/transitive_closure.html
    .. [transitive-wiki] http://en.wikipedia.org/wiki/Transitive_closure

    """

    if not g.is_directed():
        raise ValueError("graph must be directed for transitive closure.")
    tg = Graph()
    libgraph_tool_topology.transitive_closure(g._Graph__graph,
                                              tg._Graph__graph)
    return tg


def label_components(g, vprop=None, directed=None, attractors=False):
    """
    Label the components to which each vertex in the graph belongs. If the
    graph is directed, it finds the strongly connected components.

    A property map with the component labels is returned, together with an
    histogram of component labels.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    vprop : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property to store the component labels. If none is supplied, one
        is created.
    directed : bool (optional, default: ``None``)
        Treat graph as directed or not, independently of its actual
        directionality.
    attractors : bool (optional, default: ``False``)
        If ``True``, and the graph is directed, an additional array with Boolean
        values is returned, specifying if the strongly connected components are
        attractors or not.

    Returns
    -------
    comp : :class:`~graph_tool.VertexPropertyMap`
        Vertex property map with component labels.
    hist : :class:`numpy.ndarray`
        Histogram of component labels.
    is_attractor : :class:`numpy.ndarray`
        A Boolean array specifying if the strongly connected components are
        attractors or not. This returned only if ``attractors == True``, and the
        graph is directed.

    Notes
    -----
    The components are arbitrarily labeled from 0 to N-1, where N is the total
    number of components.

    The algorithm runs in :math:`O(V + E)` time.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(43)
       gt.seed_rng(43)

    >>> g = gt.random_graph(100, lambda: (poisson(2), poisson(2)))
    >>> comp, hist, is_attractor = gt.label_components(g, attractors=True)
    >>> print(comp.a)
    [12 12  4 12 12 13 16 12 12 12 12 17 12 12 12 12 12 12 12  7 14 12 12 18
     19 12 20 12 12 21 12 11 12 22 12 12 12  1 23 12 10 12 24 12  3 12  9  6
     12 12 12 12 12  0  5 12 12 25  8 12 12  2 26 15 12 12 12 12 27 12 12 12
     12 12 12 12 28 29 12 12 12 12 12 12 12 30 12 31 12 12 12 32 33 12 12 12
     12 12 12 12]
    >>> print(hist)
    [ 1  1  1  1  1  1  1  1  1  1  1  1 67  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1]
    >>> print(is_attractor)
    [ True  True  True False  True  True  True  True  True  True  True  True
     False  True  True False False False False False False False False False
     False False  True False False False False False False False]
    """

    if vprop is None:
        vprop = g.new_vertex_property("int32_t")

    _check_prop_writable(vprop, name="vprop")
    _check_prop_scalar(vprop, name="vprop")

    if directed is not None:
        g = GraphView(g, directed=directed)

    hist = libgraph_tool_topology.\
               label_components(g._Graph__graph, _prop("v", g, vprop))

    if attractors and g.is_directed() and directed != False:
        is_attractor = numpy.ones(len(hist), dtype="bool")
        libgraph_tool_topology.\
               label_attractors(g._Graph__graph, _prop("v", g, vprop),
                                is_attractor)
        return vprop, hist, is_attractor
    else:
        return vprop, hist


def label_largest_component(g, directed=None):
    """
    Label the largest component in the graph. If the graph is directed, then the
    largest strongly connected component is labelled.

    A property map with a boolean label is returned.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    directed : bool (optional, default: ``None``)
        Treat graph as directed or not, independently of its actual
        directionality.

    Returns
    -------
    comp : :class:`~graph_tool.VertexPropertyMap`
         Boolean vertex property map which labels the largest component.

    Notes
    -----
    The algorithm runs in :math:`O(V + E)` time.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(100, lambda: poisson(1), directed=False)
    >>> l = gt.label_largest_component(g)
    >>> print(l.a)
    [0 1 0 0 1 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 1 0 0 1 0 1 0 1 0 0 0 0 1 1 0
     1 0 1 0 1 0 0 0 0 0 0 0 1 0 1 1 0 0 0 1 1 1 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 1 1 0 0 0]
    >>> u = gt.GraphView(g, vfilt=l)   # extract the largest component as a graph
    >>> print(u.num_vertices())
    28
    """

    label = g.new_vertex_property("bool")
    c, h = label_components(g, directed=directed)
    label.fa = c.fa == h.argmax()
    return label

def extract_largest_component(g, directed=None, prune=False):
    """Extract the largest (strong) component in the graph as a
    :class:`~graph_tool.GraphView` (or :class:`~graph_tool.Graph` if
    ``prune==True``).

    If the graph is directed, then the largest strongly connected component is
    returned.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    directed : bool (optional, default: ``None``)
        Treat graph as directed or not, independently of its actual
        directionality.
    prune : bool (optional, default: ``False``)
        If ``True``, a pruned copy of the component is returned. Otherwise a
        :class:`~graph_tool.GraphView` is returned.

    Returns
    -------
    comp : :class:`~graph_tool.GraphView` or :class:`~graph_tool.Graph`
         Largest (strong) component.

    Notes
    -----
    The algorithm runs in :math:`O(V + E)` time.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(100, lambda: poisson(1), directed=False)
    >>> u = gt.extract_largest_component(g)
    >>> print(u)
    <GraphView object, undirected, with 28 vertices and 27 edges, edges filtered by (<EdgePropertyMap object with value type 'bool', for Graph 0x..., at 0x...>, False), vertices filtered by (<VertexPropertyMap object with value type 'bool', for Graph 0x..., at 0x...>, False), at 0x...>
    """

    c = label_largest_component(g, directed=directed)
    u = GraphView(g, vfilt=c)
    if prune:
        u = Graph(u, prune=True)
    return u


def label_out_component(g, root, label=None):
    """
    Label the out-component (or simply the component for undirected graphs) of a
    root vertex.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    root : :class:`~graph_tool.Vertex`
        The root vertex.
    label : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        If provided, this must be an initialized Boolean vertex property map
        where the out-component will be labeled.

    Returns
    -------
    label : :class:`~graph_tool.VertexPropertyMap`
         Boolean vertex property map which labels the out-component.

    Notes
    -----
    The algorithm runs in :math:`O(V + E)` time.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(100, lambda: poisson(2.2), directed=False)
    >>> l = gt.label_out_component(g, g.vertex(2))
    >>> print(l.a)
    [1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 0 0 1 0 1 0
     1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1
     1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0]

    The in-component can be obtained by reversing the graph.

    >>> l = gt.label_out_component(gt.GraphView(g, reversed=True, directed=True),
    ...                            g.vertex(1))
    >>> print(l.a)
    [0 1 1 0 0 1 0 0 0 1 1 0 1 0 0 0 0 1 1 1 0 0 0 1 1 0 0 0 1 1 0 0 0 0 0 1 0
     1 1 0 0 0 1 1 1 1 0 1 1 1 1 1 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 1 0 1 0 0 0
     0 1 0 1 0 1 0 0 1 0 1 1 1 1 0 1 0 0 0 1 0 0 1 0 0 0]

    """

    if label is None:
        label = g.new_vertex_property("bool")
    elif label.value_type() != "bool":
        raise ValueError("value type of `label` must be `bool`, not %s" %
                         label.value_type())
    libgraph_tool_topology.\
             label_out_component(g._Graph__graph, int(root),
                                 _prop("v", g, label))
    return label


def label_biconnected_components(g, eprop=None, vprop=None):
    """
    Label the edges of biconnected components, and the vertices which are
    articulation points.

    An edge property map with the component labels is returned, together a
    boolean vertex map marking the articulation points, and an histogram of
    component labels.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.

    eprop : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge property to label the biconnected components.

    vprop : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property to mark the articulation points. If none is supplied,
        one is created.


    Returns
    -------
    bicomp : :class:`~graph_tool.EdgePropertyMap`
        Edge property map with the biconnected component labels.
    articulation : :class:`~graph_tool.VertexPropertyMap`
        Boolean vertex property map which has value 1 for each vertex which is
        an articulation point, and zero otherwise.
    nc : int
        Number of biconnected components.

    Notes
    -----

    A connected graph is biconnected if the removal of any single vertex (and
    all edges incident on that vertex) can not disconnect the graph. More
    generally, the biconnected components of a graph are the maximal subsets of
    vertices such that the removal of a vertex from a particular component will
    not disconnect the component. Unlike connected components, vertices may
    belong to multiple biconnected components: those vertices that belong to
    more than one biconnected component are called "articulation points" or,
    equivalently, "cut vertices". Articulation points are vertices whose removal
    would increase the number of connected components in the graph. Thus, a
    graph without articulation points is biconnected. Vertices can be present in
    multiple biconnected components, but each edge can only be contained in a
    single biconnected component.

    The algorithm runs in :math:`O(V + E)` time.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(100, lambda: poisson(2), directed=False)
    >>> comp, art, hist = gt.label_biconnected_components(g)
    >>> print(comp.a)
    [31 31 31  0 31 31 31 31 31 15 27 31 28 31 31 14 31 31 31 31  3 31 31 31
     32 31 31 19 16 29 20 31 31 31 31 31 31 31 31 31 31 31 31 10 24 31  4  8
     22 31 31 31 31  2 31 31 31 31 31 18 31 31 31 31 31 11 21 23  1 31 31 30
      7 31 31 31 25 31 31 31 31 31 31  6 26 17 31 31 31 13 31 31 31 12  9 31
     31 31 31  5 31 31 31]
    >>> print(art.a)
    [1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 1 0 0 0 0 0 1 1 0 0 0 1 0 1 0 0 0 0 0 0
     1 0 1 0 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 1 1 0
     1 0 0 1 0 1 0 0 1 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0]
    >>> print(hist)
    [ 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1 71  1]
    """

    if vprop is None:
        vprop = g.new_vertex_property("bool")
    if eprop is None:
        eprop = g.new_edge_property("int32_t")

    _check_prop_writable(vprop, name="vprop")
    _check_prop_scalar(vprop, name="vprop")
    _check_prop_writable(eprop, name="eprop")
    _check_prop_scalar(eprop, name="eprop")

    g = GraphView(g, directed=False)
    hist = libgraph_tool_topology.\
             label_biconnected_components(g._Graph__graph, _prop("e", g, eprop),
                                          _prop("v", g, vprop))
    return eprop, vprop, hist

def vertex_percolation(g, vertices, second=False):
    """Compute the size of the largest or second-largest component as vertices
    are (virtually) removed from the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    vertices : :class:`numpy.ndarray` or iterable of ints
        List of vertices in reversed order of removal.
    second : bool (optional, default: ``False``)
        If ``True``, the size of the second-largest component will be computed.

    Returns
    -------
    size : :class:`numpy.ndarray`
        Size of the largest (or second-largest) component prior to removal of
        each vertex.
    comp : :class:`~graph_tool.VertexPropertyMap`
        Vertex property map with component labels.

    Notes
    -----

    The algorithm runs in :math:`O(V + E)` time.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(10000, lambda: geometric(1./4) + 1, directed=False)
    >>> vertices = sorted([v for v in g.vertices()], key=lambda v: v.out_degree())
    >>> sizes, comp = gt.vertex_percolation(g, vertices)
    >>> numpy.random.shuffle(vertices)
    >>> sizes2, comp = gt.vertex_percolation(g, vertices)
    >>> figure()
    <...>
    >>> plot(sizes, label="Targeted")
    [...]
    >>> plot(sizes2, label="Random")
    [...]
    >>> xlabel("Vertices remaining")
    Text(...)
    >>> ylabel("Size of largest component")
    Text(...)
    >>> legend(loc="upper left")
    <...>
    >>> savefig("vertex-percolation.svg")

    .. figure:: vertex-percolation.*
        :align: center

        Targeted and random vertex percolation of a random graph with an
        exponential degree distribution.

    References
    ----------
    .. [newman-ziff] M. E. J. Newman, R. M. Ziff, "A fast Monte Carlo algorithm
       for site or bond percolation", Phys. Rev. E 64, 016706 (2001)
       :doi:`10.1103/PhysRevE.64.016706`, :arxiv:`cond-mat/0101295`

    """
    vertices = numpy.asarray(vertices, dtype="uint64")

    tree = g.vertex_index.copy("int64_t")
    size = g.new_vertex_property("int64_t", 1)
    visited = g.new_vertex_property("bool", False)
    max_size = numpy.zeros(len(vertices), dtype="uint64")

    u = GraphView(g, directed=False)

    libgraph_tool_topology.\
        percolate_vertex(u._Graph__graph,
                         _prop("v", u, tree),
                         _prop("v", u, size),
                         _prop("v", u, visited),
                         vertices, max_size, second)

    return max_size, tree

def edge_percolation(g, edges, second=False):
    """Compute the size of the largest or second-largest component as edges are
    (virtually) removed from the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    edges : :class:`numpy.ndarray` or iterable of pairs of ints
        List of edges in reversed order of removal. If the type is
        :class:`numpy.ndarray`, it should have a shape ``(E, 2)``, where ``E``
        is the number of edges, such that ``edges[i,0]`` and ``edges[i,1]`` are
        the both endpoints of edge ``i``.
    second : bool (optional, default: ``False``)
        If ``True``, the size of the second-largest component will be computed.

    Returns
    -------
    size : :class:`numpy.ndarray`
        Size of the largest (or second-largest) component prior to removal of
        each edge.
    comp : :class:`~graph_tool.VertexPropertyMap`
        Vertex property map with component labels.

    Notes
    -----

    The algorithm runs in :math:`O(E)` time.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(10000, lambda: geometric(1./4) + 1, directed=False)
    >>> edges = sorted([(e.source(), e.target()) for e in g.edges()],
    ...                key=lambda e: e[0].out_degree() * e[1].out_degree())
    >>> sizes, comp = gt.edge_percolation(g, edges)
    >>> numpy.random.shuffle(edges)
    >>> sizes2, comp = gt.edge_percolation(g, edges)
    >>> figure()
    <...>
    >>> plot(sizes, label="Targeted")
    [...]
    >>> plot(sizes2, label="Random")
    [...]
    >>> xlabel("Edges remaining")
    Text(...)
    >>> ylabel("Size of largest component")
    Text(...)
    >>> legend(loc="lower right")
    <...>
    >>> savefig("edge-percolation.svg")

    .. figure:: edge-percolation.*
        :align: center

        Targeted and random edge percolation of a random graph with an
        exponential degree distribution.

    References
    ----------
    .. [newman-ziff] M. E. J. Newman, R. M. Ziff, "A fast Monte Carlo algorithm
       for site or bond percolation", Phys. Rev. E 64, 016706 (2001)
       :doi:`10.1103/PhysRevE.64.016706`, :arxiv:`cond-mat/0101295`

    """
    edges = numpy.asarray(edges, dtype="uint64")

    tree = g.vertex_index.copy("int64_t")
    size = g.new_vertex_property("int64_t", 1)
    max_size = numpy.zeros(len(edges), dtype="uint64")

    u = GraphView(g, directed=False)

    libgraph_tool_topology.\
        percolate_edge(u._Graph__graph,
                       _prop("v", u, tree),
                       _prop("v", u, size),
                       edges, max_size, second)
    return max_size, tree

def kcore_decomposition(g, vprop=None):
    """Perform a k-core decomposition of the given graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    vprop : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property to store the decomposition. If ``None`` is supplied,
        one is created.

    Returns
    -------
    kval : :class:`~graph_tool.VertexPropertyMap`
        Vertex property map with the k-core decomposition, i.e. a given vertex v
        belongs to the ``kval[v]``-core.

    Notes
    -----

    The k-core is a maximal set of vertices such that its induced subgraph only
    contains vertices with degree larger than or equal to k.

    For directed graphs, the degree is assumed to be the total (in + out)
    degree.

    The algorithm accepts graphs with parallel edges and self loops, in which
    case these edges contribute to the degree in the usual fashion.

    This algorithm is described in [batagelj-algorithm]_ and runs in :math:`O(V + E)`
    time.

    Examples
    --------

    >>> g = gt.collection.data["netscience"]
    >>> g = gt.GraphView(g, vfilt=gt.label_largest_component(g))
    >>> kcore = gt.kcore_decomposition(g)
    >>> gt.graph_draw(g, pos=g.vp["pos"], vertex_fill_color=kcore, vertex_text=kcore, output="netsci-kcore.svg")
    <...>

    .. figure:: netsci-kcore.svg
        :align: center
        :width: 80%

        K-core decomposition of a network of network scientists.

    References
    ----------
    .. [k-core] http://en.wikipedia.org/wiki/Degeneracy_%28graph_theory%29
    .. [batagelj-algorithm]  Vladimir Batagelj, Matjaž Zaveršnik, "Fast
       algorithms for determining (generalized) core groups in social
       networks", Advances in Data Analysis and Classification
       Volume 5, Issue 2, pp 129-145 (2011), :DOI:`10.1007/s11634-010-0079-y`,
       :arxiv:`cs/0310049`

    """

    if vprop is None:
        vprop = g.new_vertex_property("int32_t")

    _check_prop_writable(vprop, name="vprop")
    _check_prop_scalar(vprop, name="vprop")

    libgraph_tool_topology.\
               kcore_decomposition(g._Graph__graph, _prop("v", g, vprop))

    return vprop


def shortest_distance(g, source=None, target=None, weights=None,
                      negative_weights=False, max_dist=None, directed=None,
                      dense=False, dist_map=None, pred_map=False,
                      return_reached=False, dag=False):
    r"""Calculate the distance from a source to a target vertex, or to of all
    vertices from a given source, or the all pairs shortest paths, if the source
    is not specified.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    source : :class:`~graph_tool.Vertex` (optional, default: ``None``)
        Source vertex of the search. If unspecified, the all pairs shortest
        distances are computed.
    target : :class:`~graph_tool.Vertex` or iterable of such objects (optional, default: ``None``)
        Target vertex (or vertices) of the search. If unspecified, the distance
        to all vertices from the source will be computed.
    weights : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        The edge weights. If provided, the shortest path will correspond to the
        minimal sum of weights.
    negative_weights : ``bool`` (optional, default: ``False``)
        If `True`, this will trigger the use of the Bellman-Ford algorithm.
        Ignored if ``source`` is ``None``.
    max_dist : scalar value (optional, default: ``None``)
        If specified, this limits the maximum distance of the vertices
        searched. This parameter has no effect if source is ``None``, or if
        `negative_weights=True`.
    directed : ``bool`` (optional, default:``None``)
        Treat graph as directed or not, independently of its actual
        directionality.
    dense : ``bool`` (optional, default: ``False``)
        If ``True``, and source is ``None``, the Floyd-Warshall algorithm is used,
        otherwise the Johnson algorithm is used. If source is not ``None``, this option
        has no effect.
    dist_map : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property to store the distances. If none is supplied, one
        is created.

        .. warning::

            If this parameter is supplied, the user is responsible for
            initializing it to infinity. This can be done as:

            >>> dist_map = g.new_vp("double", numpy.inf)                     # doctest: +SKIP

            or

            >>> dist_map = g.new_vp("int32_t", numpy.iinfo(numpy.int32).max) # doctest: +SKIP

            depending on the distance type.

    pred_map : ``bool`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``False``)
        If ``True``, a vertex property map with the predecessors is returned.
        If a :class:`~graph_tool.VertexPropertyMap` is passed, it must be of value
        ``int64_t`` and it will be used to store the predecessors.  Ignored if
        ``source`` is ``None``.

        .. warning::

            If a property map is supplied, the user is responsible for
            initializing to the identity map. This can be done as:

            >>> pred_map = g.vertex_index.copy()                             # doctest: +SKIP

    return_reached : ``bool`` (optional, default: ``False``)
        If ``True``, return an array of visited vertices.
    dag : ``bool`` (optional, default:``False``)
        If ``True``, assume that the graph is a Directed Acyclic Graph (DAG),
        which will be faster if ``weights`` are given, in which case they are
        also allowed to contain negative values (irrespective of the parameter
        ``negative_weights``). Ignored if ``source`` is ``None``.

    Returns
    -------
    dist_map : :class:`~graph_tool.VertexPropertyMap` or :class:`numpy.ndarray`
        Vertex property map with the distances from source. If ``source`` is
        ``None``, it will have a vector value type, with the distances to every
        vertex. If ``target`` is an iterable, instead of
        :class:`~graph_tool.VertexPropertyMap`, this will be of type
        :class:`numpy.ndarray`, and contain only the distances to those specific
        targets.
    pred_map : :class:`~graph_tool.VertexPropertyMap` (optional, if ``pred_map == True``)
        Vertex property map with the predecessors in the search tree.
    visited : :class:`numpy.ndarray` (optional, if ``return_reached == True``)
        Array containing vertices visited during the search.

    Notes
    -----

    If a source is given, the distances are calculated with a breadth-first
    search (BFS) or Dijkstra's algorithm [dijkstra]_, if weights are given. If
    ``negative_weights == True``, the Bellman-Ford algorithm is used
    [bellman-ford]_, which accepts negative weights, as long as there are no
    negative loops. If source is not given, the distances are calculated with
    Johnson's algorithm [johnson-apsp]_. If dense=True, the Floyd-Warshall
    algorithm [floyd-warshall-apsp]_ is used instead.

    If there is no path between two vertices, the computed distance will
    correspond to the maximum value allowed by the value type of ``dist_map``,
    or ``inf`` in case of floating point types.

    If source is specified, the algorithm runs in :math:`O(V + E)` time, or
    :math:`O(V \log V)` if weights are given (if ``dag == True`` this improves
    to :math:`O(V+E)`). If ``negative_weights == True``, the complexity is
    :math:`O(VE)`. If source is not specified, the algorithm runs in parallel
    with complexity :math:`O(V (V + E))`, if weights are given it runs
    in :math:`O(VE\log V)` time, or :math:`O(V^3)` if dense == True.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> from numpy.random import poisson
    >>> g = gt.random_graph(100, lambda: (poisson(3), poisson(3)))
    >>> dist = gt.shortest_distance(g, source=g.vertex(0))
    >>> print(dist.a)
    [         0          4          5          4          6          4
              5          5          2          3          4          4
              6          5          7          5 2147483647          4
     2147483647          3          3          4          3          2
              4          5          5          4          4          4
              4          6          3          4          3          5
              1          5          4          3          4          3
              6 2147483647          4          5          3          5
              6          4          5          4          4          3
              3          4          6          4          2          3
              4          5          4          5          4          2
              5          5          3          3 2147483647          4
     2147483647          7          6          3          4          4
              5          4          1 2147483647          3          2
              4          4          5          3          4          5
              5          5          5          5          5          5
              4          3          2          3]
    >>>
    >>> dist = gt.shortest_distance(g)
    >>> print(dist[g.vertex(0)].a)
    [         0          4          5          4          6          4
              5          5          2          3          4          4
              6          5          7          5 2147483647          4
     2147483647          3          3          4          3          2
              4          5          5          4          4          4
              4          6          3          4          3          5
              1          5          4          3          4          3
              6 2147483647          4          5          3          5
              6          4          5          4          4          3
              3          4          6          4          2          3
              4          5          4          5          4          2
              5          5          3          3 2147483647          4
     2147483647          7          6          3          4          4
              5          4          1 2147483647          3          2
              4          4          5          3          4          5
              5          5          5          5          5          5
              4          3          2          3]
    >>> dist = gt.shortest_distance(g, source=g.vertex(0), target=g.vertex(2))
    >>> print(dist)
    5
    >>> dist = gt.shortest_distance(g, source=g.vertex(0), target=[g.vertex(2), g.vertex(6)])
    >>> print(dist)
    [5 5]

    References
    ----------
    .. [bfs] Edward Moore, "The shortest path through a maze", International
       Symposium on the Theory of Switching (1959), Harvard University Press.
    .. [bfs-boost] http://www.boost.org/libs/graph/doc/breadth_first_search.html
    .. [dijkstra] E. Dijkstra, "A note on two problems in connexion with
       graphs." Numerische Mathematik, 1:269-271, 1959.
    .. [dijkstra-boost] http://www.boost.org/libs/graph/doc/dijkstra_shortest_paths.html
    .. [johnson-apsp] http://www.boost.org/libs/graph/doc/johnson_all_pairs_shortest.html
    .. [floyd-warshall-apsp] http://www.boost.org/libs/graph/doc/floyd_warshall_shortest.html
    .. [bellman-ford] http://www.boost.org/libs/graph/doc/bellman_ford_shortest.html

    """

    tgtlist = False
    if isinstance(target, collections.abc.Iterable):
        tgtlist = True
        target = numpy.asarray(target, dtype="int64")
    elif target is None:
        target = numpy.array([], dtype="int64")
    else:
        target = numpy.asarray([int(g.vertex(int(target)))], dtype="int64")

    if weights is None:
        dist_type = 'int32_t'
    else:
        dist_type = weights.value_type()

    if dist_map is None:
        if source is not None:
            dist_map = g.new_vertex_property(dist_type)
        else:
            dist_map = g.new_vertex_property("vector<%s>" % dist_type)

    _check_prop_writable(dist_map, name="dist_map")
    if source is not None:
        _check_prop_scalar(dist_map, name="dist_map")
    else:
        _check_prop_vector(dist_map, name="dist_map")

    if max_dist is None:
        max_dist = 0

    if directed is not None:
        u = GraphView(g, directed=directed)
    else:
        u = g

    if source is not None:
        if numpy.issubdtype(dist_map.a.dtype, numpy.integer):
            dist_map.fa = numpy.iinfo(dist_map.a.dtype).max
        else:
            dist_map.fa = numpy.inf
        if isinstance(pred_map, PropertyMap):
            pmap = pred_map
            if pmap.value_type() != "int64_t":
                raise ValueError("supplied pred_map must be of value type 'int64_t'")
        else:
            pmap = u.copy_property(u.vertex_index, value_type="int64_t")
        reached = libcore.Vector_size_t()
        libgraph_tool_topology.get_dists(u._Graph__graph,
                                         int(u.vertex(int(source))),
                                         target,
                                         _prop("v", u, dist_map),
                                         _prop("e", u, weights),
                                         _prop("v", u, pmap),
                                         float(max_dist),
                                         negative_weights, reached, dag)
    else:
        libgraph_tool_topology.get_all_dists(u._Graph__graph,
                                             _prop("v", u, dist_map),
                                             _prop("e", u, weights), dense)

    if source is not None and len(target) > 0:
        if len(target) == 1 and not tgtlist:
            dist_map = dist_map.a[target[0]]
        else:
            dist_map = numpy.array(dist_map.a[target])

    if source is not None:
        if pred_map:
            ret = (dist_map, pmap)
        else:
            ret = (dist_map,)
        if return_reached:
            return ret + (numpy.asarray(reached.a.copy()),)
        else:
            if len(ret) == 1:
                return ret[0]
            return ret
    else:
        return dist_map

def shortest_path(g, source, target, weights=None, negative_weights=False,
                  pred_map=None, dag=False):
    r"""Return the shortest path from ``source`` to ``target``.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    source : :class:`~graph_tool.Vertex`
        Source vertex of the search.
    target : :class:`~graph_tool.Vertex`
        Target vertex of the search.
    weights : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        The edge weights.
    negative_weights : ``bool`` (optional, default: ``False``)
        If ``True``, this will trigger the use of the Bellman-Ford algorithm.
    pred_map :  :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map with the predecessors in the search tree. If this is
        provided, the shortest paths are not computed, and are obtained directly
        from this map.
    dag : ``bool`` (optional, default:``False``)
        If ``True``, assume that the graph is a Directed Acyclic Graph (DAG),
        which will be faster if ``weights`` are given, in which case they are
        also allowed to contain negative values (irrespective of the parameter
        ``negative_weights``).

    Returns
    -------
    vertex_list : list of :class:`~graph_tool.Vertex`
        List of vertices from `source` to `target` in the shortest path.
    edge_list : list of :class:`~graph_tool.Edge`
        List of edges from `source` to `target` in the shortest path.

    Notes
    -----

    The paths are computed with a breadth-first search (BFS) or Dijkstra's
    algorithm [dijkstra]_, if weights are given. If ``negative_weights ==
    True``, the Bellman-Ford algorithm is used [bellman-ford]_, which accepts
    negative weights, as long as there are no negative loops.

    The algorithm runs in :math:`O(V + E)` time, or :math:`O(V \log V)` if
    weights are given (if ``dag == True`` this improves to :math:`O(V+E)`).

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(43)
       gt.seed_rng(43)

    >>> from numpy.random import poisson
    >>> g = gt.random_graph(300, lambda: (poisson(4), poisson(4)))
    >>> vlist, elist = gt.shortest_path(g, g.vertex(10), g.vertex(11))
    >>> print([str(v) for v in vlist])
    ['10', '114', '96', '97', '11']
    >>> print([str(e) for e in elist])
    ['(10, 114)', '(114, 96)', '(96, 97)', '(97, 11)']

    References
    ----------
    .. [bfs] Edward Moore, "The shortest path through a maze", International
       Symposium on the Theory of Switching (1959), Harvard University
       Press
    .. [bfs-boost] http://www.boost.org/libs/graph/doc/breadth_first_search.html
    .. [dijkstra] E. Dijkstra, "A note on two problems in connexion with
       graphs." Numerische Mathematik, 1:269-271, 1959.
    .. [dijkstra-boost] http://www.boost.org/libs/graph/doc/dijkstra_shortest_paths.html
    .. [bellman-ford] http://www.boost.org/libs/graph/doc/bellman_ford_shortest.html

    """

    if pred_map is None:
        pred_map = shortest_distance(g, source, target, weights=weights,
                                     negative_weights=negative_weights,
                                     pred_map=True, dag=dag)[1]

    if pred_map[target] == int(target):  # no path to target
        return [], []

    vlist = [target]
    elist = []

    if weights is not None:
        max_w = weights.a.max() + 1
    else:
        max_w = None

    source = g.vertex(source)
    target = g.vertex(target)
    v = target
    while v != source:
        p = g.vertex(pred_map[v])
        min_w = max_w
        pe = None
        s = None
        for e in v.in_edges() if g.is_directed() else v.out_edges():
            s = e.source() if g.is_directed() else e.target()
            if s == p:
                if weights is not None:
                    if weights[e] < min_w:
                        min_w = weights[e]
                        pe = e
                else:
                    pe = e
                    break
        elist.insert(0, pe)
        vlist.insert(0, p)
        v = p
    return vlist, elist

def all_predecessors(g, dist_map, pred_map, weights=None, epsilon=1e-8):
    """Return a property map with all possible predecessors in the search tree
    determined by ``dist_map`` and ``pred_map``.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    dist_map : :class:`~graph_tool.VertexPropertyMap`
        Vertex property map with the distances from ``source`` to all other
        vertices.
    pred_map : :class:`~graph_tool.VertexPropertyMap`
        Vertex property map with the predecessors in the search tree.
    weights : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        The edge weights.
    epsilon : ``float`` (optional, default: ``1e-8``)
        Maximum relative difference between distances to be considered "equal",
        in case floating-point weights are used.

    Returns
    -------
    all_preds_map : :class:`~graph_tool.VertexPropertyMap`
        Vector-valued vertex property map with all possible predecessors in the
        search tree.

    """

    preds = g.new_vertex_property("vector<int64_t>")
    libgraph_tool_topology.get_all_preds(g._Graph__graph,
                                         _prop("v", g, dist_map),
                                         _prop("v", g, pred_map),
                                         _prop("e", g, weights),
                                         _prop("v", g, preds),
                                         epsilon)
    return preds

def all_shortest_paths(g, source, target, dist_map=None, pred_map=None,
                       all_preds_map=None, epsilon=1e-8, edges=False, **kwargs):
    """Return an iterator over all shortest paths from `source` to `target`.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    source : :class:`~graph_tool.Vertex`
        Source vertex of the search.
    target : :class:`~graph_tool.Vertex`
        Target vertex of the search.
    dist_map : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map with the distances from ``source`` to all other
        vertices.
    pred_map : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map with the predecessors in the search tree. If this is
        provided, the shortest paths are not computed, and are obtained directly
        from this map.
    all_preds_map : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vector-valued vertex property map with all possible predecessors in the
        search tree. If this is provided, the shortest paths are obtained
        directly from this map.
    epsilon : ``float`` (optional, default: ``1e-8``)
        Maximum relative difference between distances to be considered "equal",
        in case floating-point weights are used.
    edges : ``bool`` (optional, default: ``False``)
        If ``True``, the returned iterator is over edge descriptors.
    **kwargs : Keyword parameter list
        The remaining parameters will be passed to
        :func:`~graph_tool.topology.shortest_distance`.

    Returns
    -------
    path_iterator : iterator over a sequence of integers
        Iterator over sequences of vertices from `source` to `target` in the
        shortest path. If ``edges == True``, the iterator is over sequences of
        edge descriptors (:class:`~graph_tool.Edge`).

    Notes
    -----

    The paths are computed in the same manner as in
    :func:`~graph_tool.topology.shortest_distance`, which is used internally.

    If both ``dist_map`` and ``pred_map`` are provided, the
    :func:`~graph_tool.topology.shortest_distance` is not called.

    Examples
    --------

    >>> g = gt.collection.data["pgp-strong-2009"]
    >>> for path in gt.all_shortest_paths(g, 92, 45):
    ...     print(path)
    [  92  107 2176 7027   26   21   45]
    [  92  107 2176 7033   26   21   45]
    [  92   82   94 5877 5879   34   45]
    [  92   89   94 5877 5879   34   45]

    """

    if dist_map is None or pred_map is None:
        dist_map, pred_map = shortest_distance(g, source, **dict(kwargs,
                                                                 pred_map=True))

    if pred_map[target] == int(target):
        return

    if all_preds_map is None:
        all_preds_map = all_predecessors(g, dist_map, pred_map,
                                         kwargs.get("weights", None), epsilon)

    path_iterator = \
        libgraph_tool_topology.get_all_shortest_paths(g._Graph__graph,
                                                      int(source),
                                                      int(target),
                                                      _prop("v", g, all_preds_map),
                                                      _prop("e", g, kwargs.get("weights", None)),
                                                      edges)
    for p in path_iterator:
        yield p

def random_shortest_path(g, source, target, dist_map=None, pred_map=None,
                         all_preds_map=None, epsilon=1e-8, nsamples=1,
                         iterator=False, **kwargs):
    """Return a random shortest path from `source` to `target`, uniformly sampled
    from all paths of equal length.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    source : :class:`~graph_tool.Vertex`
        Source vertex of the search.
    target : :class:`~graph_tool.Vertex`
        Target vertex of the search.
    dist_map : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map with the distances from ``source`` to all other
        vertices.
    pred_map : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map with the predecessors in the search tree. If this is
        provided, the shortest paths are not computed, and are obtained directly
        from this map.
    all_preds_map : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vector-valued vertex property map with all possible predecessors in the
        search tree. If this is provided, the shortest paths are obtained
        directly from this map.
    epsilon : ``float`` (optional, default: ``1e-8``)
        Maximum relative difference between distances to be considered "equal",
        in case floating-point weights are used.
    nsamples : ``int`` (optional, default: ``1``)
        Number of paths to sample.
    iterator : ``bool`` (optional, default: ``False``)
        If ``True``, an iterator is returned.
    **kwargs : Keyword parameter list
        The remaining parameters will be passed to
        :func:`~graph_tool.topology.shortest_path`.

    Returns
    -------
    path : :class:`numpy.ndarray`, or list of :class:`numpy.ndarray` or iterator over :class:`numpy.ndarray`
        Sequence of vertices from `source` to `target` in the shortest path. If
        ``nsamples > 1`` a list (or iterator) over paths is returned.

    Notes
    -----

    The paths are computed in the same manner as in
    :func:`~graph_tool.topology.shortest_path`, which is used internally.

    If both ``dist_map`` and ``pred_map`` are provided, the
    :func:`~graph_tool.topology.shortest_path` is not called.

    Examples
    --------

    .. testcode::
       :hide:

       gt.seed_rng(42)

    >>> g = gt.collection.data["pgp-strong-2009"]
    >>> path = gt.random_shortest_path(g, 92, 45)
    >>> print(path)
    [  92   89   94 5877 5879   34   45]
    """

    if dist_map is None or pred_map is None:
        dist_map, pred_map = shortest_distance(g, source,
                                               **dict(kwargs, pred_map=True))

    if all_preds_map is None:
        all_preds_map = all_predecessors(g, dist_map, pred_map,
                                         kwargs.get("weights", None), epsilon)

    all_succ_map = g.new_vp("vector<int64_t>")
    count_map = g.new_vp("int64_t")
    visited_map = g.new_vp("bool")

    libgraph_tool_topology.get_weighted_succs(int(target),
                                              all_preds_map._get_any(),
                                              all_succ_map._get_any(),
                                              count_map._get_any(),
                                              visited_map._get_any())

    def sample_gen():
        if pred_map[target] == int(target):
            return

        for i in range(nsamples):
            path = Vector_size_t()

            libgraph_tool_topology.get_random_shortest_path(int(source),
                                                            int(target),
                                                            all_succ_map._get_any(),
                                                            count_map._get_any(),
                                                            path, _get_rng())
            yield numpy.array(path.a)

    if iterator:
        return sample_gen()
    else:
        samples = list(sample_gen())
        if len(samples) == 1:
            return samples[0]
        return samples

def count_shortest_paths(g, source, target, dist_map=None, pred_map=None,
                         all_preds_map=None, epsilon=1e-8, **kwargs):
    """Return the number of shortest paths from `source` to `target`.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    source : :class:`~graph_tool.Vertex`
        Source vertex of the search.
    target : :class:`~graph_tool.Vertex`
        Target vertex of the search.
    dist_map : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map with the distances from ``source`` to all other
        vertices.
    pred_map : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map with the predecessors in the search tree. If this is
        provided, the shortest paths are not computed, and are obtained directly
        from this map.
    all_preds_map : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vector-valued vertex property map with all possible predecessors in the
        search tree. If this is provided, the shortest paths are obtained
        directly from this map.
    epsilon : ``float`` (optional, default: ``1e-8``)
        Maximum relative difference between distances to be considered "equal",
        in case floating-point weights are used.
    **kwargs : Keyword parameter list
        The remaining parameters will be passed to
        :func:`~graph_tool.topology.shortest_path`.

    Returns
    -------
    n_paths : ``int``
        Number of shortest paths from `source` to `target`.

    Notes
    -----

    The paths are computed in the same manner as in
    :func:`~graph_tool.topology.shortest_path`, which is used internally.

    If both ``dist_map`` and ``pred_map`` are provided, the
    :func:`~graph_tool.topology.shortest_path` is not called.

    Examples
    --------

    >>> g = gt.collection.data["pgp-strong-2009"]
    >>> n_paths = gt.count_shortest_paths(g, 92, 45)
    >>> print(n_paths)
    4

    """

    if dist_map is None or pred_map is None:
        dist_map, pred_map = shortest_distance(g, source,
                                               **dict(kwargs, pred_map=True))
    if pred_map[target] == int(target):
        return 0

    if all_preds_map is None:
        all_preds_map = all_predecessors(g, dist_map, pred_map,
                                         kwargs.get("weights", None), epsilon)

    all_succ_map = g.new_vp("vector<int64_t>")
    count_map = g.new_vp("int64_t")
    visited_map = g.new_vp("bool")

    libgraph_tool_topology.get_weighted_succs(int(target),
                                              all_preds_map._get_any(),
                                              all_succ_map._get_any(),
                                              count_map._get_any(),
                                              visited_map._get_any())

    return count_map[source]

def all_paths(g, source, target, cutoff=None, edges=False):
    """Return an iterator over all paths from `source` to `target`.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    source : :class:`~graph_tool.Vertex`
        Source vertex of the search.
    target : :class:`~graph_tool.Vertex`
        Target vertex of the search.
    cutoff : ``int`` (optional, default: ``None``)
        Maximum path length.
    edges : ``bool`` (optional, default: ``False``)
        If ``True``, the returned iterator is over edge descriptors.

    Returns
    -------
    path_iterator : iterator over a sequence of integers (or :class:`~graph_tool.Edge`)
        Iterator over sequences of vertices from `source` to `target` in the
        path. If ``edges == True``, the iterator is over sequences of edge
        descriptors (:class:`~graph_tool.Edge`).

    Notes
    -----

    The algorithm uses a depth-first search to find all the paths.

    The total number of paths between any two vertices can be quite large,
    possibly scaling as :math:`O(V!)`.

    Examples
    --------

    >>> g = gt.collection.data["football"]
    >>> for path in gt.all_paths(g, 13, 2, cutoff=2):
    ...     print(path)
    [13  2]
    [13 15  2]
    [13 60  2]
    [13 64  2]
    [ 13 100   2]
    [ 13 106   2]

    """

    if cutoff is None:
        cutoff = g.num_edges() + 1
    visited = g.new_vp("bool", False)
    path_iterator = libgraph_tool_topology.get_all_paths(g._Graph__graph,
                                                         int(source),
                                                         int(target),
                                                         cutoff,
                                                         _prop("v", g, visited),
                                                         edges)
    for p in path_iterator:
        yield p

def all_circuits(g, unique=False):
    """Return an iterator over all the cycles in a directed graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        A directed graph to be used. (Undirected graphs are also accepted, in
        which case each undirected edge is assumed to be equivalent to two
        directed edges in both directions.)
    unique : ``bool`` (optional, default: ``None``)
        If ``True``, parallel edges and self-loops will be ignored.

    Returns
    -------
    cycle_iterator : iterator over a sequence of integers
        Iterator over sequences of vertices that form a circuit.

    Notes
    -----
    This algorithm [johnson-finding-1975]_[hawick-enumerating-2008]_ runs in
    worst time :math:`O[(V + E)(C + 1)]`, where :math:`C` is the number of
    circuits.

    Examples
    --------
    .. testcode::
       :hide:

       gt.seed_rng(42)

    >>> g = gt.random_graph(10, lambda: (1, 1))
    >>> for c in gt.all_circuits(g):
    ...     print(c)
    [0 6]
    [1 5 4 9 7 2 8 3]

    References
    ----------
    .. [johnson-finding-1975] D.B. Johnson, "Finding all the elementary circuits of a directed graph",
       SIAM Journal on Computing, 1975. :doi:`10.1137/0204007`
    .. [hawick-enumerating-2008] K.A. Hawick and H.A. James, "Enumerating
       Circuits and Loops in Graphs with Self-Arcs and Multiple-Arcs.",
       In Proceedings of FCS. 2008, 14-20,
       http://cssg.massey.ac.nz/cstn/013/cstn-013.html
    .. [hawick-bgl] http://www.boost.org/doc/libs/graph/doc/hawick_circuits.html

    """

    circuits_iterator = libgraph_tool_topology.get_all_circuits(g._Graph__graph,
                                                                unique)
    return circuits_iterator


def pseudo_diameter(g, source=None, weights=None):
    r"""
    Compute the pseudo-diameter of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    source : :class:`~graph_tool.Vertex` (optional, default: `None`)
        Source vertex of the search. If not supplied, the first vertex
        in the graph will be chosen.
    weights : :class:`~graph_tool.EdgePropertyMap` (optional, default: `None`)
        The edge weights.

    Returns
    -------
    pseudo_diameter : int
        The pseudo-diameter of the graph.
    end_points : pair of :class:`~graph_tool.Vertex`
        The two vertices which correspond to the pseudo-diameter found.

    Notes
    -----

    The pseudo-diameter is an approximate graph diameter. It is obtained by
    starting from a vertex `source`, and finds a vertex `target` that is
    farthest away from `source`. This process is repeated by treating
    `target` as the new starting vertex, and ends when the graph distance no
    longer increases. A vertex from the last level set that has the smallest
    degree is chosen as the final starting vertex u, and a traversal is done
    to see if the graph distance can be increased. This graph distance is
    taken to be the pseudo-diameter.

    The paths are computed with a breadth-first search (BFS) or Dijkstra's
    algorithm [dijkstra]_, if weights are given.

    The algorithm runs in :math:`O(V + E)` time, or :math:`O(V \log V)` if
    weights are given.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> from numpy.random import poisson
    >>> g = gt.random_graph(300, lambda: (poisson(3), poisson(3)))
    >>> dist, ends = gt.pseudo_diameter(g)
    >>> print(dist)
    8.0
    >>> print(int(ends[0]), int(ends[1]))
    0 157

    References
    ----------
    .. [pseudo-diameter] http://en.wikipedia.org/wiki/Distance_%28graph_theory%29
    .. [dijkstra] E. Dijkstra, "A note on two problems in connexion with
       graphs." Numerische Mathematik, 1:269-271, 1959.
    """

    if source is None:
        source = g.vertex(0, use_index=False)
    dist, target = 0, source
    while True:
        new_source = target
        new_target, new_dist = libgraph_tool_topology.get_diam(g._Graph__graph,
                                                               int(new_source),
                                                               _prop("e", g, weights))
        if new_dist > dist:
            target = new_target
            source = new_source
            dist = new_dist
        else:
            break
    return dist, (g.vertex(source), g.vertex(target))


def is_bipartite(g, partition=False, find_odd_cycle=False):
    """Test if the graph is bipartite.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    partition : bool (optional, default: ``False``)
        If ``True``, return the two partitions in case the graph is bipartite.
    find_odd_cycle : bool (optional, default: ``False``)
        If ``True``, return an odd cycle if the graph is not bipartite.

    Returns
    -------
    is_bipartite : ``bool``
        Whether or not the graph is bipartite.
    partition : :class:`~graph_tool.VertexPropertyMap` (only if ``partition=True``)
        A vertex property map with the graph partitioning (or ``None``) if the
        graph is not bipartite.
    odd_cycle : list of vertices (only if ``find_odd_cycle=True``)
        A list of vertices corresponding to an odd cycle, or ``None`` if none is
        found.

    Notes
    -----

    An undirected graph is bipartite if one can partition its set of vertices
    into two sets, such that all edges go from one set to the other.

    This algorithm runs in :math:`O(V + E)` time.

    Examples
    --------
    >>> g = gt.lattice([10, 10])
    >>> is_bi, part = gt.is_bipartite(g, partition=True)
    >>> print(is_bi)
    True
    >>> gt.graph_draw(g, vertex_fill_color=part, output="bipartite.pdf")
    <...>

    .. testcleanup::

       conv_png("bipartite.pdf")

    .. figure:: bipartite.png
        :align: center
        :width: 40%

        Bipartition of a 2D lattice.

    References
    ----------
    .. [boost-bipartite] http://www.boost.org/libs/graph/doc/is_bipartite.html

    """

    if partition:
        part = g.new_vertex_property("bool")
    else:
        part = None
    g = GraphView(g, directed=False, skip_properties=True)
    cycle = []
    is_bi = libgraph_tool_topology.is_bipartite(g._Graph__graph,
                                                _prop("v", g, part),
                                                find_odd_cycle, cycle)
    if not is_bi and part is not None:
        part.a = 0

    if len(cycle) == 0:
        cycle = None
    else:
        cycle.append(cycle[0])

    if find_odd_cycle:
        if partition:
            return is_bi, part, cycle
        else:
            return is_bi, cycle
    else:
        if partition:
            return is_bi, part
        else:
            return is_bi



def is_planar(g, embedding=False, kuratowski=False):
    """
    Test if the graph is planar.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    embedding : bool (optional, default: False)
        If true, return a mapping from vertices to the clockwise order of
        out-edges in the planar embedding.
    kuratowski : bool (optional, default: False)
        If true, the minimal set of edges that form the obstructing Kuratowski
        subgraph will be returned as a property map, if the graph is not planar.

    Returns
    -------
    is_planar : bool
        Whether or not the graph is planar.
    embedding : :class:`~graph_tool.VertexPropertyMap` (only if `embedding=True`)
        A vertex property map with the out-edges indices in clockwise order in
        the planar embedding,
    kuratowski : :class:`~graph_tool.EdgePropertyMap` (only if `kuratowski=True`)
        An edge property map with the minimal set of edges that form the
        obstructing Kuratowski subgraph (if the value of kuratowski[e] is 1,
        the edge belongs to the set)

    Notes
    -----

    A graph is planar if it can be drawn in two-dimensional space without any of
    its edges crossing. This algorithm performs the Boyer-Myrvold planarity
    testing [boyer-myrvold]_. See [boost-planarity]_ for more details.

    This algorithm runs in :math:`O(V)` time.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> from numpy.random import random
    >>> g = gt.triangulation(random((100,2)))[0]
    >>> p, embed_order = gt.is_planar(g, embedding=True)
    >>> print(p)
    True
    >>> print(list(embed_order[g.vertex(0)]))
    [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1]
    >>> g = gt.random_graph(100, lambda: 4, directed=False)
    >>> p, kur = gt.is_planar(g, kuratowski=True)
    >>> print(p)
    False
    >>> g.set_edge_filter(kur, True)
    >>> gt.graph_draw(g, output="kuratowski.pdf")
    <...>

    .. testcleanup::

       conv_png("kuratowski.pdf")

    .. figure:: kuratowski.png
       :align: center
       :width: 40%

       Obstructing Kuratowski subgraph of a random graph.

    References
    ----------
    .. [boyer-myrvold] John M. Boyer and Wendy J. Myrvold, "On the Cutting Edge:
       Simplified O(n) Planarity by Edge Addition" Journal of Graph Algorithms
       and Applications, 8(2): 241-273, 2004. http://www.emis.ams.org/journals/JGAA/accepted/2004/BoyerMyrvold2004.8.3.pdf
    .. [boost-planarity] http://www.boost.org/libs/graph/doc/boyer_myrvold.html
    """

    u = GraphView(g, directed=False)

    if embedding:
        embed = g.new_vertex_property("vector<int>")
    else:
        embed = None

    if kuratowski:
        kur = g.new_edge_property("bool")
    else:
        kur = None

    is_planar = libgraph_tool_topology.is_planar(u._Graph__graph,
                                                 _prop("v", g, embed),
                                                 _prop("e", g, kur))

    ret = [is_planar]
    if embed is not None:
        ret.append(embed)
    if kur is not None:
        ret.append(kur)
    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


def make_maximal_planar(g):
    """
    Add edges to the graph to make it maximally planar.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used. It must be a biconnected planar graph with at least 3
        vertices.

    Notes
    -----

    A graph is maximal planar if no additional edges can be added to it without
    creating a non-planar graph. By Euler's formula, a maximal planar graph with
    V > 2 vertices always has 3V - 6 edges and 2V - 4 faces.

    The input graph to make_maximal_planar() must be a biconnected planar graph
    with at least 3 vertices.

    This algorithm runs in :math:`O(V + E)` time.

    Examples
    --------
    >>> g = gt.lattice([10, 10])
    >>> gt.make_maximal_planar(g)
    >>> gt.is_planar(g)
    True
    >>> print(g.num_vertices(), g.num_edges())
    100 294
    >>> pos = gt.planar_layout(g)
    >>> gt.graph_draw(g, pos, output="maximal_planar.pdf")
    <...>

    .. testcleanup::

       conv_png("maximal_planar.pdf")

    .. figure:: maximal_planar.png
        :align: center
        :width: 40%

        A maximally planar graph.

    References
    ----------
    .. [boost-planarity] http://www.boost.org/libs/graph/doc/make_maximal_planar.html
    """

    g = GraphView(g, directed=False)
    libgraph_tool_topology.maximal_planar(g._Graph__graph)


def is_DAG(g):
    """
    Return `True` if the graph is a directed acyclic graph (DAG).

    Notes
    -----
    The time complexity is :math:`O(V + E)`.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(30, lambda: (3, 3))
    >>> print(gt.is_DAG(g))
    False
    >>> tree = gt.min_spanning_tree(g)
    >>> g.set_edge_filter(tree)
    >>> print(gt.is_DAG(g))
    True

    References
    ----------
    .. [DAG-wiki] http://en.wikipedia.org/wiki/Directed_acyclic_graph

    """

    topological_order = Vector_int32_t()
    is_DAG = libgraph_tool_topology.\
        topological_sort(g._Graph__graph, topological_order)
    return is_DAG


def max_cardinality_matching(g, weight=None, bipartite=False,
                             init_match="extra_greedy", heuristic=False,
                             minimize=False, edges=False, brute_force=False):
    r"""Find a maximum cardinality matching in the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    weight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        If provided, the matching will maximize the sum of edge weights.
    bipartite : ``bool`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``False``)
        If ``True``, the graph will be assumed to be bipartite. If a
        :class:`~graph_tool.VertexPropertyMap` is passed, it should correspond
        to an existing bi-partition.
    init_match : ``string`` (optional, default: ``"extra_greedy"``)
        Initial matching strategy. Can be one of: `"empty"`, `"greedy"`,
        `"extra_greedy"`. Ignored if ``weight`` is given, or
        ``heuristic == True``.
    minimize : ``bool`` (optional, default: ``True``)
        If ``True``, the matching will minimize the weights, otherwise they will
        be maximized. This option has no effect if ``heuristic == False``.
    heuristic : ``bool`` (optional, default: ``False``)
        If ``True``, a random heuristic will be used, which runs in linear time.
    edges : ``bool`` (optional, default: ``False``)
        If ``True``, an edge property map will be returned, instead of a vertex
        property map.
    brute_force : ``bool`` (optional, default: ``False``)
        If ``True``, and ``weight`` is not ``None`` and ``heuristic`` is
        ``False``, a slower, brute-force algorithm is used.

    Returns
    -------
    match : :class:`~graph_tool.VertexPropertyMap`
        Vertex property map where the matching is specified. If ``edges ==
        True`` a boolean-valued edge property map is returned instead.

    Notes
    -----
    A *matching* is a subset of the edges of a graph such that no two edges
    share a common vertex. A *maximum cardinality matching* has maximum size
    over all matchings in the graph.

    If the parameter ``weight`` is provided, a matching with maximum cardinality
    *and* maximum weight is returned.

    If ``heuristic == True`` the algorithm does not necessarily return the
    maximum matching, instead the focus is to run on linear time
    [matching-heuristic]_.

    This algorithm runs in time :math:`O(EV\times\alpha(E,V))`, where
    :math:`\alpha(m,n)` is a slow growing function that is at most 4 for any
    feasible input.

    If weights are given, the algorithm runs in time :math:`O(V^3)`.

    If `heuristic == True`, the algorithm runs in time :math:`O(V + E)`.

    If `brute_force == True`, the algorithm runs in time :math:`O(exp(E))`.

    For a more detailed description, see [boost-max-matching]_ and
    [boost-max-weighted-matching]_.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(43)
       gt.seed_rng(43)

    >>> g = gt.GraphView(gt.price_network(300), directed=False)
    >>> w = gt.max_cardinality_matching(g, edges=True)
    >>> w = w.copy("double")
    >>> w.a = 2 * w.a + 2
    >>> gt.graph_draw(g, edge_color=w, edge_pen_width=w, vertex_fill_color="grey",
    ...               output="max_card_match.pdf")
    <...>

    .. testcleanup::

       conv_png("max_card_match.pdf")

    .. figure:: max_card_match.png
       :align: center
       :width: 80%

       Edges belonging to the matching are in yellow.

    References
    ----------
    .. [boost-max-matching] http://www.boost.org/libs/graph/doc/maximum_matching.html
    .. [boost-max-weighted-matching] http://www.boost.org/libs/graph/doc/maximum_weighted_matching.html
    .. [matching-heuristic] B. Hendrickson and R. Leland. "A Multilevel Algorithm
       for Partitioning Graphs." In S. Karin, editor, Proc. Supercomputing ’95,
       San Diego. ACM Press, New York, 1995, :doi:`10.1145/224170.224228`

    """
    match = g.new_vp("int64_t")
    if weight is not None:
        _check_prop_scalar(weight, "weight")

    u = GraphView(g, directed=False)
    if not heuristic:
        if bipartite is False:
            if weight is None:
                libgraph_tool_topology.\
                    get_max_matching(u._Graph__graph, init_match,
                                     _prop("v", u, match))
            else:
                libgraph_tool_topology.\
                    get_max_weighted_matching(u._Graph__graph,
                                              _prop("e", u, weight),
                                              _prop("v", u, match), brute_force)
        else:
            if isinstance(bipartite, VertexPropertyMap):
                partition = bipartite
            else:
                is_bip, partition = is_bipartite(u, partition=True)
                if not is_bip:
                    raise ValueError("Supplied graph is not bipartite")
            libgraph_tool_topology.\
                get_max_bip_weighted_matching(u._Graph__graph,
                                              _prop("v", u, partition),
                                              _prop("e", u, weight),
                                              _prop("v", u, match))
    else:
         libgraph_tool_topology.\
                random_matching(u._Graph__graph, _prop("e", u, weight),
                                 _prop("v", u, match), minimize, _get_rng())

    if edges:
        ematch = g.new_ep("bool")
        libgraph_tool_topology.match_edges(u._Graph__graph,
                                           _prop("v", u, match),
                                           _prop("e", u, ematch))
        return ematch
    return match


def max_independent_vertex_set(g, high_deg=False, mivs=None):
    r"""Find a maximal independent vertex set in the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    high_deg : bool (optional, default: `False`)
        If `True`, vertices with high degree will be included first in the set,
        otherwise they will be included last.
    mivs : :class:`~graph_tool.VertexPropertyMap` (optional, default: `None`)
        Vertex property map where the vertex set will be specified.

    Returns
    -------
    mivs : :class:`~graph_tool.VertexPropertyMap`
        Boolean vertex property map where the set is specified.

    Notes
    -----
    A maximal independent vertex set is an independent set such that adding any
    other vertex to the set forces the set to contain an edge between two
    vertices of the set.

    This implements the algorithm described in [mivs-luby]_, which runs in time
    :math:`O(V + E)`.

    Examples
    --------
    .. testcode::
       :hide:

       import numpy.random
       numpy.random.seed(43)
       gt.seed_rng(43)

    >>> g = gt.GraphView(gt.price_network(300), directed=False)
    >>> res = gt.max_independent_vertex_set(g)
    >>> gt.graph_draw(g, vertex_fill_color=res, output="mivs.pdf")
    <...>

    .. testcleanup::

       conv_png("mivs.pdf")

    .. figure:: mivs.png
       :align: center
       :width: 80%

       Vertices belonging to the set are in yellow.

    References
    ----------
    .. [mivs-wikipedia] http://en.wikipedia.org/wiki/Independent_set_%28graph_theory%29
    .. [mivs-luby] Luby, M., "A simple parallel algorithm for the maximal independent set problem",
       Proc. 17th Symposium on Theory of Computing, Association for Computing Machinery, pp. 1-10, (1985)
       :doi:`10.1145/22145.22146`.

    """
    if mivs is None:
        mivs = g.new_vertex_property("bool")
    _check_prop_scalar(mivs, "mivs")
    _check_prop_writable(mivs, "mivs")

    u = GraphView(g, directed=False)
    libgraph_tool_topology.\
        maximal_vertex_set(u._Graph__graph, _prop("v", u, mivs), high_deg,
                           _get_rng())
    mivs = g.own_property(mivs)
    return mivs


@_parallel
def edge_reciprocity(g, weight=None):
    r"""Calculate the edge reciprocity of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used
        edges.
    weight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge weights.

    Returns
    -------
    reciprocity : float
        The reciprocity value.

    Notes
    -----

    The edge [reciprocity]_ is defined as :math:`E^\leftrightarrow/E`, where
    :math:`E^\leftrightarrow` and :math:`E` are the number of bidirectional and
    all edges in the graph, respectively.

    If weights are provided, the number of edges is replaced by the sum of edge
    weights.

    The algorithm runs with complexity :math:`O(E + V)`.

    @parallel@

    Examples
    --------

    >>> g = gt.Graph()
    >>> g.add_vertex(2)
    <...>
    >>> g.add_edge(g.vertex(0), g.vertex(1))
    <...>
    >>> gt.edge_reciprocity(g)
    0.0
    >>> g.add_edge(g.vertex(1), g.vertex(0))
    <...>
    >>> gt.edge_reciprocity(g)
    1.0
    >>> g = gt.collection.data["pgp-strong-2009"]
    >>> gt.edge_reciprocity(g)
    0.692196963163...

    References
    ----------
    .. [reciprocity] S. Wasserman and K. Faust, "Social Network Analysis".
       (Cambridge University Press, Cambridge, 1994)
    .. [lopez-reciprocity-2007] Gorka Zamora-López, Vinko Zlatić, Changsong
       Zhou, Hrvoje Štefančić, and Jürgen Kurths "Reciprocity of networks with
       degree correlations and arbitrary degree sequences", Phys. Rev. E 77,
       016106 (2008) :doi:`10.1103/PhysRevE.77.016106`, :arxiv:`0706.3372`

    """

    if weight is None:
        ew = libcore.any()
    else:
        _check_prop_scalar(weight, name="eweight")
        ew = _prop("e", g, weight)
    r = libgraph_tool_topology.reciprocity(g._Graph__graph, ew)
    return r


def tsp_tour(g, src, weight=None):
    r"""Return a traveling salesman tour of the graph, which is guaranteed to be
    twice as long as the optimal tour in the worst case.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used. The graph must be undirected.
    src : :class:`~graph_tool.Vertex`
        The source (and target) of the tour.
    weight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge weights.

    Returns
    -------
    tour : :class:`numpy.ndarray`
        List of vertex indices corresponding to the tour.

    Notes
    -----
    The algorithm runs with :math:`O(E\log V)` complexity.

    Examples
    --------
    >>> g = gt.lattice([10, 10])
    >>> tour = gt.tsp_tour(g, g.vertex(0))
    >>> print(tour)
    [ 0  1  2 11 12 21 22 31 32 41 42 51 52 61 62 71 72 81 82 83 73 63 53 43
     33 23 13  3  4  5  6  7  8  9 19 29 39 49 59 69 79 89 14 24 34 44 54 64
     74 84 91 92 93 94 95 85 75 65 55 45 35 25 15 16 17 18 27 28 37 38 47 48
     57 58 67 68 77 78 87 88 97 98 99 26 36 46 56 66 76 86 96 10 20 30 40 50
     60 70 80 90  0]

    References
    ----------
    .. [tsp-bgl] http://www.boost.org/libs/graph/doc/metric_tsp_approx.html
    .. [tsp] http://en.wikipedia.org/wiki/Travelling_salesman_problem

    """

    if g.is_directed():
        raise ValueError("The graph must be undirected.")
    tour = libgraph_tool_topology.\
        get_tsp(g._Graph__graph, int(src), _prop("e", g, weight))
    return tour.a.copy()


def sequential_vertex_coloring(g, order=None, color=None):
    """Returns a vertex coloring of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    order : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Order with which the vertices will be colored.
    color : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Integer-valued vertex property map to store the colors.

    Returns
    -------
    color : :class:`~graph_tool.VertexPropertyMap`
        Integer-valued vertex property map with the vertex colors.

    Notes
    -----
    The time complexity is :math:`O(V(d+k))`, where :math:`V` is the number of
    vertices, :math:`d` is the maximum degree of the vertices in the graph, and
    :math:`k` is the number of colors used.

    Examples
    --------
    >>> g = gt.lattice([10, 10])
    >>> colors = gt.sequential_vertex_coloring(g)
    >>> print(colors.a)
    [0 1 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1
     0 1 0 0 1 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 0 1 1 0 1 0
     1 0 1 0 1 0 0 1 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 1 0]

    References
    ----------
    .. [sgc-bgl] http://www.boost.org/libs/graph/doc/sequential_vertex_coloring.html
    .. [graph-coloring] http://en.wikipedia.org/wiki/Graph_coloring

    """

    if order is None:
        order = g.vertex_index
    if color is None:
        color = g.new_vertex_property("int")

    libgraph_tool_topology.\
        sequential_coloring(g._Graph__graph,
                            _prop("v", g, order),
                            _prop("v", g, color))
    return color


from .. flow import libgraph_tool_flow
