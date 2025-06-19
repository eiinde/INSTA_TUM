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
``graph_tool.draw``
-------------------

This module contains functions for graph drawing and layout.

Layout algorithms
=================

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   sfdp_layout
   fruchterman_reingold_layout
   arf_layout
   radial_tree_layout
   planar_layout
   random_layout

Graph drawing
=============

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   graph_draw
   draw_hierarchy
   graphviz_draw
   prop_to_size
   get_hierarchy_control_points


Low-level graph drawing
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   cairo_draw
   interactive_window

.. autosummary::
   :nosignatures:
   :toctree: autosummary
   :template: class-no-inherit.rst

   GraphWidget
   GraphWindow

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   GraphArtist

"""

from .. import Graph, GraphView, _check_prop_vector, _check_prop_scalar, \
    group_vector_property, ungroup_vector_property, infect_vertex_property, \
    _prop, _get_rng, VertexPropertyMap
from .. topology import max_cardinality_matching, max_independent_vertex_set, \
    label_components, shortest_distance, make_maximal_planar, is_planar
from .. generation import predecessor_tree, condensation_graph
from .. inference import minimize_blockmodel_dl, BlockState, ModularityState
from .. inference.util import nested_contiguous_map
import numpy.random
from numpy import sqrt
from collections.abc import Iterable

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_layout")


__all__ = ["graph_draw", "graphviz_draw", "fruchterman_reingold_layout",
           "arf_layout", "sfdp_layout", "planar_layout", "random_layout",
           "radial_tree_layout", "cairo_draw", "prop_to_size",
           "get_hierarchy_control_points", "default_cm", "default_cm_bin",
           "default_clrs"]


def random_layout(g, shape=None, pos=None, dim=2):
    r"""Performs a random layout of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    shape : tuple or list (optional, default: ``None``)
        Rectangular shape of the bounding area. The size of this parameter must
        match `dim`, and each element can be either a pair specifying a range,
        or a single value specifying a range starting from zero. If None is
        passed, a square of linear size :math:`\sqrt{N}` is used.
    pos : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vector vertex property maps where the coordinates should be stored.
    dim : int (optional, default: ``2``)
        Number of coordinates per vertex.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        A vector-valued vertex property map with the coordinates of the
        vertices.

    Notes
    -----
    This algorithm has complexity :math:`O(V)`.

    Examples
    --------
    .. testcode::
       :hide:

       np.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(100, lambda: (3, 3))
    >>> shape = [[50, 100], [1, 2], 4]
    >>> pos = gt.random_layout(g, shape=shape, dim=3)
    >>> pos[g.vertex(0)].a
    array([68.72700594,  1.03142919,  2.56812658])

    """

    if pos is None:
        pos = g.new_vertex_property("vector<double>")
    _check_prop_vector(pos, name="pos")

    pos = ungroup_vector_property(pos, list(range(0, dim)))

    if shape is None:
        shape = [sqrt(g.num_vertices())] * dim

    for i in range(dim):
        if hasattr(shape[i], "__len__"):
            if len(shape[i]) != 2:
                raise ValueError("The elements of 'shape' must have size 2.")
            r = [min(shape[i]), max(shape[i])]
        else:
            r = [min(shape[i], 0), max(shape[i], 0)]
        d = r[1] - r[0]

        # deal with filtering
        p = pos[i].fa
        pos[i].fa = numpy.random.random(len(p)) * d + r[0]

    pos = group_vector_property(pos)
    return pos


def planar_layout(g, pos=None):
    r"""Performs a canonical layout of a planar graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Planar graph to be used.
    pos : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vector vertex property maps where the coordinates should be stored.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        A vector-valued vertex property map with the coordinates of the
        vertices.

    Notes
    -----
    This algorithm has complexity :math:`O(V + E)`.

    Examples
    --------
    >>> g = gt.lattice([10, 10])
    >>> pos = gt.planar_layout(g)
    >>> gt.graph_draw(g, pos=pos, output="lattice-planar.pdf")
    <...>

    .. testcleanup::

       conv_png("lattice-planar.pdf")


    .. figure:: lattice-planar.png
        :align: center
        :width: 60%

        Straight-line drawing of planar graph (a 2D square lattice).

    References
    ----------
    .. [straight-line-boost] http://www.boost.org/doc/libs/release/libs/graph/doc/straight_line_drawing.html
    .. [chrobak-linear-1995] M. Chrobak, T. Payne, "A Linear-time Algorithm for
       Drawing a Planar Graph on the Grid", Information Processing Letters 54:
       241-246, (1995), :doi:`10.1016/0020-0190(95)00020-D`
    """

    if g.num_vertices() < 3:
        raise ValueError("Graph must have at least 3 vertices.")
    if not is_planar(g):
        raise ValueError("Graph is not planar.")
    u = Graph(GraphView(g, directed=False, skip_properties=True))
    make_maximal_planar(u)
    embed = is_planar(u, embedding=True)[1]
    if pos is None:
        pos = u.new_vp("vector<double>")
    else:
        pos = u.own_property(pos)
    libgraph_tool_layout.planar_layout(u._Graph__graph,
                                       _prop("v", u, embed),
                                       _prop("v", u, pos))
    pos = g.own_property(pos)
    return pos


def fruchterman_reingold_layout(g, weight=None, a=None, r=1., scale=None,
                                circular=False, grid=True, t_range=None,
                                n_iter=100, pos=None):
    r"""Calculate the Fruchterman-Reingold spring-block layout of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    weight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        An edge property map with the respective weights.
    a : float (optional, default: :math:`V`)
        Attracting force between adjacent vertices.
    r : float (optional, default: 1.0)
        Repulsive force between vertices.
    scale : float (optional, default: :math:`\sqrt{V}`)
        Total scale of the layout (either square side or radius).
    circular : bool (optional, default: ``False``)
        If ``True``, the layout will have a circular shape. Otherwise the shape
        will be a square.
    grid : bool (optional, default: ``True``)
        If ``True``, the repulsive forces will only act on vertices which are on
        the same site on a grid. Otherwise they will act on all vertex pairs.
    t_range : tuple of floats (optional, default: ``(scale / 10, scale / 1000)``)
        Temperature range used in annealing. The temperature limits the
        displacement at each iteration.
    n_iter : int (optional, default: ``100``)
        Total number of iterations.
    pos : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vector vertex property maps where the coordinates should be stored. If
        provided, this will also be used as the initial position of the
        vertices.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        A vector-valued vertex property map with the coordinates of the
        vertices.

    Notes
    -----
    This algorithm is defined in [fruchterman-reingold]_, and has
    complexity :math:`O(\text{n-iter}\times V^2)` if `grid=False` or
    :math:`O(\text{n-iter}\times (V + E))` otherwise.

    Examples
    --------
    .. testcode::
       :hide:

       np.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.price_network(300)
    >>> pos = gt.fruchterman_reingold_layout(g, n_iter=1000)
    >>> gt.graph_draw(g, pos=pos, output="graph-draw-fr.pdf")
    <...>

    .. testcleanup::

       conv_png("graph-draw-fr.pdf")

    .. figure:: graph-draw-fr.png
       :align: center
       :width: 60%

       Fruchterman-Reingold layout of a Price network.

    References
    ----------
    .. [fruchterman-reingold] Fruchterman, Thomas M. J.; Reingold, Edward M.
       "Graph Drawing by Force-Directed Placement". Software - Practice & Experience
       (Wiley) 21 (11): 1129-1164. (1991) :doi:`10.1002/spe.4380211102`
    """

    if pos is None:
        pos = random_layout(g, dim=2)
    _check_prop_vector(pos, name="pos", floating=True)

    if a is None:
        a = float(g.num_vertices())

    if scale is None:
        scale = sqrt(g.num_vertices())

    if t_range is None:
        t_range = (scale / 10, scale / 1000)

    ug = GraphView(g, directed=False)
    libgraph_tool_layout.fruchterman_reingold_layout(ug._Graph__graph,
                                                     _prop("v", g, pos),
                                                     _prop("e", g, weight),
                                                     a, r, not circular, scale,
                                                     grid, t_range[0],
                                                     t_range[1], n_iter)
    return pos


def arf_layout(g, weight=None, d=0.5, a=10, dt=0.001, epsilon=1e-6,
               max_iter=1000, pos=None, dim=2):
    r"""Calculate the ARF spring-block layout of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    weight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        An edge property map with the respective weights.
    d : float (optional, default: ``0.5``)
        Opposing force between vertices.
    a : float (optional, default: ``10``)
        Attracting force between adjacent vertices.
    dt : float (optional, default: ``0.001``)
        Iteration step size.
    epsilon : float (optional, default: ``1e-6``)
        Convergence criterion.
    max_iter : int (optional, default: ``1000``)
        Maximum number of iterations. If this value is ``0``, it runs until
        convergence.
    pos : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vector vertex property maps where the coordinates should be stored.
    dim : int (optional, default: ``2``)
        Number of coordinates per vertex.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        A vector-valued vertex property map with the coordinates of the
        vertices.

    Notes
    -----
    This algorithm is defined in [geipel-self-organization-2007]_, and has
    complexity :math:`O(V^2)`.

    Examples
    --------
    .. testcode::
       :hide:

       np.random.seed(43)
       gt.seed_rng(45)

    >>> g = gt.price_network(300)
    >>> pos = gt.arf_layout(g)
    >>> gt.graph_draw(g, pos=pos, output="graph-draw-arf.pdf")
    <...>

    .. testcleanup::

       conv_png("graph-draw-arf.pdf")

    .. figure:: graph-draw-arf.png
       :align: center
       :width: 60%

       ARF layout of a Price network.

    References
    ----------
    .. [geipel-self-organization-2007] Markus M. Geipel, "Self-Organization
       applied to Dynamic Network Layout", International Journal of Modern
       Physics C vol. 18, no. 10 (2007), pp. 1537-1549,
       :doi:`10.1142/S0129183107011558`, :arxiv:`0704.1748v5`
    .. _arf: http://www.sg.ethz.ch/research/graphlayout
    """

    if pos is None:
        pos = random_layout(g, dim=dim, shape=[[0, 1]] * dim)
    _check_prop_vector(pos, name="pos", floating=True)

    ug = GraphView(g, directed=False)
    libgraph_tool_layout.arf_layout(ug._Graph__graph, _prop("v", g, pos),
                                    _prop("e", g, weight), d, a, dt, max_iter,
                                    epsilon, dim)
    return pos


def _coarse_graph(g, vweight, eweight, kind="matching", groups=None):
    mivs = None
    if groups is None:
        if kind == "mivs":
            mivs = max_independent_vertex_set(g, high_deg=True)
            u = GraphView(g, vfilt=mivs, directed=False)
            c = label_components(u)[0]
            c.fa += 1
            u = GraphView(g, directed=False)
            infect_vertex_property(u, c,
                                   list(range(1, c.fa.max() + 1)))
            c = g.own_property(c)
        elif kind == "ec":
            m = max_cardinality_matching(GraphView(g, directed=False),
                                         heuristic=True, weight=eweight,
                                         minimize=False, edges=True)
            u = GraphView(g, efilt=m, directed=False)
            c = label_components(u)[0]
            c = g.own_property(c)
            u = GraphView(g, directed=False)
        elif kind == "modularity":
            B = max(g.num_vertices() // 2, 1)
            state = minimize_blockmodel_dl(GraphView(g, directed=False),
                                           state=ModularityState,
                                           state_args=dict(eweight=eweight),
                                           multilevel_mcmc_args=dict(B_min=B,
                                                                     B_max=B))
            print(state)
            c = state.b.copy()
        elif kind in ["sbm", "dcsbm"]:
            B = max(g.num_vertices() // 2, 1)
            w = eweight.copy("int") if eweight is not None else None
            state = minimize_blockmodel_dl(GraphView(g, directed=False),
                                           state=BlockState,
                                           state_args=dict(eweight=w,
                                                           deg_corr=kind == "dcsbm"),
                                           multilevel_mcmc_args=dict(B_min=B,
                                                                     B_max=B))
            c = state.b.copy()
        else:
            raise ValueError("invalid coarsening method:", kind)
    else:
        c = g.new_vp("int", vals=groups)
    cg, cc, vcount, ecount = condensation_graph(g, c, vweight, eweight,
                                                self_loops=True)[:4]
    return cg, cc, vcount, ecount, c, mivs


def _propagate_pos(g, cg, c, cc, cpos, delta, mivs):
    pos = g.new_vertex_property(cpos.value_type())

    if mivs is not None:
        g = GraphView(g, vfilt=mivs)
    libgraph_tool_layout.propagate_pos(g._Graph__graph,
                                       cg._Graph__graph,
                                       _prop("v", g, c),
                                       _prop("v", cg, cc),
                                       _prop("v", g, pos),
                                       _prop("v", cg, cpos),
                                       delta if mivs is None else 0,
                                       _get_rng())

    if mivs is not None:
        g = g.base
        u = GraphView(g, directed=False)
        libgraph_tool_layout.propagate_pos_mivs(u._Graph__graph,
                                                _prop("v", u, mivs),
                                                _prop("v", u, pos),
                                                delta, _get_rng())
    return pos


def _avg_edge_distance(g, pos):
    libgraph_tool_layout.sanitize_pos(g._Graph__graph, _prop("v", g, pos))
    ad = libgraph_tool_layout.avg_dist(g._Graph__graph, _prop("v", g, pos))
    if numpy.isnan(ad) or ad == 0:
        ad = 1.
    return ad


def coarse_graphs(g, method="hybrid", mivs_thres=0.9, ec_thres=0.75,
                  weighted_coarse=False, eweight=None, vweight=None,
                  groups=None, gamma=None, verbose=False):
    cg = [[g, None, None, None, None, None]]
    if weighted_coarse:
        cg[-1][2], cg[-1][3] = vweight, eweight
    hybrid = False
    if method == "hybrid":
        method = "ec"
        hybrid = True
    if verbose:
        print(f"Coarse level (none): 0, N={g.num_vertices()}")
    while True:
        if groups is None or len(groups) < len(cg):
            b = None
        else:
            b = groups[len(cg)-1]
        u = _coarse_graph(cg[-1][0], cg[-1][2], cg[-1][3], method, b)
        if hybrid:
            thres = mivs_thres if method == "mivs" else ec_thres
            if u[0].num_vertices() >= thres * cg[-1][0].num_vertices():
                if method == "ec":
                    method = "mivs"
                else:
                    break
        if u[0].num_vertices() <= 2:
            break
        cg.append(u)
        if verbose:
            print(f"Coarse level ({method}): {len(cg)}, N={u[0].num_vertices()}")
    cg.reverse()
    Ks = []
    pos = random_layout(cg[0][0], dim=2)
    for i in range(len(cg)):
        if i == 0:
            u = cg[i][0]
            K = _avg_edge_distance(u, pos)
            if K == 0:
                K = 1.
            Ks.append(K)
            continue
        if weighted_coarse:
            r = 1.
        else:
            r = 0.75
        Ks.append(Ks[-1] * r)

    if gamma is None:
        gamma = [0] * len(cg)
    else:
        if not isinstance(gamma, Iterable):
            gamma = [gamma]
        gamma = list(gamma) + [0] * (len(cg) - len(gamma))

    for i in range(len(cg)):
        u, cc, vcount, ecount, c, mivs = cg[i]
        if groups is None or len(cg) - i > len(groups):
            b = None
        else:
            b = groups[len(cg) - 1 - i:]
        yield u, pos, Ks[i], vcount, ecount, b, gamma[len(cg)-i-1:]

        if verbose:
            print("avg edge distance:", _avg_edge_distance(u, pos))

        if i < len(cg) - 1:
            if verbose:
                print("propagating...", end=' ')
                print(mivs.a.sum() if mivs is not None else "")
            pos = _propagate_pos(cg[i + 1][0], u, c, cc, pos,
                                 #Ks[i] / 1000.,
                                 1e-6,
                                 mivs)


def sfdp_layout(g, vweight=None, eweight=None, pin=None, C=0.2, K=None, p=2.,
                theta=0.6, max_level=15, r=1., kc=10, groups=None, gamma=.1,
                mu=2., kappa=1., rmap=None, R=1, init_step=None,
                cooling_step=0.95, adaptive_cooling=True, epsilon=1e-2,
                max_iter=0, pos=None, multilevel=None, coarse_method="hybrid",
                mivs_thres=0.9, ec_thres=0.75, weighted_coarse=False,
                verbose=False):
    r"""Obtain the SFDP spring-block layout of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    vweight : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        A vertex property map with the respective weights.
    eweight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        An edge property map with the respective weights.
    pin : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        A vertex property map with boolean values, which, if given,
        specify the vertices which will not have their positions modified.
    C : float (optional, default: ``0.2``)
        Relative strength of repulsive forces.
    K : float (optional, default: ``None``)
        Optimal edge length. If not provided, it will be taken to be the average
        edge distance in the initial layout.
    p : float (optional, default: ``2``)
        Repulsive force exponent.
    theta : float (optional, default: ``0.6``)
        Quadtree opening parameter, a.k.a. Barnes-Hut opening criterion.
    max_level : int (optional, default: ``15``)
        Maximum quadtree level.
    r : float (optional, default: ``1.``)
        Strength of attractive force between connected components.
    kc : int (optional, default: ``10``)
        Number of connected components to subsample.
    groups : :class:`~graph_tool.VertexPropertyMap` or list of list of integers (optional, default: ``None``)
        A vertex property map with group assignments. Vertices belonging to the
        same group will be put close together. Optionally, this can take an
        hierarchical partition, represented by a list of lists, where the list
        above correspond to the partition of the list below.
    gamma : float or list of floats (optional, default: ``.1``)
        Strength of the attractive force between nodes of the same group to
        their center of mass. In case of a hierarchical partition, this should
        correspond to a list of floats, containing the strenghts for each
        hierarchical level. This option has no effect if ``groups`` is ``None``.
    rmap : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex rank to be used around to order them preferentially in the
        :math:`y` direction.
    R : float (optional, default: ``1.0``)
        Strength of the rank ordering in the :math:`y` direction.
    init_step : float (optional, default: ``None``)
        Initial update step. If not provided, it will be chosen automatically.
    cooling_step : float (optional, default: ``0.95``)
        Cooling update step.
    adaptive_cooling : bool (optional, default: ``True``)
        Use an adaptive cooling scheme.
    epsilon : float (optional, default: ``0.01``)
        Relative convergence criterion.
    max_iter : int (optional, default: ``0``)
        Maximum number of iterations. If this value is ``0``, it runs until
        convergence.
    pos : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Initial vertex layout. If not provided, it will be randomly chosen.
    multilevel : bool (optional, default: ``None``)
        Use a multilevel layout algorithm. If ``None`` is given, it will be
        activated based on the size of the graph.
    coarse_method : str (optional, default: ``"hybrid"``)
        Coarsening method used if ``multilevel == True``. Allowed methods are
        ``"hybrid"``, ``"mivs"`` and ``"ec"``.
    mivs_thres : float (optional, default: ``0.9``)
        If the relative size of the MIVS coarse graph is above this value, the
        coarsening stops.
    ec_thres : float (optional, default: ``0.75``)
        If the relative size of the EC coarse graph is above this value, the
        coarsening stops.
    weighted_coarse : bool (optional, default: ``False``)
        Use weighted coarse graphs.
    verbose : bool (optional, default: ``False``)
        Provide verbose information.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        A vector-valued vertex property map with the coordinates of the
        vertices.

    Notes
    -----
    This algorithm is defined in [hu-multilevel-2005]_, and has
    complexity :math:`O(V\log V)`.

    Examples
    --------
    .. testcode::
       :hide:

       np.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.price_network(3000)
    >>> pos = gt.sfdp_layout(g)
    >>> gt.graph_draw(g, pos=pos, output="graph-draw-sfdp.pdf")
    <...>

    .. testcleanup::

       conv_png("graph-draw-sfdp.pdf")

    .. figure:: graph-draw-sfdp.png
       :align: center
       :width: 60%

       SFDP layout of a Price network.

    References
    ----------
    .. [hu-multilevel-2005] Yifan Hu, "Efficient and High Quality Force-Directed
       Graph", Mathematica Journal, vol. 10, Issue 1, pp. 37-71, (2005)
       http://www.mathematica-journal.com/issue/v10i1/graph_draw.html

    """

    if pos is None:
        pos = random_layout(g, dim=2)
    _check_prop_vector(pos, name="pos", floating=True)

    if isinstance(groups, VertexPropertyMap):
        groups = [numpy.asarray(groups.a.copy(), dtype="int32")]
    elif groups is not None:
        if isinstance(groups[0], VertexPropertyMap):
            groups = [groups[0].a.copy()] + list(groups[1:])
        groups = nested_contiguous_map(groups)

    if groups is None:
        gamma = [0]
    else:
        if not isinstance(gamma, Iterable):
            gamma = [gamma]
        if len(gamma) < len(groups):
            gamma = list(gamma) + [0] * (len(groups) - len(gamma))

    g_ = g
    g = GraphView(g, directed=False)

    if pin is not None:
        if pin.value_type() != "bool":
            raise ValueError("'pin' property must be of type 'bool'.")
    else:
        pin = g.new_vertex_property("bool")

    if K is None:
        K = _avg_edge_distance(g, pos)

    mu *= K

    if rmap is None:
        rmap = g.new_vp("double")
        R = 0
    elif rmap.value_type() != "double":
        rmap = rmap.copy("double")

    if init_step is None:
        init_step = 2 * max(_avg_edge_distance(g, pos), K)

    if multilevel is None:
        multilevel = g.num_vertices() > 1000

    if multilevel:
        if eweight is not None or vweight is not None:
            weighted_coarse = True
        cgs = coarse_graphs(g, method=coarse_method,
                            mivs_thres=mivs_thres,
                            ec_thres=ec_thres,
                            weighted_coarse=weighted_coarse,
                            eweight=eweight,
                            vweight=vweight,
                            groups=groups,
                            gamma=gamma,
                            verbose=verbose)
        for count, (u, pos, K_, vcount, ecount, groups, gamma) in enumerate(cgs):
            if verbose:
                print(f"Positioning level: {count}, {u.num_vertices()}", end=" ")
                print(f"with K={K_}, gamma={gamma} ...")
            pos = sfdp_layout(u, pos=pos,
                              vweight=vcount if weighted_coarse else None,
                              eweight=ecount if weighted_coarse else None,
                              groups=groups,
                              C=C, K=K_, p=p,
                              theta=theta,
                              gamma=gamma,
                              epsilon=epsilon,
                              max_iter=max_iter,
                              cooling_step=cooling_step,
                              adaptive_cooling=False,
                              # init_step=max(2 * K,
                              #               _avg_edge_distance(u, pos)),
                              multilevel=False,
                              verbose=False)
        pos = g_.own_property(pos)
        return pos

    if g.num_vertices() <= 1:
        return pos
    if g.num_vertices() == 2:
        vs = [g.vertex(0, False), g.vertex(1, False)]
        pos[vs[0]] = [0, 0]
        pos[vs[1]] = [1, 1]
        return pos
    if g.num_vertices() <= 50:
        max_level = 0

    if groups is None:
        groups = [numpy.zeros(g.num_vertices(True), dtype="int32")]

    c = label_components(g)[0]

    libgraph_tool_layout.sanitize_pos(g._Graph__graph, _prop("v", g, pos))
    libgraph_tool_layout.sfdp_layout(g._Graph__graph, _prop("v", g, pos),
                                     _prop("v", g, vweight),
                                     _prop("e", g, eweight),
                                     _prop("v", g, pin),
                                     (C, K, p, gamma, groups, r, kc,
                                      _prop("v", g, c), R, _prop("v", g, rmap)),
                                     theta, init_step, cooling_step, max_level,
                                     epsilon, max_iter, not adaptive_cooling,
                                     verbose, _get_rng())
    pos = g_.own_property(pos)
    return pos

def radial_tree_layout(g, root, rel_order=None, rel_order_leaf=False,
                       weighted=False, node_weight=None, r=1.):
    r"""Computes a radial layout of the graph according to the minimum spanning
    tree centered at the ``root`` vertex.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    root : :class:`~graph_tool.Vertex` or ``int``
        The root of the radial tree.
    rel_order : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Relative order of the nodes at each respective branch.
    rel_order_leaf : ``bool`` (optional, default: ``False``)
        If ``True``, the relative order of the leafs will propagate to the
        root. Otherwise they will propagate in the opposite direction.
    weighted : ``bool`` (optional, default: ``False``)
        If true, the angle between the child branches will be computed according
        to weight of the entire sub-branches.
    node_weight : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        If given, the relative spacing between leafs will correspond to the node
        weights.
    r : ``float`` (optional, default: ``1.``)
        Layer spacing.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        A vector-valued vertex property map with the coordinates of the
        vertices.

    Notes
    -----
    This algorithm has complexity :math:`O(V + E)`, or :math:`O(V\log V + E)` if
    ``rel_order`` is given.

    Examples
    --------
    .. testcode::
       :hide:

       np.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.price_network(1000)
    >>> pos = gt.radial_tree_layout(g, g.vertex(0))
    >>> gt.graph_draw(g, pos=pos, output="graph-draw-radial.pdf")
    <...>

    .. testcleanup::

       conv_png("graph-draw-radial.pdf")

    .. figure:: graph-draw-radial.png
       :align: center
       :width: 60%

       Radial tree layout of a Price network.

    """

    levels, pred_map = shortest_distance(GraphView(g, directed=False), root,
                                         pred_map=True)
    t = predecessor_tree(g, pred_map)
    pos = t.new_vertex_property("vector<double>")
    levels = t.own_property(levels)
    if rel_order is None:
        rel_order = g.vertex_index.copy("int")
    if node_weight is None:
        node_weight = g.new_vertex_property("double", 1)
    elif node_weight.value_type() != "double":
        node_weight = node_weight.copy("double")
    libgraph_tool_layout.get_radial(t._Graph__graph,
                                    _prop("v", t, pos),
                                    _prop("v", t, levels),
                                    _prop("v", g, rel_order),
                                    _prop("v", g, node_weight),
                                    int(root), weighted, r,
                                    rel_order_leaf)
    return g.own_property(pos)

def prop_to_size(prop, mi=0, ma=5, log=False, power=0.5):
    r"""Convert property map values to be more useful as a vertex size, or edge
    width. The new values are taken to be

    .. math::

        y_i = mi + (ma - mi) \left(\frac{x_i - \min(x)} {\max(x) - \min(x)}\right)^\text{power}

    If ``log=True``, :math:`x_i` is replaced with :math:`\ln(x_i)`.

    If :math:`\max(x) - \min(x)` is zero, :math:`y_i = mi`.

    """
    prop = prop.copy(value_type="double")
    if len(prop.fa) == 0:
        return prop

    if log:
        vals = numpy.log(prop.fa)
    else:
        vals = prop.fa

    delta = vals.max() - vals.min()
    if delta == 0:
        delta = 1
    prop.fa = mi + (ma - mi) * ((vals - vals.min()) / delta) ** power
    return prop

try:
    from . cairo_draw import graph_draw, cairo_draw, \
        get_hierarchy_control_points, default_cm,  default_cm_bin, \
        default_clrs, draw_hierarchy
    __all__ += ["draw_hierarchy"]
except ImportError:
    pass

try:
    from . cairo_draw import GraphWidget, GraphWindow, \
        interactive_window, GraphArtist
    __all__ += ["interactive_window", "GraphWidget", "GraphWindow", "GraphArtist"]
except ImportError:
    pass

try:
   from . graphviz_draw import graphviz_draw
except ImportError:
   pass
