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
#
# -----------------------------------------------------------------------------
# This collection of small graphs is a modification of on the collection
# available in the NetworkX library, released under the 3-clause BSD license
# below.
# -----------------------------------------------------------------------------
#
# Copyright (C) 2004-2023, NetworkX Developers
# Aric Hagberg <hagberg@lanl.gov>
# Dan Schult <dschult@colgate.edu>
# Pieter Swart <swart@lanl.gov>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#
#   * Neither the name of the NetworkX Developers nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from .. import Graph
from .. generation import circular_graph, complete_graph, remove_parallel_edges

def LCF_graph(n, shift_list, repeats):
    """Returns the cubic graph specified in LCF notation.

    Parameters
    ----------
    n : ``int``
        Number of nodes. The starting graph is the n-cycle with nodes
        :math:`0,\dots,n-1`.  (The empty graph is returned if ``n < 0``.)

    shift_list : ``list``
        A list :math:`[s_1,s_2,\dots,s_k]` of integer shifts :math:`\mod n`.

    repeats : ``int``
      Integer specifying the number of times that shifts in ``shift_list``
      are successively applied to each ``v_current`` in the n-cycle
      to generate an edge between ``v_current`` and `v_current + shift mod n.`

    Notes
    -----

    The Lederberg-Coxeter-Fruchte (LCF) notation is a compressed notation used
    in the generation of various cubic Hamiltonian graphs of high symmetry
    [LCF]_.

    See, for example, :func:`~graph_tool.collection.dodecahedral_graph`,
    :func:`~graph_tool.collection.desargues_graph`,
    :func:`~graph_tool.collection.heawood_graph` and
    :func:`~graph_tool.collection.pappus_graph`.

    For ``v1`` cycling through the n-cycle a total of ``k * repeats`` with shift
    cycling through shiftlist repeats times connect ``v1`` with ``v1 + shift mod
    n``.

    Examples
    --------
    The utility graph :math:`K_{3,3}`

    >>> g = gt.LCF_graph(6, [3, -3], 3)

    The Heawood graph

    >>> g = gt.LCF_graph(14, [5, -5], 7)

    References
    ----------
    .. [LCF] http://mathworld.wolfram.com/LCFNotation.html

    """

    if n <= 0:
        return Graph(directed=False)

    g = circular_graph(n)

    ne = repeats * len(shift_list)
    if ne < 1:
        return g

    for i in range(ne):
        shift = shift_list[i % len(shift_list)]
        v1 = i % n
        v2 = (i + shift) % n
        if g.edge(v1, v2) is None:
            g.add_edge(v1, v2)
    return g

def petersen_graph():
    """Returns the Petersen graph.

    Notes
    -----
    The Peterson graph is a cubic, undirected graph with 10 nodes and 15 edges
    [petersen]_.

    Julius Petersen constructed the graph as the smallest counterexample against
    the claim that a connected bridgeless cubic graph has an edge colouring with
    three colours [petersen_color]_.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Petersen graph

    References
    ----------
    .. [petersen] https://en.wikipedia.org/wiki/Petersen_graph
    .. [petersen_color] https://www.win.tue.nl/~aeb/drg/graphs/Petersen.html

    """

    return Graph({ 0: [1, 4, 5],
                   1: [2, 6],
                   2: [3, 7],
                   3: [4, 8],
                   4: [9],
                   5: [7, 8],
                   6: [8, 9],
                   7: [9],
                  }, directed=False)

def tutte_graph():
    """Returns the Tutte graph.

    Notes
    -----
    The Tutte graph is a cubic polyhedral, non-Hamiltonian graph. It has
    46 nodes and 69 edges.

    It is a counterexample to Tait's conjecture that every 3-regular polyhedron
    has a Hamiltonian cycle.

    It can be realized geometrically from a tetrahedron by multiply truncating
    three of its vertices [tutte_graph]_.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Tutte graph

    References
    ----------
    .. [tutte_graph] https://en.wikipedia.org/wiki/Tutte_graph

    """

    return Graph({0: [1, 2, 3],
                  1: [4, 26],
                  2: [10, 11],
                  3: [18, 19],
                  4: [5, 33],
                  5: [6, 29],
                  6: [7, 27],
                  7: [8, 14],
                  8: [9, 38],
                  9: [10, 37],
                  10: [39],
                  11: [12, 39],
                  12: [13, 35],
                  13: [14, 15],
                  14: [34],
                  15: [16, 22],
                  16: [17, 44],
                  17: [18, 43],
                  18: [45],
                  19: [20, 45],
                  20: [21, 41],
                  21: [22, 23],
                  22: [40],
                  23: [24, 27],
                  24: [25, 32],
                  25: [26, 31],
                  26: [33],
                  27: [28],
                  28: [29, 32],
                  29: [30],
                  30: [31, 33],
                  31: [32],
                  34: [35, 38],
                  35: [36],
                  36: [37, 39],
                  37: [38],
                  40: [41, 44],
                  41: [42],
                  42: [43, 45],
                  43: [44],
                  }, directed = False)

def bull_graph():
    """
    Returns the Bull Graph

    Notes
    -----
    The Bull Graph has 5 nodes and 5 edges. It is a planar undirected
    graph in the form of a triangle with two disjoint pendant edges [bull_graph]_
    The name comes from the triangle and pendant edges representing
    respectively the body and legs of a bull.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        A bull graph with 5 nodes

    References
    ----------
    .. [bull_graph] https://en.wikipedia.org/wiki/Bull_graph

    """
    return Graph({0: [1, 2], 1: [2, 3], 2: [4]},
                 directed=False)

def chvatal_graph():
    """
    Returns the Chvátal Graph

    Notes
    -----
    The Chvátal Graph is an undirected graph with 12 nodes and 24 edges
    [chvatal_wiki]_.  It has 370 distinct (directed) Hamiltonian cycles, giving
    a unique generalized LCF notation of order 4, two of order 6 , and 43 of
    order 1 [chvatal]_.


    Returns
    -------
    g : :class:`~graph_tool.Graph`
        The Chvátal graph with 12 nodes and 24 edges

    References
    ----------
    .. [chvatal_wiki] https://en.wikipedia.org/wiki/Chv%C3%A1tal_graph
    .. [chvatal] https://mathworld.wolfram.com/ChvatalGraph.html

    """
    return Graph({0: [1, 4, 6, 9],
                  1: [2, 5, 7],
                  2: [3, 6, 8],
                  3: [4, 7, 9],
                  4: [5, 8],
                  5: [10, 11],
                  6: [10, 11],
                  7: [8, 11],
                  8: [10],
                  9: [10, 11]}, directed=False)

def cubical_graph():
    """Returns the 3-regular Platonic Cubical Graph

    Notes
    -----
    The skeleton of the cube (the nodes and edges) form a graph, with 8 nodes,
    and 12 edges. It is a special case of the hypercube graph.  It is one of 5
    Platonic graphs, each a skeleton of its Platonic solid [cubical]_. Such
    graphs arise in parallel processing in computers.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        A cubical graph with 8 nodes and 12 edges

    References
    ----------
    .. [cubical] https://en.wikipedia.org/wiki/Cube#Cubical_graph

    """
    return Graph({0: [1, 3, 4],
                  1: [2, 7],
                  2: [3, 6],
                  3: [5],
                  4: [5, 7],
                  5: [6],
                  6: [7],
                  },directed=False)
    return G

def desargues_graph():
    """
    Returns the Desargues Graph

    Notes
    -----
    The Desargues Graph is a non-planar, distance-transitive cubic graph
    with 20 nodes and 30 edges [desargues_wiki]_.
    It is a symmetric graph. It can be represented in LCF notation
    as :math:`[5,-5,9,-9]^5` [desargues]_.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Desargues Graph with 20 nodes and 30 edges

    References
    ----------
    .. [desargues_wiki] https://en.wikipedia.org/wiki/Desargues_graph
    .. [desargues] https://mathworld.wolfram.com/DesarguesGraph.html
    """
    return LCF_graph(20, [5, -5, 9, -9], 5)

def diamond_graph():
    """
    Returns the Diamond graph

    Notes
    -----
    The Diamond Graph is  planar undirected graph with 4 nodes and 5 edges.
    It is also sometimes known as the double triangle graph or kite graph [diamond]_.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Diamond Graph with 4 nodes and 5 edges

    References
    ----------
    .. [diamond] https://mathworld.wolfram.com/DiamondGraph.html
    """
    return Graph({0: [1, 2], 1: [2, 3], 2: [3]}, directed=False)

def dodecahedral_graph():
    """
    Returns the Platonic Dodecahedral graph.

    Notes
    -----
    The dodecahedral graph has 20 nodes and 30 edges. The skeleton of the
    dodecahedron forms a graph. It is one of 5 Platonic graphs [dodecahedral_wiki]_.
    It can be described in LCF notation as
    :math:`[10, 7, 4, -4, -7, 10, -4, 7, -7, 4]^2` [dodecahedral]_.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Dodecahedral Graph with 20 nodes and 30 edges

    References
    ----------
    .. [dodecahedral_wiki] https://en.wikipedia.org/wiki/Regular_dodecahedron#Dodecahedral_graph
    .. [dodecahedral] https://mathworld.wolfram.com/DodecahedralGraph.html

    """
    return LCF_graph(20, [10, 7, 4, -4, -7, 10, -4, 7, -7, 4], 2)

def frucht_graph():
    """Returns the Frucht Graph.

    Notes
    -----
    The Frucht Graph is the smallest cubical graph whose automorphism group
    consists only of the identity element [frucht_wiki]_.  It has 12 nodes and
    18 edges and no nontrivial symmetries. It is planar and Hamiltonian
    [frucht]_.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Frucht Graph with 12 nodes and 18 edges

    References
    ----------
    .. [frucht_wiki] https://en.wikipedia.org/wiki/Frucht_graph
    .. [frucht] https://mathworld.wolfram.com/FruchtGraph.html

    """
    g = circular_graph(7)
    g.add_edge_list([[0, 7],
                     [1, 7],
                     [2, 8],
                     [3, 9],
                     [4, 9],
                     [5, 10],
                     [6, 10],
                     [7, 11],
                     [8, 11],
                     [8, 9],
                     [10, 11]], directed=False)
    return g



def heawood_graph():
    """
    Returns the Heawood Graph, a (3,6) cage.

    Notes
    -----
    The Heawood Graph is an undirected graph with 14 nodes and 21 edges, named
    after Percy John Heawood [heawood_wiki]_. It is cubic symmetric, nonplanar,
    Hamiltonian, and can be represented in LCF notation as :math:`[5,-5]^7`
    [heawood]_.  It is the unique (3,6)-cage: the regular cubic graph of girth 6
    with minimal number of vertices [heawood_cage]_.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Heawood Graph with 14 nodes and 21 edges

    References
    ----------
    .. [heawood_wiki] https://en.wikipedia.org/wiki/Heawood_graph
    .. [heawood] https://mathworld.wolfram.com/HeawoodGraph.html
    .. [heawood_cage] https://www.win.tue.nl/~aeb/graphs/Heawood.html

    """
    return LCF_graph(14, [5, -5], 7)

def hoffman_singleton_graph():
    """
    Returns the Hoffman-Singleton Graph.

    Notes
    -----
    The Hoffman–Singleton graph is a symmetrical undirected graph
    with 50 nodes and 175 edges.

    All indices lie in :math:`\mathbb{Z} \mod 5`, that is, the integers modulo 5
    [hoffman]_.

    It is the only regular graph of vertex degree 7, diameter 2, and girth 5.

    It is the unique (7,5)-cage graph and Moore graph, and contains many
    copies of the Petersen graph [hoffman-singleton]_.

    Constructed from pentagon and pentagram as follows [hoffman-wiki]_:

    1. Take five pentagons :math:`P_h` and five pentagrams :math:`Q_i` .
    2. Join vertex :math:`j` of :math:`P_h` to vertex :math:`h\times i + j` of :math:`Q_i`.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Hoffman–Singleton Graph with 50 nodes and 175 edges

    References
    ----------
    .. [hoffman] https://blogs.ams.org/visualinsight/2016/02/01/hoffman-singleton-graph/
    .. [hoffman-singleton] https://mathworld.wolfram.com/Hoffman-SingletonGraph.html
    .. [hoffman-wiki] https://en.wikipedia.org/wiki/Hoffman%E2%80%93Singleton_graph

    """
    def elist():
        for i in range(5):
            for j in range(5):
                yield ("pentagon", i, j), ("pentagon", i, (j - 1) % 5)
                yield ("pentagon", i, j), ("pentagon", i, (j + 1) % 5)
                yield ("pentagram", i, j), ("pentagram", i, (j - 2) % 5)
                yield ("pentagram", i, j), ("pentagram", i, (j + 2) % 5)
                for k in range(5):
                    yield ("pentagon", i, j), ("pentagram", k, (i * k + j) % 5)
    g = Graph(directed=False)
    g.add_edge_list(elist(), hashed=True, hash_type="object")
    remove_parallel_edges(g)
    return g

def house_graph(x=False):
    """Returns the House graph (square with triangle on top).

    Notes
    -----
    The house graph is a simple undirected graph with
    5 nodes and 6 edges [house]_.

    Parameters
    ----------
    x : ``bool`` (optional, default: ``False``)
       If ``True``, then two edges are added connecting diagonally opposite
       vertices of the square base.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        House graph in the form of a square with a triangle on top.

    References
    ----------
    .. [house] https://mathworld.wolfram.com/HouseGraph.html

    """
    g = Graph({0: [1, 2], 1: [3], 2: [3, 4], 3: [4]},
              directed=False)
    if x:
        g.add_edge_list([(0, 3), (1, 2)])
    return g

def icosahedral_graph():
    """Returns the Platonic Icosahedral graph.

    Notes
    -----
    The icosahedral graph has 12 nodes and 30 edges. It is a Platonic graph
    whose nodes have the connectivity of the icosahedron. It is undirected,
    regular and Hamiltonian [icosahedral]_.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Icosahedral graph with 12 nodes and 30 edges.

    References
    ----------
    .. [icosahedral] https://mathworld.wolfram.com/IcosahedralGraph.html

    """
    return Graph({0: [1, 5, 7, 8, 11],
                  1: [2, 5, 6, 8],
                  2: [3, 6, 8, 9],
                  3: [4, 6, 9, 10],
                  4: [5, 6, 10, 11],
                  5: [6, 11],
                  7: [8, 9, 10, 11],
                  8: [9],
                  9: [10],
                  10: [11],
                  }, directed=False)


def krackhardt_kite_graph():
    """Returns the Krackhardt Kite Social Network.

    Notes
    -----
    A 10 actor social network introduced by David Krackhardt
    to illustrate different centrality measures [krackhardt]_.

    The traditional labeling is:

    Andre=1, Beverley=2, Carol=3, Diane=4, Ed=5, Fernando=6, Garth=7, Heather=8,
    Ike=9, Jane=10.


    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Krackhardt Kite graph with 10 nodes and 18 edges

    References
    ----------
    .. [krackhardt] Krackhardt, David. “Assessing the Political Landscape: Structure,
       Cognition, and Power in Organizations”. Administrative Science Quarterly.
       35 (2): 342–369. JSTOR 2393394. June 1990. :doi:`10.2307/2393394`

    """
    g = Graph({0: [1, 2, 3, 5],
               1: [3, 4, 6],
               2: [3, 5],
               3: [4, 5, 6],
               4: [6],
               5: [6, 7],
               6: [7],
               7: [8],
               8: [9],
               }, directed=False)
    g.vp.label = g.new_vp("string",
                          vals=["Andre", "Beverley", "Carol", "Diane", "Ed",
                                "Fernando", "Garth", "Heather", "Ike", "Jane"])
    return g

def moebius_kantor_graph():
    """Returns the Moebius-Kantor graph.

    Notes
    -----
    The Möbius-Kantor graph is the cubic symmetric graph on 16 nodes.  Its LCF
    notation is :math:`[5,-5]^8`, and it is isomorphic to the generalized
    Petersen graph [moebius_kantor]_.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Moebius-Kantor graph

    References
    ----------
    .. [moebius_kantor] https://en.wikipedia.org/wiki/M%C3%B6bius%E2%80%93Kantor_graph

    """
    return LCF_graph(16, [5, -5], 8)

def octahedral_graph():
    """
    Returns the Platonic Octahedral graph.

    Notes
    -----
    The octahedral graph is the 6-node 12-edge Platonic graph having the
    connectivity of the octahedron [octahedral]_. If 6 couples go to a party,
    and each person shakes hands with every person except his or her partner,
    then this graph describes the set of handshakes that take place;
    for this reason it is also called the cocktail party graph [octahedral_wiki]_.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Octahedral graph

    References
    ----------
    .. [octahedral] https://mathworld.wolfram.com/OctahedralGraph.html
    .. [octahedral_wiki] https://en.wikipedia.org/wiki/Tur%C3%A1n_graph#Special_cases

    """
    return Graph({0: [1, 2, 3, 4], 1: [2, 3, 5], 2: [4, 5], 3: [4, 5], 4: [5]},
                 directed=False)

def pappus_graph():
    """
    Returns the Pappus graph.

    Notes
    -----
    The Pappus graph is a cubic symmetric distance-regular graph with 18 nodes
    and 27 edges. It is Hamiltonian and can be represented in LCF notation as
    :math:`[5,7,-7,7,-7,-5]^3` [pappus]_.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Pappus graph

    References
    ----------
    .. [pappus] https://en.wikipedia.org/wiki/Pappus_graph
    """
    return LCF_graph(18, [5, 7, -7, 7, -7, -5], 3)

def sedgewick_maze_graph():
    """
    Return a small maze with a cycle.

    Notes
    -----
    This is the maze used in Sedgewick, 3rd Edition, Part 5, Graph
    Algorithms, Chapter 18, e.g. Figure 18.2 and following [sedgewick]_.
    Nodes are numbered ``0,..,7``.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Small maze with a cycle

    References
    ----------
    .. [sedgewick] Figure 18.2, Chapter 18, Graph Algorithms (3rd Ed), Sedgewick
    """
    return Graph([[0, 2], [0, 7], [0, 5], [1, 7], [2, 6], [3, 4], [3, 5],
                  [4, 5], [4, 7], [4, 6]], directed=False)

def tetrahedral_graph():
    """
    Returns the 3-regular Platonic Tetrahedral graph.

    Notes
    -----
    Tetrahedral graph has 4 nodes and 6 edges. It is a
    special case of the complete graph, K4, and wheel graph, W4.
    It is one of the 5 platonic graphs [tetrahedral]_.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Tetrahedral Grpah

    References
    ----------
    .. [tetrahedral] https://en.wikipedia.org/wiki/Tetrahedron#Tetrahedral_graph

    """
    return complete_graph(4)

def truncated_cube_graph():
    """Returns the skeleton of the truncated cube.

    Notes
    -----
    The truncated cube is an Archimedean solid with 14 regular faces (6
    octagonal and 8 triangular), 36 edges and 24 nodes [truncated_cube]_.  The
    truncated cube is created by truncating (cutting off) the tips of the cube
    one third of the way into each edge [truncated_cube_cut]_.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Skeleton of the truncated cube

    References
    ----------
    .. [truncated_cube] https://en.wikipedia.org/wiki/Truncated_cube
    .. [truncated_cube_cut] https://www.coolmath.com/reference/polyhedra-truncated-cube

    """
    return Graph({0: [1, 2, 4],
                  1: [11, 14],
                  2: [3, 4],
                  3: [6, 8],
                  4: [5],
                  5: [16, 18],
                  6: [7, 8],
                  7: [10, 12],
                  8: [9],
                  9: [17, 20],
                  10: [11, 12],
                  11: [14],
                  12: [13],
                  13: [21, 22],
                  14: [15],
                  15: [19, 23],
                  16: [17, 18],
                  17: [20],
                  18: [19],
                  19: [23],
                  20: [21],
                  21: [22],
                  22: [23],
                  }, directed=False)

def truncated_tetrahedron_graph():
    """Returns the skeleton of the truncated Platonic tetrahedron.

    Notes
    -----
    The truncated tetrahedron is an Archimedean solid with 4 regular hexagonal
    faces, 4 equilateral triangle faces, 12 nodes and 18 edges. It can be
    constructed by truncating all 4 vertices of a regular tetrahedron at one
    third of the original edge length [truncated_tetrahedron]_.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        Skeleton of the truncated tetrahedron

    References
    ----------
    .. [truncated_tetrahedron] https://en.wikipedia.org/wiki/Truncated_tetrahedron

    """
    g = circular_graph(12)
    g.remove_edge((11, 0))
    g.add_edge_list([(0, 2), (0, 9), (1, 6), (3, 11), (4, 11), (5, 7), (8, 10)])
    return g
