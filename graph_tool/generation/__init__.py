#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2024 Tiago de Paula Peixoto <tiago@skewed.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.s

"""
``graph_tool.generation``
-------------------------

This module contains functions that generate different kinds of graphs.

Random graph generation
+++++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   random_graph
   random_rewire
   add_random_edges
   remove_random_edges
   generate_triadic_closure
   price_network

Stochastic block models
+++++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   generate_sbm
   generate_maxent_sbm
   solve_sbm_fugacities

Geometric models
++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   generate_knn
   geometric_graph
   triangulation

Graph transformations
+++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   predecessor_tree
   line_graph
   condensation_graph
   contract_parallel_edges
   remove_parallel_edges
   expand_parallel_edges
   label_parallel_edges
   remove_self_loops
   label_self_loops

Graph operations
++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   graph_union

Deterministic graphs
++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   lattice
   complete_graph
   circular_graph

"""

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_generation")

from .. import Graph, GraphView, _check_prop_scalar, _prop, _limit_args, \
    _gt_type, _get_rng, Vector_double, VertexPropertyMap, _parallel
import inspect
import types
import numpy
import numpy.random
import scipy.optimize
import scipy.sparse

__all__ = ["random_graph", "random_rewire", "add_random_edges",
           "remove_random_edges", "generate_sbm", "solve_sbm_fugacities",
           "generate_maxent_sbm", "generate_knn", "generate_triadic_closure",
           "predecessor_tree", "line_graph", "graph_union", "triangulation",
           "lattice", "geometric_graph", "price_network", "complete_graph",
           "circular_graph", "condensation_graph", "contract_parallel_edges",
           "expand_parallel_edges", "remove_parallel_edges", "remove_self_loops",
           "label_parallel_edges", "label_self_loops", "remove_labeled_edges"]

def random_graph(N, deg_sampler, directed=True,
                 parallel_edges=False, self_loops=False, block_membership=None,
                 block_type="int", degree_block=False,
                 random=True, verbose=False, **kwargs):
    r"""
    Generate a random graph, with a given degree distribution and (optionally)
    vertex-vertex correlation.

    The graph will be randomized via the :func:`~graph_tool.generation.random_rewire`
    function, and any remaining parameters will be passed to that function.
    Please read its documentation for all the options regarding the different
    statistical models which can be chosen.

    Parameters
    ----------
    N : int
        Number of vertices in the graph.
    deg_sampler : function
        A degree sampler function which is called without arguments, and returns
        a tuple of ints representing the in and out-degree of a given vertex (or
        a single int for undirected graphs, representing the out-degree). This
        function is called once per vertex, but may be called more times, if the
        degree sequence cannot be used to build a graph.

        Optionally, you can also pass a function which receives one or two
        arguments. If ``block_membership is None``, the single argument passed
        will be the index of the vertex which will receive the degree.  If
        ``block_membership is not None``, the first value passed will be the vertex
        index, and the second will be the block value of the vertex.
    directed : bool (optional, default: ``True``)
        Whether the generated graph should be directed.
    parallel_edges : bool (optional, default: ``False``)
        If ``True``, parallel edges are allowed.
    self_loops : bool (optional, default: ``False``)
        If ``True``, self-loops are allowed.
    block_membership : list or :class:`numpy.ndarray` or function (optional, default: ``None``)
        If supplied, the graph will be sampled from a stochastic blockmodel
        ensemble, and this parameter specifies the block membership of the
        vertices, which will be passed to the
        :func:`~graph_tool.generation.random_rewire` function.

        If the value is a list or a :class:`numpy.ndarray`, it must have
        ``len(block_membership) == N``, and the values will define to which
        block each vertex belongs.

        If this value is a function, it will be used to sample the block
        types. It must be callable either with no arguments or with a single
        argument which will be the vertex index. In either case it must return
        a type compatible with the ``block_type`` parameter.

        See the documentation for the ``vertex_corr`` parameter of the
        :func:`~graph_tool.generation.random_rewire` function which specifies
        the correlation matrix.
    block_type : string (optional, default: ``"int"``)
        Value type of block labels. Valid only if ``block_membership is not None``.
    degree_block : bool (optional, default: ``False``)
        If ``True``, the degree of each vertex will be appended to block labels
        when constructing the blockmodel, such that the resulting block type
        will be a pair :math:`(r, k)`, where :math:`r` is the original block
        label.
    random : bool (optional, default: ``True``)
        If ``True``, the returned graph is randomized. Otherwise a deterministic
        placement of the edges will be used.
    verbose : bool (optional, default: ``False``)
        If ``True``, verbose information is displayed.

    Returns
    -------
    random_graph : :class:`~graph_tool.Graph`
        The generated graph.
    blocks : :class:`~graph_tool.VertexPropertyMap`
        A vertex property map with the block values. This is only returned if
        ``block_membership is not None``.

    See Also
    --------
    random_rewire: in-place graph shuffling

    Notes
    -----
    The algorithm makes sure the degree sequence is graphical (i.e. realizable)
    and keeps re-sampling the degrees if is not. With a valid degree sequence,
    the edges are placed deterministically, and later the graph is shuffled with
    the :func:`~graph_tool.generation.random_rewire` function, with all
    remaining parameters passed to it.

    The complexity is :math:`O(V + E)` if parallel edges are allowed, and
    :math:`O(V + E \times\text{n-iter})` if parallel edges are not allowed.


    .. note ::

        If ``parallel_edges == False`` this algorithm only guarantees that the
        returned graph will be a random sample from the desired ensemble if
        ``n_iter`` is sufficiently large. The algorithm implements an
        efficient Markov chain based on edge swaps, with a mixing time which
        depends on the degree distribution and correlations desired. If degree
        correlations are provided, the mixing time tends to be larger.

    Examples
    --------

    .. testcode::
       :hide:

       import numpy.random
       from pylab import *
       np.random.seed(43)
       gt.seed_rng(42)

    This is a degree sampler which uses rejection sampling to sample from the
    distribution :math:`P(k)\propto 1/k`, up to a maximum.

    >>> def sample_k(max):
    ...     accept = False
    ...     while not accept:
    ...         k = np.random.randint(1,max+1)
    ...         accept = np.random.random() < 1.0/k
    ...     return k

    The following generates a random undirected graph with degree distribution
    :math:`P(k)\propto 1/k` (with k_max=40) and an *assortative* degree
    correlation of the form:

    .. math::

        P(i,k) \propto \frac{1}{1+|i-k|}

    >>> g = gt.random_graph(1000, lambda: sample_k(40), model="probabilistic-configuration",
    ...                     edge_probs=lambda i, k: 1.0 / (1 + abs(i - k)), directed=False,
    ...                     n_iter=100)

    The following samples an in,out-degree pair from the joint distribution:

    .. math::

        p(j,k) = \frac{1}{2}\frac{e^{-m_1}m_1^j}{j!}\frac{e^{-m_1}m_1^k}{k!} +
                 \frac{1}{2}\frac{e^{-m_2}m_2^j}{j!}\frac{e^{-m_2}m_2^k}{k!}

    with :math:`m_1 = 4` and :math:`m_2 = 20`.

    >>> def deg_sample():
    ...    if random() > 0.5:
    ...        return np.random.poisson(4), np.random.poisson(4)
    ...    else:
    ...        return np.random.poisson(20), np.random.poisson(20)
    ...

    The following generates a random directed graph with this distribution, and
    plots the combined degree correlation.

    >>> g = gt.random_graph(20000, deg_sample)
    >>>
    >>> hist = gt.combined_corr_hist(g, "in", "out")
    >>>
    >>> figure()
    <...>
    >>> imshow(hist[0].T, interpolation="nearest", origin="lower")
    <...>
    >>> colorbar()
    <...>
    >>> xlabel("in-degree")
    Text(...)
    >>> ylabel("out-degree")
    Text(...)
    >>> tight_layout()
    >>> savefig("combined-deg-hist.svg")

    .. figure:: combined-deg-hist.*
        :align: center

        Combined degree histogram.

    A correlated directed graph can be build as follows. Consider the following
    degree correlation:

    .. math::

         P(j',k'|j,k)=\frac{e^{-k}k^{j'}}{j'!}
         \frac{e^{-(20-j)}(20-j)^{k'}}{k'!}

    i.e., the in->out correlation is "disassortative", the out->in correlation
    is "assortative", and everything else is uncorrelated.
    We will use a flat degree distribution in the range [1,20).

    >>> p = scipy.stats.poisson
    >>> g = gt.random_graph(20000, lambda: (sample_k(19), sample_k(19)),
    ...                     model="probabilistic-configuration",
    ...                     edge_probs=lambda a,b: (p.pmf(a[0], b[1]) *
    ...                                             p.pmf(a[1], 20 - b[0])),
    ...                     n_iter=100)

    Lets plot the average degree correlations to check.

    >>> figure(figsize=(8,3))
    <...>
    >>> corr = gt.avg_neighbor_corr(g, "in", "in")
    >>> errorbar(corr[2][:-1], corr[0], yerr=corr[1], fmt="o-",
    ...         label=r"$\left<\text{in}\right>$ vs in")
    <...>
    >>> corr = gt.avg_neighbor_corr(g, "in", "out")
    >>> errorbar(corr[2][:-1], corr[0], yerr=corr[1], fmt="o-",
    ...         label=r"$\left<\text{out}\right>$ vs in")
    <...>
    >>> corr = gt.avg_neighbor_corr(g, "out", "in")
    >>> errorbar(corr[2][:-1], corr[0], yerr=corr[1], fmt="o-",
    ...          label=r"$\left<\text{in}\right>$ vs out")
    <...>
    >>> corr = gt.avg_neighbor_corr(g, "out", "out")
    >>> errorbar(corr[2][:-1], corr[0], yerr=corr[1], fmt="o-",
    ...          label=r"$\left<\text{out}\right>$ vs out")
    <...>
    >>> legend(loc='center left', bbox_to_anchor=(1, 0.5))
    <...>
    >>> xlabel("Source degree")
    Text(...)
    >>> ylabel("Average target degree")
    Text(...)
    >>> tight_layout()
    >>> box = gca().get_position()
    >>> gca().set_position([box.x0, box.y0, box.width * 0.7, box.height])
    >>> savefig("deg-corr-dir.svg")

    .. figure:: deg-corr-dir.*
        :align: center

        Average nearest neighbor correlations.


    **Stochastic blockmodels**


    The following example shows how a stochastic blockmodel
    [holland-stochastic-1983]_ [karrer-stochastic-2011]_ can be generated. We
    will consider a system of 10 blocks, which form communities. The connection
    probability will be given by

    >>> def prob(a, b):
    ...    if a == b:
    ...        return 0.999
    ...    else:
    ...        return 0.001

    The blockmodel can be generated as follows.

    >>> g, bm = gt.random_graph(2000, lambda: poisson(10), directed=False,
    ...                         model="blockmodel",
    ...                         block_membership=lambda: randint(10),
    ...                         edge_probs=prob)
    >>> gt.graph_draw(g, vertex_fill_color=bm, edge_color="black", output="blockmodel.pdf")
    <...>

    .. testcleanup::

       conv_png("blockmodel.pdf")

    .. figure:: blockmodel.png
       :align: center
       :width: 80%

       Simple blockmodel with 10 blocks.


    References
    ----------
    .. [metropolis-equations-1953]  Metropolis, N.; Rosenbluth, A.W.;
       Rosenbluth, M.N.; Teller, A.H.; Teller, E. "Equations of State
       Calculations by Fast Computing Machines". Journal of Chemical Physics 21
       (6): 1087-1092 (1953). :doi:`10.1063/1.1699114`
    .. [hastings-monte-carlo-1970] Hastings, W.K. "Monte Carlo Sampling Methods
       Using Markov Chains and Their Applications". Biometrika 57 (1): 97-109 (1970).
       :doi:`10.1093/biomet/57.1.97`
    .. [holland-stochastic-1983] Paul W. Holland, Kathryn Blackmond Laskey, and
       Samuel Leinhardt, "Stochastic blockmodels: First steps," Social Networks
       5, no. 2: 109-13 (1983) :doi:`10.1016/0378-8733(83)90021-7`
    .. [karrer-stochastic-2011] Brian Karrer and M. E. J. Newman, "Stochastic
       blockmodels and community structure in networks," Physical Review E 83,
       no. 1: 016107 (2011) :doi:`10.1103/PhysRevE.83.016107` :arxiv:`1008.3926`
    """

    g = Graph()

    if (type(block_membership) is types.FunctionType or
        type(block_membership) is types.LambdaType):
        btype = block_type
        bm = []
        if len(inspect.getfullargspec(block_membership)[0]) == 0:
            for i in range(N):
                bm.append(block_membership())
        else:
            for i in range(N):
                bm.append(block_membership(i))
        block_membership = bm
    elif block_membership is not None:
        btype = _gt_type(block_membership[0])

    if len(inspect.getfullargspec(deg_sampler)[0]) > 0:
        if block_membership is not None:
            sampler = lambda i: deg_sampler(i, block_membership[i])
        else:
            sampler = deg_sampler
    else:
        sampler = lambda i: deg_sampler()

    if not directed:
        def sampler_wrap(*args):
            k = sampler(*args)
            try:
                return int(k)
            except:
                raise ValueError("degree value not understood: " + str(k))
    else:
        def sampler_wrap(*args):
            k = sampler(*args)
            try:
                return int(k[0]), int(k[1])
            except:
                raise ValueError("(in,out)-degree value pair not understood: " +
                                 str(k))

    libgraph_tool_generation.gen_graph(g._Graph__graph, N, sampler_wrap,
                                       not parallel_edges,
                                       not self_loops, not directed,
                                       _get_rng(), verbose, True)
    g.set_directed(directed)

    if degree_block:
        if btype in ["object", "string"] or "vector" in btype:
            btype = "object"
        elif btype in ["int", "int32_t", "bool"]:
            btype = "vector<int32_t>"
        elif btype in ["long", "int64_t"]:
            btype = "vector<int64_t>"
        elif btype in ["double"]:
            btype = "vector<double>"
        elif btype in ["long double"]:
            btype = "vector<long double>"

    if block_membership is not None:
        bm = g.new_vertex_property(btype)
        if btype in ["object", "string"] or "vector" in btype:
            for v in g.vertices():
                if not degree_block:
                    bm[v] = block_membership[int(v)]
                else:
                    if g.is_directed():
                        bm[v] = (block_membership[int(v)], v.in_degree(),
                                 v.out_degree())
                    else:
                        bm[v] = (block_membership[int(v)], v.out_degree())
        else:
            try:
                bm.a = block_membership
            except ValueError:
                bm = g.new_vertex_property("object")
                for v in g.vertices():
                    bm[v] = block_membership[int(v)]
    else:
        bm = None

    if random:
        g.set_fast_edge_removal(True)
        random_rewire(g, parallel_edges=parallel_edges,
                      self_loops=self_loops, verbose=verbose,
                      block_membership=bm, **kwargs)
        g.set_fast_edge_removal(False)

    if bm is None:
        return g
    else:
        return g, bm


@_limit_args({"model": ["erdos", "configuration", "constrained-configuration",
                        "probabilistic-configuration", "blockmodel-degree",
                        "blockmodel", "blockmodel-micro"]})
def random_rewire(g, model="configuration", n_iter=10, edge_sweep=True,
                  parallel_edges=False, self_loops=False, configuration=True,
                  edge_probs=None, block_membership=None, cache_probs=True,
                  persist=False, pin=None, ret_fail=False, verbose=False):
    r"""Shuffle the graph in-place, following a variety of possible statistical
    models, chosen via the parameter ``model``.


    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be shuffled. The graph will be modified.
    model : string (optional, default: ``"configuration"``)
        The following statistical models can be chosen, which determine how the
        edges are rewired.

        ``erdos``
           The edges will be rewired entirely randomly, and the resulting graph
           will correspond to the :math:`G(N,E)` Erdős–Rényi model.
        ``configuration``
           The edges will be rewired randomly, but the degree sequence of the
           graph will remain unmodified.
        ``constrained-configuration``
           The edges will be rewired randomly, but both the degree sequence of
           the graph and the *vertex-vertex (in,out)-degree correlations* will
           remain exactly preserved. If the ``block_membership`` parameter is
           passed, the block variables at the endpoints of the edges will be
           preserved, instead of the degree-degree correlation.
        ``probabilistic-configuration``
           This is similar to ``constrained-configuration``, but the
           vertex-vertex correlations are not preserved, but are instead sampled
           from an arbitrary degree-based probabilistic model specified via the
           ``edge_probs`` parameter. The degree-sequence is preserved.
        ``blockmodel-degree``
           This is just like ``probabilistic-configuration``, but the values
           passed to the ``edge_probs`` function will correspond to the block
           membership values specified by the ``block_membership`` parameter.
        ``blockmodel``
           This is just like ``blockmodel-degree``, but the degree sequence *is
           not* preserved during rewiring.
        ``blockmodel-micro``
           This is like ``blockmodel``, but the exact number of edges between
           groups is preserved as well.
    n_iter : int (optional, default: ``10``)
        Number of iterations. If ``edge_sweep == True``, each iteration
        corresponds to an entire "sweep" over all edges. Otherwise this
        corresponds to the total number of edges which are randomly chosen for a
        swap attempt (which may repeat).
    edge_sweep : bool (optional, default: ``True``)
        If ``True``, each iteration will perform an entire "sweep" over the
        edges, where each edge is visited once in random order, and a edge swap
        is attempted.
    parallel_edges : bool (optional, default: ``False``)
        If ``True``, parallel edges are allowed.
    self_loops : bool (optional, default: ``False``)
        If ``True``, self-loops are allowed.
    configuration : bool (optional, default: ``True``)
        If ``True``, graphs are sampled from the corresponding maximum-entropy
        ensemble of configurations (i.e. distinguishable half-edge pairings),
        otherwise they are sampled from the maximum-entropy ensemble of graphs
        (i.e. indistinguishable half-edge pairings). The distinction is only
        relevant if parallel edges are allowed.
    edge_probs : function or sequence of triples (optional, default: ``None``)
        A function which determines the edge probabilities in the graph. In
        general it should have the following signature:

        .. code::

            def prob(r, s):
                ...
                return p

        where the return value should be a non-negative scalar.

        Alternatively, this parameter can be a list of triples of the form ``(r,
        s, p)``, with the same meaning as the ``r``, ``s`` and ``p`` values
        above. If a given ``(r, s)`` combination is not present in this list,
        the corresponding value of ``p`` is assumed to be zero. If the same
        ``(r, s)`` combination appears more than once, their ``p`` values will
        be summed together. This is useful when the correlation matrix is
        sparse, i.e. most entries are zero.

        If ``model == probabilistic-configuration`` the parameters ``r`` and
        ``s`` correspond respectively to the (in, out)-degree pair of the source
        vertex of an edge, and the (in,out)-degree pair of the target of the
        same edge (for undirected graphs, both parameters are scalars
        instead). The value of ``p`` should be a number proportional to the
        probability of such an edge existing in the generated graph.

        If ``model == blockmodel-degree`` or ``model == blockmodel``, the ``r``
        and ``s`` values passed to the function will be the block values of the
        respective vertices, as specified via the ``block_membership``
        parameter. The value of ``p`` should be a number proportional to the
        probability of such an edge existing in the generated graph.
    block_membership : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        If supplied, the graph will be rewired to conform to a blockmodel
        ensemble. The value must be a vertex property map which defines the
        block of each vertex.
    cache_probs : bool (optional, default: ``True``)
        If ``True``, the probabilities returned by the ``edge_probs`` parameter
        will be cached internally. This is crucial for good performance, since
        in this case the supplied python function is called only a few times,
        and not at every attempted edge rewire move. However, in the case were
        the different parameter combinations to the probability function is very
        large, the memory and time requirements to keep the cache may not be
        worthwhile.
    persist : bool (optional, default: ``False``)
        If ``True``, an edge swap which is rejected will be attempted again
        until it succeeds. This may improve the quality of the shuffling for
        some probabilistic models, and should be sufficiently fast for sparse
        graphs, but otherwise it may result in many repeated attempts for
        certain corner-cases in which edges are difficult to swap.
    pin : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge property map which, if provided, specifies which edges are allowed
        to be rewired. Edges for which the property value is ``1`` (or ``True``)
        will be left unmodified in the graph.
    verbose : bool (optional, default: ``False``)
        If ``True``, verbose information is displayed.


    Returns
    -------
    rejection_count : int
        Number of rejected edge moves (due to parallel edges or self-loops, or
        the probabilistic model used).

    See Also
    --------
    random_graph: random graph generation

    Notes
    -----
    This algorithm iterates through all the edges in the network and tries to
    swap its target or source with the target or source of another edge. The
    selected canditate swaps are chosen according to the ``model`` parameter.

    .. note::

        If ``parallel_edges = False``, parallel edges are not placed during
        rewiring. In this case, the returned graph will be a uncorrelated sample
        from the desired ensemble only if ``n_iter`` is sufficiently large. The
        algorithm implements an efficient Markov chain based on edge swaps, with
        a mixing time which depends on the degree distribution and correlations
        desired. If degree probabilistic correlations are provided, the mixing
        time tends to be larger.

        If ``model`` is either "probabilistic-configuration", "blockmodel" or
        "blockmodel-degree", the Markov chain still needs to be mixed, even if
        parallel edges and self-loops are allowed. In this case the Markov chain
        is implemented using the Metropolis-Hastings
        [metropolis-equations-1953]_ [hastings-monte-carlo-1970]_
        acceptance/rejection algorithm. It will eventually converge to the
        desired probabilities for sufficiently large values of ``n_iter``.


    Each edge is tentatively swapped once per iteration, so the overall
    complexity is :math:`O(V + E \times \text{n-iter})`. If ``edge_sweep ==
    False``, the complexity becomes :math:`O(V + E + \text{n-iter})`.

    Examples
    --------

    Some small graphs for visualization.

    .. testcode::
       :hide:

       from numpy.random import random, seed
       from pylab import *
       seed(43)
       gt.seed_rng(42)

    >>> g, pos = gt.triangulation(np.random.random((1000,2)))
    >>> pos = gt.arf_layout(g)
    >>> gt.graph_draw(g, pos=pos, output="rewire_orig.pdf")
    <...>

    >>> ret = gt.random_rewire(g, "constrained-configuration")
    >>> pos = gt.arf_layout(g)
    >>> gt.graph_draw(g, pos=pos, output="rewire_corr.pdf")
    <...>

    >>> ret = gt.random_rewire(g)
    >>> pos = gt.arf_layout(g)
    >>> gt.graph_draw(g, pos=pos, output="rewire_uncorr.pdf")
    <...>

    >>> ret = gt.random_rewire(g, "erdos")
    >>> pos = gt.arf_layout(g)
    >>> gt.graph_draw(g, pos=pos, output="rewire_erdos.pdf")
    <...>

    .. testcleanup::

       conv_png("rewire_orig.pdf")
       conv_png("rewire_corr.pdf")
       conv_png("rewire_uncorr.pdf")
       conv_png("rewire_erdos.pdf")

    Some `ridiculograms <http://www.youtube.com/watch?v=YS-asmU3p_4>`_ :

    .. image:: rewire_orig.png
       :width: 24%
    .. image:: rewire_corr.png
       :width: 24%
    .. image:: rewire_uncorr.png
       :width: 24%
    .. image:: rewire_erdos.png
       :width: 24%

    **From left to right**: Original graph; Shuffled graph, with degree correlations;
    Shuffled graph, without degree correlations; Shuffled graph, with random degrees.

    We can try with larger graphs to get better statistics, as follows.

    >>> def sample_k(max):
    ...     accept = False
    ...     while not accept:
    ...         k = np.random.randint(1,max+1)
    ...         accept = np.random.random() < 1.0/k
    ...     return k
    >>> figure(figsize=(8,3))
    <...>
    >>> g = gt.random_graph(30000, lambda: sample_k(20), model="probabilistic-configuration",
    ...                     edge_probs=lambda i, j: exp(abs(i-j)), directed=False,
    ...                     n_iter=100)
    >>> corr = gt.avg_neighbor_corr(g, "out", "out")
    >>> errorbar(corr[2][:-1], corr[0], yerr=corr[1], fmt="o-", label="Original")
    <...>
    >>> ret = gt.random_rewire(g, "constrained-configuration")
    >>> corr = gt.avg_neighbor_corr(g, "out", "out")
    >>> errorbar(corr[2][:-1], corr[0], yerr=corr[1], fmt="*", label="Correlated")
    <...>
    >>> ret = gt.random_rewire(g)
    >>> corr = gt.avg_neighbor_corr(g, "out", "out")
    >>> errorbar(corr[2][:-1], corr[0], yerr=corr[1], fmt="o-", label="Uncorrelated")
    <...>
    >>> ret = gt.random_rewire(g, "erdos")
    >>> corr = gt.avg_neighbor_corr(g, "out", "out")
    >>> errorbar(corr[2][:-1], corr[0], yerr=corr[1], fmt="o-", label=r"Erd\H{o}s")
    <...>
    >>> xlabel("$k$")
    Text(...)
    >>> ylabel(r"$\left<k_{nn}\right>$")
    Text(...)
    >>> legend(loc='center left', bbox_to_anchor=(1, 0.5))
    <...>
    >>> tight_layout()
    >>> box = gca().get_position()
    >>> gca().set_position([box.x0, box.y0, box.width * 0.7, box.height])
    >>> savefig("shuffled-stats.svg")

    .. figure:: shuffled-stats.*
        :align: center

        Average degree correlations for the different shuffled and non-shuffled
        graphs. The shuffled graph with correlations displays exactly the same
        correlation as the original graph.

    Now let's do it for a directed graph. See
    :func:`~graph_tool.generation.random_graph` for more details.

    >>> def sample_k(max):
    ...     accept = False
    ...     while not accept:
    ...         k = np.random.randint(1,max+1)
    ...         accept = np.random.random() < 1.0/k
    ...     return k
    >>> p = scipy.stats.poisson
    >>> g = gt.random_graph(20000, lambda: (sample_k(19), sample_k(19)),
    ...                     model="probabilistic-configuration",
    ...                     edge_probs=lambda a, b: (p.pmf(a[0], b[1]) * p.pmf(a[1], 20 - b[0])),
    ...                     n_iter=100)
    >>> figure(figsize=(9,3))
    <...>
    >>> corr = gt.avg_neighbor_corr(g, "in", "out")
    >>> errorbar(corr[2][:-1], corr[0], yerr=corr[1], fmt="o-",
    ...          label=r"$\left<\text{o}\right>$ vs i")
    <...>
    >>> corr = gt.avg_neighbor_corr(g, "out", "in")
    >>> errorbar(corr[2][:-1], corr[0], yerr=corr[1], fmt="o-",
    ...          label=r"$\left<\text{i}\right>$ vs o")
    <...>
    >>> ret = gt.random_rewire(g, "constrained-configuration")
    >>> corr = gt.avg_neighbor_corr(g, "in", "out")
    >>> errorbar(corr[2][:-1], corr[0], yerr=corr[1], fmt="o-",
    ...          label=r"$\left<\text{o}\right>$ vs i, corr.")
    <...>
    >>> corr = gt.avg_neighbor_corr(g, "out", "in")
    >>> errorbar(corr[2][:-1], corr[0], yerr=corr[1], fmt="o-",
    ...          label=r"$\left<\text{i}\right>$ vs o, corr.")
    <...>
    >>> ret = gt.random_rewire(g, "configuration")
    >>> corr = gt.avg_neighbor_corr(g, "in", "out")
    >>> errorbar(corr[2][:-1], corr[0], yerr=corr[1], fmt="o-",
    ...          label=r"$\left<\text{o}\right>$ vs i, uncorr.")
    <...>
    >>> corr = gt.avg_neighbor_corr(g, "out", "in")
    >>> errorbar(corr[2][:-1], corr[0], yerr=corr[1], fmt="o-",
    ...          label=r"$\left<\text{i}\right>$ vs o, uncorr.")
    <...>
    >>> legend(loc='center left', bbox_to_anchor=(1, 0.5))
    <...>
    >>> xlabel("Source degree")
    Text(...)
    >>> ylabel("Average target degree")
    Text(...)
    >>> tight_layout()
    >>> box = gca().get_position()
    >>> gca().set_position([box.x0, box.y0, box.width * 0.55, box.height])
    >>> savefig("shuffled-deg-corr-dir.svg")

    .. figure:: shuffled-deg-corr-dir.*
        :align: center

        Average degree correlations for the different shuffled and non-shuffled
        directed graphs. The shuffled graph with correlations displays exactly
        the same correlation as the original graph.

    References
    ----------
    .. [metropolis-equations-1953]  Metropolis, N.; Rosenbluth, A.W.;
       Rosenbluth, M.N.; Teller, A.H.; Teller, E. "Equations of State
       Calculations by Fast Computing Machines". Journal of Chemical Physics 21
       (6): 1087-1092 (1953). :doi:`10.1063/1.1699114`
    .. [hastings-monte-carlo-1970] Hastings, W.K. "Monte Carlo Sampling Methods
       Using Markov Chains and Their Applications". Biometrika 57 (1): 97-109 (1970).
       :doi:`10.1093/biomet/57.1.97`
    .. [holland-stochastic-1983] Paul W. Holland, Kathryn Blackmond Laskey, and
       Samuel Leinhardt, "Stochastic blockmodels: First steps," Social Networks
       5, no. 2: 109-13 (1983) :doi:`10.1016/0378-8733(83)90021-7`
    .. [karrer-stochastic-2011] Brian Karrer and M. E. J. Newman, "Stochastic
       blockmodels and community structure in networks," Physical Review E 83,
       no. 1: 016107 (2011) :doi:`10.1103/PhysRevE.83.016107` :arxiv:`1008.3926`

    """

    if (edge_probs is not None and not g.is_directed()) and "blockmodel" not in model:
        corr = lambda i, j: edge_probs(i[1], j[1])
    else:
        corr = edge_probs

    if model not in ["probabilistic-configuration", "blockmodel",
                     "blockmodel-degree"]:
        g = GraphView(g, reversed=False)
    elif edge_probs is None:
        raise ValueError("A function must be supplied as the 'edge_probs' parameter")

    traditional = True
    micro = False
    if model == "blockmodel-degree":
        model = "blockmodel"
        traditional = False
    if model == "blockmodel-micro":
        model = "blockmodel"
        micro = True

    if pin is None:
        pin = g.new_edge_property("bool")

    if pin.value_type() != "bool":
        pin = pin.copy(value_type="bool")

    fast = g.get_fast_edge_removal()
    if not fast:
        g.set_fast_edge_removal(True)
    pcount = libgraph_tool_generation.random_rewire(g._Graph__graph,
                                                    model,
                                                    n_iter, not edge_sweep,
                                                    self_loops, parallel_edges,
                                                    configuration, traditional,
                                                    micro, persist, corr,
                                                    _prop("e", g, pin),
                                                    _prop("v", g, block_membership),
                                                    cache_probs,
                                                    _get_rng(), verbose)
    if not fast:
        g.set_fast_edge_removal(False)
    return pcount

def generate_sbm(b, probs, out_degs=None, in_degs=None, directed=False,
                 micro_ers=False, micro_degs=False):
    r"""Generate a random graph by sampling from the Poisson or microcanonical
    stochastic block model.

    Parameters
    ----------
    b : iterable or :class:`numpy.ndarray`
        Group membership for each node.
    probs : two-dimensional :class:`numpy.ndarray` or :class:`scipy.sparse.spmatrix`
        Matrix with edge propensities between groups. The value ``probs[r,s]``
        corresponds to the average number of edges between groups ``r`` and
        ``s`` (or twice the average number if ``r == s`` and the graph is
        undirected).
    out_degs : iterable or :class:`numpy.ndarray` (optional, default: ``None``)
        Out-degree propensity for each node. If not provided, a constant value
        will be used. Note that the values will be normalized inside each group,
        if they are not already so.
    in_degs : iterable or :class:`numpy.ndarray` (optional, default: ``None``)
        In-degree propensity for each node. If not provided, a constant value
        will be used. Note that the values will be normalized inside each group,
        if they are not already so.
    directed : ``bool`` (optional, default: ``False``)
        Whether the graph is directed.
    micro_ers : ``bool`` (optional, default: ``False``)
        If true, the `microcanonical` version of the model will be evoked, where
        the numbers of edges between groups will be given `exactly` by the
        parameter ``probs``, and this will not fluctuate between samples.
    micro_degs : ``bool`` (optional, default: ``False``)
        If true, the `microcanonical` version of the degree-corrected model will
        be evoked, where the degrees of nodes will be given `exactly` by the
        parameters ``out_degs`` and ``in_degs``, and they will not fluctuate
        between samples. (If ``micro_degs == True`` it implies ``micro_ers ==
        True``.)


    Returns
    -------
    g : :class:`~graph_tool.Graph`
        The generated graph.

    See Also
    --------
    random_graph: random graph generation

    Notes
    -----

    The algorithm generates multigraphs with self-loops, according to the
    Poisson degree-corrected stochastic block model (SBM)
    [karrer-stochastic-2011]_, which includes the traditional SBM as a special
    case.

    The multigraphs are generated with probability

    .. math::

        P({\boldsymbol A}|{\boldsymbol \theta},{\boldsymbol \lambda},{\boldsymbol b})
            = \prod_{i<j}\frac{e^{-\lambda_{b_ib_j}\theta_i\theta_j}(\lambda_{b_ib_j}\theta_i\theta_j)^{A_{ij}}}{A_{ij}!}
              \times\prod_i\frac{e^{-\lambda_{b_ib_i}\theta_i^2/2}(\lambda_{b_ib_i}\theta_i^2/2)^{A_{ij}/2}}{(A_{ij}/2)!},

    where :math:`\lambda_{rs}` is the edge propensity between groups :math:`r`
    and :math:`s`, and :math:`\theta_i` is the propensity of node i to receive
    edges, which is proportional to its expected degree. Note that in the
    algorithm it is assumed that the node propensities are normalized for each
    group,

    .. math::

        \sum_i\theta_i\delta_{b_i,r} = 1,

    such that the value :math:`\lambda_{rs}` will correspond to the average
    number of edges between groups :math:`r` and :math:`s` (or twice that if
    :math:`r = s`). If the supplied values of :math:`\theta_i` are not
    normalized as above, they will be normalized prior to the generation of the
    graph.

    For directed graphs, the probability is analogous, with :math:`\lambda_{rs}`
    being in general asymmetric:

    .. math::

        P({\boldsymbol A}|{\boldsymbol \theta},{\boldsymbol \lambda},{\boldsymbol b})
            = \prod_{ij}\frac{e^{-\lambda_{b_ib_j}\theta^+_i\theta^-_j}(\lambda_{b_ib_j}\theta^+_i\theta^-_j)^{A_{ij}}}{A_{ij}!}.

    Again, the same normalization is assumed:

    .. math::

        \sum_i\theta_i^+\delta_{b_i,r} = \sum_i\theta_i^-\delta_{b_i,r} = 1,

    such that the value :math:`\lambda_{rs}` will correspond to the average
    number of directed edges between groups :math:`r` and :math:`s`.

    The traditional (i.e. non-degree-corrected) SBM is recovered from the above
    model by setting :math:`\theta_i=1/n_{b_i}` (or
    :math:`\theta^+_i=\theta^-_i=1/n_{b_i}` in the directed case), which is done
    automatically if ``out_degs`` and ``in_degs`` are not specified.

    In case the parameter ``micro_degs == True`` is passed, a `microcanical
    <https://en.wikipedia.org/wiki/Microcanonical_ensemble>`_ model is used
    instead, where both the number of edges between groups as well as the
    degrees of the nodes are preserved `exactly`, instead of only on expectation
    [peixoto-nonparametric-2017]_. In this case, the parameters are interpreted
    as :math:`{\boldsymbol\lambda}\equiv{\boldsymbol e}` and
    :math:`{\boldsymbol\theta}\equiv{\boldsymbol k}`, where :math:`e_{rs}` is
    the number of edges between groups :math:`r` and :math:`s` (or twice that if
    :math:`r=s` in the undirected case), and :math:`k_i` is the degree of node
    :math:`i`. This model is a generalization of the configuration model, where
    multigraphs are sampled with probability

    .. math::

        P({\boldsymbol A}|{\boldsymbol k},{\boldsymbol e},{\boldsymbol b}) =
        \frac{\prod_{r<s}e_{rs}!\prod_re_{rr}!!\prod_ik_i!}{\prod_re_r!\prod_{i<j}A_{ij}!\prod_iA_{ii}!!}.

    and in the directed case with probability

    .. math::

        P({\boldsymbol A}|{\boldsymbol k}^+,{\boldsymbol k}^-,{\boldsymbol e},{\boldsymbol b}) =
        \frac{\prod_{rs}e_{rs}!\prod_ik^+_i!k^-_i!}{\prod_re^+_r!e^-_r!\prod_{ij}A_{ij}!}.

    where :math:`e^+_r = \sum_se_{rs}`, :math:`e^-_r = \sum_se_{sr}`,
    :math:`k^+_i = \sum_jA_{ij}` and :math:`k^-_i = \sum_jA_{ji}`.

    In the non-degree-corrected case, if ``micro_ers == True``, the
    microcanonical model corresponds to

    .. math::

        P({\boldsymbol A}|{\boldsymbol e},{\boldsymbol b}) =
        \frac{\prod_{r<s}e_{rs}!\prod_re_{rr}!!}{\prod_rn_r^{e_r}\prod_{i<j}A_{ij}!\prod_iA_{ii}!!},

    and in the directed case to

    .. math::

        P({\boldsymbol A}|{\boldsymbol e},{\boldsymbol b}) =
        \frac{\prod_{rs}e_{rs}!}{\prod_rn_r^{e_r^+ + e_r^-}\prod_{ij}A_{ij}!}.

    In every case above, the final graph is generated in time :math:`O(V + E +
    B)`, where :math:`B` is the number of groups.

    Examples
    --------

    >>> g = gt.collection.data["polblogs"]
    >>> g = gt.GraphView(g, vfilt=gt.label_largest_component(g))
    >>> g = gt.Graph(g, prune=True)
    >>> state = gt.minimize_blockmodel_dl(g)
    >>> u = gt.generate_sbm(state.b.a, gt.adjacency(state.get_bg(),
    ...                                             state.get_ers()).T,
    ...                     g.degree_property_map("out").a,
    ...                     g.degree_property_map("in").a, directed=True)
    >>> gt.graph_draw(g, g.vp.pos, output="polblogs-sbm.pdf")
    <...>
    >>> gt.graph_draw(u, u.own_property(g.vp.pos), output="polblogs-sbm-generated.pdf")
    <...>

    .. testcleanup::

       conv_png("polblogs-sbm.pdf")
       conv_png("polblogs-sbm-generated.pdf")


    .. image:: polblogs-sbm.png
        :width: 40%
    .. image:: polblogs-sbm-generated.png
        :width: 40%

    *Left:* Political blogs network. *Right:* Sample from the degree-corrected
    SBM fitted to the original network.

    References
    ----------
    .. [karrer-stochastic-2011] Brian Karrer and M. E. J. Newman, "Stochastic
       blockmodels and community structure in networks," Physical Review E 83,
       no. 1: 016107 (2011) :doi:`10.1103/PhysRevE.83.016107` :arxiv:`1008.3926`
    .. [peixoto-nonparametric-2017] Tiago P. Peixoto, "Nonparametric Bayesian
       inference of the microcanonical stochastic block model", Phys. Rev. E 95
       012317 (2017). :doi:`10.1103/PhysRevE.95.012317`, :arxiv:`1610.02703`

    """

    g = Graph()
    g.add_vertex(len(b))
    b = g.new_vp("int", b)

    if micro_degs:
        if (out_degs is not None and
            not numpy.equal(numpy.mod(out_degs, 1), 0).all()):
            raise ValueError("The 'out_degs' parameter must contain only integer values if 'micro_degs' is set to True.")
        if (in_degs is not None and
            not numpy.equal(numpy.mod(in_degs, 1), 0).all()):
            raise ValueError("The 'out_degs' parameter must contain only integer values if 'micro_degs' is set to True.")

    deg_type = "double" if not micro_degs else "int64_t"
    p_type = "double" if not micro_degs else "uint64"
    if not directed:
        if out_degs is None:
            out_degs = in_degs = g.new_vp(deg_type, 1)
        else:
            out_degs = in_degs = g.new_vp(deg_type, out_degs)
    else:
        if out_degs is None:
            out_degs = g.new_vp(deg_type, 1)
        else:
            out_degs = g.new_vp(deg_type, out_degs)
        if in_degs is None:
            in_degs = g.new_vp(deg_type, 1)
        else:
            in_degs = g.new_vp(deg_type, in_degs)

    r, s = probs.nonzero()
    if not directed:
        idx = r <= s
        r = r[idx]
        s = s[idx]
    p = numpy.squeeze(numpy.array(probs[r, s]))

    if len(p.shape) == 0: # B == 1 special case
        p = numpy.array([p])

    if micro_ers:
        if not numpy.equal(numpy.mod(p, 1), 0).all():
            raise ValueError("The 'probs' parameter must contain only integer values if 'micro_ers' is set to True.")

    g.set_directed(directed)

    libgraph_tool_generation.gen_sbm(g._Graph__graph,
                                     _prop("v", g, b),
                                     numpy.asarray(r, dtype="int64"),
                                     numpy.asarray(s, dtype="int64"),
                                     numpy.asarray(p, dtype=p_type),
                                     _prop("v", g, in_degs),
                                     _prop("v", g, out_degs),
                                     micro_ers,
                                     micro_degs,
                                     _get_rng())
    return g

def solve_sbm_fugacities(b, ers, out_degs=None, in_degs=None, multigraph=False,
                         self_loops=False, epsilon=1e-8, iter_solve=True,
                         max_iter=0, min_args={}, root_args={}, verbose=False):
    r"""Obtain SBM fugacities, given expected degrees and edge counts between
    groups.

    Parameters
    ----------
    b : iterable or :class:`numpy.ndarray`
        Group membership for each node.
    ers : two-dimensional :class:`numpy.ndarray` or :class:`scipy.sparse.spmatrix`
        Matrix with expected edge counts between groups. The value ``ers[r,s]``
        corresponds to the average number of edges between groups ``r`` and
        ``s`` (or twice the average number if ``r == s`` and the graph is
        undirected).
    out_degs : iterable or :class:`numpy.ndarray`
        Expected out-degree for each node.
    in_degs : iterable or :class:`numpy.ndarray` (optional, default: ``None``)
        Expected in-degree for each node. If not given, the graph is assumed to
        be undirected.
    multigraph : ``bool`` (optional, default: ``False``)
        Whether parallel edges are allowed.
    self_loops : ``bool`` (optional, default: ``False``)
        Whether self-loops are allowed.
    epsilon : ``float`` (optional, default: ``1e-8``)
        Whether self-loops are allowed.
    iter_solve : ``bool`` (optional, default: ``True``)
        Solve the system by simple iteration, not gradient-based
        root-solving. Relevant only if ``multigraph == False``, otherwise
        `iter_solve = True` is always assumed.
    max_iter : ``int`` (optional, default: ``0``)
        If non-zero, this will limit the maximum number of iterations.
    min_args : ``{}`` (optional, default: ``{}``)
        Options to be passed to :func:`scipy.optimize.minimize`. Only relevant
        if ``iter_solve=False``.
    root_args : ``{}`` (optional, default: ``{}``)
        Options to be passed to :func:`scipy.optimize.root`. Only relevant
        if ``iter_solve=False``.
    verbose : ``bool`` (optional, default: ``False``)
        If ``True``, verbose information will be displayed.

    Returns
    -------
    mrs : :class:`scipy.sparse.spmatrix`
        Edge count fugacities.
    out_theta : :class:`numpy.ndarray`
        Node out-degree fugacities.
    in_theta : :class:`numpy.ndarray`
        Node in-degree fugacities. Only returned if ``in_degs is not None``.

    See Also
    --------
    generate_maxent_sbm: Generate maximum-entropy SBM graphs

    Notes
    -----

    For simple directed graphs, the fugacities obey the following self-consistency equations:

    .. math::

        \theta^+_i &= \frac{k^+_i}{\sum_{j\ne i}\frac{\theta^-_j\mu_{b_i,b_j}}{1+\theta^+_i\theta^-_j\mu_{b_i,b_j}}}\\
        \theta^-_i &= \frac{k^-_i}{\sum_{j\ne i}\frac{\theta^+_j\mu_{b_j,b_i}}{1+\theta^+_i\theta^-_j\mu_{b_j,b_i}}}\\
        \mu_{rs} &= \frac{e_{rs}}{\sum_{i\ne j}\delta_{b_i,r}\delta_{b_j,s}\frac{\theta^+_i\theta^-_j}{1+\theta^+_i\theta^-_j\mu_{r,s}}}

    For directed multigraphs, we have instead:

    .. math::

        \theta^+_i &= \frac{k^+_i}{\sum_{j\ne i}\frac{\theta^-_j\mu_{b_i,b_j}}{1-\theta^+_i\theta^-_j\mu_{b_i,b_j}}}\\
        \theta^-_i &= \frac{k^-_i}{\sum_{j\ne i}\frac{\theta^+_j\mu_{b_j,b_i}}{1-\theta^+_i\theta^-_j\mu_{b_j,b_i}}}\\
        \mu_{rs} &= \frac{e_{rs}}{\sum_{i\ne j}\delta_{b_i,r}\delta_{b_j,s}\frac{\theta^+_i\theta^-_j}{1-\theta^+_i\theta^-_j\mu_{r,s}}}

    For undirected graphs, we have the above equations with
    :math:`\theta^+_i=\theta^-_i=\theta_i`, and :math:`\mu_{rs} = \mu_{sr}`.

    References
    ----------
    .. [peixoto-latent-2020] Tiago P. Peixoto, "Latent Poisson models for
       networks with heterogeneous density", Phys. Rev. E 102 012309 (2020)
       :doi:`10.1103/PhysRevE.102.012309`, :arxiv:`2002.07803`
    """

    b = numpy.asarray(b, dtype="int32")
    out_degs = numpy.asarray(out_degs, dtype="double")
    directed = False
    if in_degs is None:
        in_degs = out_degs
    else:
        in_degs = numpy.asarray(in_degs, dtype="double")
        directed = True

    r, s = ers.nonzero()
    ers = numpy.squeeze(numpy.array(ers[r, s]))
    ers = numpy.asarray(ers, dtype="double")
    r = numpy.asarray(r, dtype="int64")
    s = numpy.asarray(s, dtype="int64")

    if len(ers.shape) == 0: # B == 1 special case
        ers = numpy.array([ers], dtype="double")

    mrs = numpy.zeros(ers.shape)
    in_theta = numpy.zeros(in_degs.shape)
    out_theta = numpy.zeros(out_degs.shape)

    state = libgraph_tool_generation.get_sbm_fugacities(r, s, ers, in_degs,
                                                        out_degs, b, directed,
                                                        multigraph, self_loops)

    if not multigraph and iter_solve:
        def new_x(x_):
            nx = Vector_double(len(x_))
            nx.a = x_
            state.unpack(nx)
            state.norm()
            state.new_x(nx)
            return nx.a.copy()

        state.norm()
        x = Vector_double()
        state.pack(x)
        x = x.a.copy()

        niter = 0
        delta = epsilon + 1
        while delta > epsilon:
            nx = new_x(x)
            delta = abs(nx - x).max()
            if verbose:
                print(delta, x.min(), x.max(), nx.min(), nx.max())
            x = nx

            niter += 1
            if max_iter > 0 and niter >= max_iter:
                break
    else:
        def f(x_):
            x = Vector_double(len(x_))
            if multigraph:
                x.a = (numpy.tanh(x_) + 1)/2
            else:
                x.a = numpy.exp(x_)
            state.unpack(x)
            fval = state.get_f()
            return -fval
        def diff(x_):
            x = Vector_double(len(x_))
            if multigraph:
                x.a = (numpy.tanh(x_) + 1)/2
            else:
                x.a = numpy.exp(x_)
            state.unpack(x)
            d = Vector_double(len(x_))
            state.get_diff(d)
            if multigraph:
                return -d.a * (1-numpy.tanh(x_)**2)/2
            else:
                return -d.a * x.a


        if not multigraph:
            state.norm()
        x = Vector_double()
        state.pack(x)
        x = x.a.copy()

        if multigraph:
            x[x==1] = 0.999
            x[x==0] = 0.001
            x = numpy.arctanh(2 * x - 1)
        else:
            x[x==0] = 1e-4
            x = numpy.log(x)
            print(x)

        sol = scipy.optimize.minimize(f, x, jac=diff,
                                      **dict(dict(method="L-BFGS-B"),
                                             **min_args))
        x = sol.x

        sol = scipy.optimize.root(diff, x, **dict(dict(method="krylov"),
                                                  **root_args))
        x = sol.x

        if multigraph:
            x = (numpy.tanh(x) + 1)/2
        else:
            x = numpy.exp(x)

    x_ = Vector_double()
    state.pack(x_)
    x_.a = x
    state.unpack(x_)

    state.export_args(r, s, mrs, in_degs, out_degs, in_theta, out_theta, b)

    mrs = scipy.sparse.coo_matrix((mrs, (r, s)))
    mrs = mrs.tocsr()

    if directed:
        return mrs, out_theta, in_theta
    else:
        return mrs, out_theta

def generate_maxent_sbm(b, mrs, out_theta, in_theta=None, directed=False,
                        multigraph=False, self_loops=False):
    r"""Generate a random graph by sampling from the maximum-entropy "canonical"
    stochastic block model.

    Parameters
    ----------
    b : iterable or :class:`numpy.ndarray`
        Group membership for each node.
    mrs : two-dimensional :class:`numpy.ndarray` or :class:`scipy.sparse.spmatrix`
        Matrix with edge fugacities between groups.
    out_theta : iterable or :class:`numpy.ndarray`
        Out-degree fugacities for each node.
    in_theta : iterable or :class:`numpy.ndarray` (optional, default: ``None``)
        In-degree fugacities for each node. If not provided, will be identical to ``out_theta``.
    directed : ``bool`` (optional, default: ``False``)
        Whether the graph is directed.
    multigraph : ``bool`` (optional, default: ``False``)
        Whether parallel edges are allowed.
    self_loops : ``bool`` (optional, default: ``False``)
        Whether self-loops are allowed.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        The generated graph.

    See Also
    --------
    solve_sbm_fugacities: Obtain SBM fugacities, given expected degrees and block constraints.
    generate_sbm: Generate samples from the Poisson SBM

    Notes
    -----

    The algorithm generates simple or multigraphs according to the
    degree-corrected maximum-entropy stochastic block model (SBM)
    [peixoto-latent-2020]_, which includes the non-degree-corrected SBM as a
    special case.

    The simple graphs are generated with probability:

    .. math::

        P({\boldsymbol A}|{\boldsymbol \theta},{\boldsymbol \mu},{\boldsymbol b})
            = \prod_{i<j} \frac{\left(\theta_i\theta_j\mu_{b_i,b_j}\right)^{A_{ij}}}{1+\theta_i\theta_j\mu_{b_i,b_j}},

    and the multigraphs with probability:

    .. math::

        P({\boldsymbol A}|{\boldsymbol \theta},{\boldsymbol \mu},{\boldsymbol b})
            = \prod_{i<j} \left(\theta_i\theta_j\mu_{b_i,b_j}\right)^{A_{ij}}(1-\theta_i\theta_j\mu_{b_i,b_j}).

    In the above, :math:`\mu_{rs}` is the edge fugacity between groups :math:`r`
    and :math:`s`, and :math:`\theta_i` is the edge fugacity of node i.

    For directed graphs, the probabilities are analogous, i.e.

    .. math::

        P({\boldsymbol A}|{\boldsymbol \theta}^+,{\boldsymbol \theta}^-,{\boldsymbol \mu},{\boldsymbol b})
            &= \prod_{i\ne j} \frac{\left(\theta_i^+\theta_j^-\mu_{b_i,b_j}\right)^{A_{ij}}}{1+\theta_i^+\theta_j^-\mu_{b_i,b_j}} & \quad\text{(simple graphs)},\\
        P({\boldsymbol A}|{\boldsymbol \theta}^+,{\boldsymbol \theta}^-,{\boldsymbol \mu},{\boldsymbol b})
            &= \prod_{i\ne j} \left(\theta_i^+\theta_j^-\mu_{b_i,b_j}\right)^{A_{ij}}(1-\theta_i^+\theta_j^-\mu_{b_i,b_j}) & \quad\text{(multigraphs)}.

    References
    ----------
    .. [peixoto-latent-2020] Tiago P. Peixoto, "Latent Poisson models for
       networks with heterogeneous density", :doi:`10.1103/PhysRevE.102.012309`, :arxiv:`2002.07803`

    Examples
    --------

    .. doctest:: max_ent_sbm

       >>> g = gt.collection.data["polblogs"]
       >>> g = gt.GraphView(g, vfilt=gt.label_largest_component(g))
       >>> g = gt.Graph(g, prune=True)
       >>> gt.remove_self_loops(g)
       >>> gt.remove_parallel_edges(g)
       >>> state = gt.minimize_blockmodel_dl(g)
       >>> ers = gt.adjacency(state.get_bg(), state.get_ers()).T
       >>> out_degs = g.degree_property_map("out").a
       >>> in_degs = g.degree_property_map("in").a
       >>> mrs, theta_out, theta_in = gt.solve_sbm_fugacities(state.b.a, ers, out_degs, in_degs)
       >>> u = gt.generate_maxent_sbm(state.b.a, mrs, theta_out, theta_in, directed=True)
       >>> gt.graph_draw(g, g.vp.pos, output="polblogs-maxent-sbm.pdf")
       <...>
       >>> gt.graph_draw(u, u.own_property(g.vp.pos), output="polblogs-maxent-sbm-generated.pdf")
       <...>

    .. testcleanup:: max_ent_sbm

       conv_png("polblogs-maxent-sbm.pdf")
       conv_png("polblogs-maxent-sbm-generated.pdf")

    .. image:: polblogs-maxent-sbm.png
        :width: 40%
    .. image:: polblogs-maxent-sbm-generated.png
        :width: 40%

    *Left:* Political blogs network. *Right:* Sample from the maximum-entropy
    degree-corrected SBM fitted to the original network.

    """

    g = Graph(directed=directed)
    g.add_vertex(len(b))
    b = g.new_vp("int", vals=b)
    out_theta = g.new_vp("double", vals=out_theta)
    if in_theta is not None:
        in_theta = g.new_vp("double", vals=in_theta)
    else:
        in_theta = out_theta

    r, s = mrs.nonzero()
    if not directed:
        idx = r <= s
        r = r[idx]
        s = s[idx]
    mrs = numpy.squeeze(numpy.array(mrs[r, s]))
    mrs = numpy.asarray(mrs, dtype="double")
    r = numpy.asarray(r, dtype="int64")
    s = numpy.asarray(s, dtype="int64")

    if len(mrs.shape) == 0: # B == 1 special case
        mrs = numpy.array([mrs], dtype="double")

    libgraph_tool_generation.\
        gen_maxent_sbm(g._Graph__graph, _prop("v", g, b), r, s, mrs,
                       _prop("v", g, in_theta), _prop("v", g, out_theta),
                       multigraph, self_loops, _get_rng())
    return g


def add_random_edges(g, M, parallel=False, self_loops=False, weight=None):
    r"""Add new edges to a graph, chosen uniformly at random.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modified.
    M : ``int``
        Number of edges to be added.
    parallel : ``bool`` (optional, default: ``False``)
        Wheter to allow parallel edges to be added.
    self_loops : ``bool`` (optional, default: ``False``)
        Wheter to allow self_loops to be added.
    weight : :class:`~graph_tool.EdgePropertyMap`, optional (default: ``None``)
        Integer edge multiplicities. If supplied, this will be incremented for
        edges already in the graph, instead of new edges being added.

    See Also
    --------
    remove_random_edges: remove random edges to the graph

    Notes
    -----
    If the graph is not being filtered, this algorithm runs in time :math:`O(M)`
    if ``parallel == True`` or :math:`O(M\left<k\right>)` if ``parallel ==
    False``, where :math:`\left<k\right>` is the average degree of the graph.

    For filtered graphs, this algorithm runs in time :math:`O(M + E)` if
    ``parallel == True`` or :math:`O(M\left<k\right> + E)` if ``parallel ==
    False``, where :math:`E` is the number of edges in the graph.

    Examples
    --------
    Generating a Newman–Watts–Strogatz small-world graph:

    >>> g = gt.circular_graph(100)
    >>> gt.add_random_edges(g, 30)

    """

    vfilt = g.get_vertex_filter()[0]
    filtered = vfilt is not None and vfilt.a.sum() < len(vfilt.a)

    libgraph_tool_generation.\
        add_random_edges(g._Graph__graph, M, parallel, self_loops, filtered,
                         _prop("e", g, weight), _get_rng())

def remove_random_edges(g, M, weight=None, counts=True):
    r"""Remove edges from the graph, chosen uniformly at random.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modified.
    M : ``int``
        Number of edges to be removed.
    weight : :class:`~graph_tool.EdgePropertyMap`, optional (default: ``None``)
        Integer edge multipliciites or edge removal probabilities. If supplied,
        and ``counts == True`` this will be decremented for edges removed.
    counts : ``bool``, optional (default: ``True``)
        If ``True``, the values given by ``weight`` are assumed to be integer
        edge multiplicities. Otherwise, they will be only considered to be
        proportional to the probability of an edge being removed.

    See Also
    --------
    add_random_edges: add random edges to the graph

    Notes
    -----

    This algorithm runs in time :math:`O(E\left<k\right>)` if
    ``weight is None``, otherwise :math:`O(E + \left<k\right>M\log E)`.

    .. note::

       The complexity can be improved to :math:`O(E)` and :math:`O(E + M\log
       E)`, respectively, if fast edge removal is activated via
       :meth:`~graph_tool.Graph.set_fast_edge_removal` prior to running this
       function.

    Examples
    --------
    .. testcode::
       :hide:

       gt.seed_rng(42)

    >>> g = gt.lattice([100, 100])
    >>> gt.remove_random_edges(g, 10000)
    >>> gt.label_components(g)[1].max()
    3371
    """

    libgraph_tool_generation.\
        remove_random_edges(g._Graph__graph, M, _prop("e", g, weight), counts,
                            _get_rng())

@_parallel
def generate_knn(points, k, dist=None, pairs=False, exact=False, r=.5,
                 max_rk=None, epsilon=.001, c_stop=False, max_iter=0,
                 directed=False, verbose=False):
    r"""Generate a graph of k-nearest neighbors (or pairs) from a set of
    multidimensional points.

    Parameters
    ----------
    points : iterable of lists (or :class:`numpy.ndarray`) of dimension :math:`N\times D` or ``int``
        Points of dimension :math:`D` to be considered. If the parameter `dist`
        is passed, this should be just an `int` containing the number of points.
    k : ``int``
        Number of nearest neighbors per node (or number of pairs if ``pairs is True``).
    dist : function (optional, default: ``None``)
        If given, this should be a function that returns the distance between
        two points. The arguments of this function should just be two integers,
        corresponding to the vertex index. In this case the value of ``points``
        should just be the total number of points. If ``dist is None``, then the
        L2-norm (Euclidean distance) is used.
    pairs : ``bool`` (optional, default: ``False``)
        If ``True``, the ``k`` closest pairs of nodes will be returned, otherwise
        the ``k`` nearest neighbors for every edge is returned.
    exact : ``bool`` (optional, default: ``False``)
        If ``False``, an fast approximation will be used, otherwise an exact but
        slow algorithm will be used.
    r : ``float`` (optional, default: ``.5``)
        If ``exact is False``, this specifies the fraction of randomly chosen
        neighbors that are used for the search.
    max_rk : ``int`` (optional, default: ``None``)
        If provided and ``exact is False``, this specifies the maximum number of
        randomly chosen out neighbors to consider during each iteration. A value
        of ``None`` implies that all out neighbors are considered.
    epsilon : ``float`` (optional, default: ``.001``)
        If ``exact is False`` and ``c_stop is False``, this determines the
        convergence criterion used by the algorithm. When the fraction of
        updated neighbors drops below this value, the algorithm stops.
    c_stop : ``bool`` (optional, default: ``False``)
        If ``True``, an alternative stopping criterion will be used: The
        iteration ends when the global clustering coefficient of the undirected
        KNN graph stopped increasing. In this case, the paramter ``epsilon`` is
        ignored.
    max_iter : ``int`` (optional, default: ``0``)
        If ``exact is False``, this determines the maximum number of iterations
        allowed. A value of ``0`` means that no limit is imposed.
    directed : ``bool`` (optional, default: ``False``)
        If ``True`` a directed version of the graph will be returned, otherwise
        the graph is undirected.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        The k-nearest neighbors graph.
    w : :class:`~graph_tool.EdgePropertyMap`
        Edge property map with the computed distances.

    Notes
    -----

    The approximate version of this algorithm is based on
    [dong-efficient-2011]_, and has a (conjectured) run-time of
    :math:`O(k^2N\log N)`, where :math:`N` is the number of points. The exact
    version has a complexity of :math:`O(N^2)`.

    If ``pairs is True``, the :math:`k` closest pairs are found from the nearest
    neighbors problem as described in [lenhof-k-closest]_, which has a
    complexity upper bounded by

    @parallel@

    References
    ----------
    .. [dong-efficient-2011] Wei Dong, Charikar Moses, and Kai Li, "Efficient
       k-nearest neighbor graph construction for generic similarity measures",
       In Proceedings of the 20th international conference on World wide web
       (WWW '11). Association for Computing Machinery, New York, NY, USA,
       577–586, (2011) :doi:`https://doi.org/10.1145/1963405.1963487`
    .. [lenhof-k-closest] HP Lenhof, M Smid, "The k closest pairs problem",
       https://people.scs.carleton.ca/~michiel/k-closestnote.pdf


    Examples
    --------

    >>> points = np.random.random((1000, 10))
    >>> g, w = gt.generate_knn(points, k=5)

    """

    if dist is not None:
        N = points
        points = dist
    else:
        points = numpy.asarray(points, dtype="float")
        N = points.shape[0]

    if max_rk is None:
        max_rk = N

    g = Graph(N, fast_edge_removal=True)
    w = g.new_ep("double")

    if exact:
        if pairs:
            libgraph_tool_generation.gen_k_nearest_exact(g._Graph__graph,
                                                         points, k,
                                                         _prop("e", g, w),
                                                         directed)
        else:
            libgraph_tool_generation.gen_knn_exact(g._Graph__graph, points, k,
                                                   _prop("e", g, w))
    else:
        if pairs:
            libgraph_tool_generation.gen_k_nearest(g._Graph__graph, points, k,
                                                   r, max_rk, epsilon, c_stop,
                                                   max_iter, _prop("e", g, w),
                                                   directed, verbose,
                                                   _get_rng())
        else:
            libgraph_tool_generation.gen_knn(g._Graph__graph, points, k, r,
                                             max_rk, epsilon, c_stop, max_iter,
                                             _prop("e", g, w), verbose,
                                             _get_rng())

    if not directed:
        g.set_directed(False)
        remove_parallel_edges(g)

    return g, w

def generate_triadic_closure(g, t, probs=True, curr=None, ego=None):
    r"""Closes open triads in a graph, according to an ego-based process.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modified.
    t : :class:`~graph_tool.VertexPropertyMap` or scalar
        Vertex property map (or scalar value) with the ego closure propensities
        for every node.
    probs : ``boolean`` (optional, default: ``False``)
        If ``True``, the values of ``t`` will be interpreted as the independent
        probability of connecting two neighbors of the respective
        vertex. Otherwise, it will determine the integer number of pairs of
        neighbors that will be closed.
    curr : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        If given, this should be a boolean-valued edge property map, such that
        triads will only be closed if they contain at least one edge marged with
        the value ``True``.
    ego : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        If given, this should be an integer-valued edge property map, containing
        the ego vertex for each closed triad, which will be updated with the new
        generation.

    Returns
    -------
    ego : :class:`~graph_tool.EdgePropertyMap`
        Integer-valued edge property map, containing the ego vertex for each
        closed triad.

    Notes
    -----

    This algorithm [peixoto-disentangling-2022]_ consist in, for each node
    ``u``, connecting all its neighbors with probability given by ``t[u]``. In
    case ``probs == False``, then ``t[u]`` indicates the number of random pairs
    of neighbors of ``u`` that are connected. This algorithm may generate
    parallel edges.

    This algorithm has a complexity of :math:`O(N\left<k^2\right>)`, where
    :math:`\left<k^2\right>` is the second moment of the degree distribution.

    References
    ----------
    .. [peixoto-disentangling-2022] Tiago P. Peixoto, "Disentangling homophily,
       community structure and triadic closure in networks", Phys. Rev. X 12,
       011004 (2022), :doi:`10.1103/PhysRevX.12.011004`, :arxiv:`2101.02510`

    Examples
    --------

    >>> g = gt.collection.data["karate"].copy()
    >>> gt.generate_triadic_closure(g, .5)
    <...>
    >>> gt.graph_draw(g, g.vp.pos, output="karate-triadic.png")
    <...>

    .. figure:: karate-triadic.*
       :align: center
       :width: 40%

       Karate club network with added random triadic closure edges.

    """
    if not isinstance(t, VertexPropertyMap):
        t = g.new_vp("double" if probs else "int64_t", val=t)
    _check_prop_scalar(t, name="t")
    if curr is None:
        curr = g.new_ep("bool", val=True)
    if curr.value_type() != "bool":
        curr = curr.copy("bool")
    if ego is None:
        ego = g.new_ep("int64_t", val=-1)
    if ego.value_type() != "int64_t":
        ego = ego.copy("int64_t")
    libgraph_tool_generation.gen_triadic_closure(g._Graph__graph,
                                                 _prop("e", g, curr),
                                                 _prop("e", g, ego),
                                                 _prop("v", g, t),
                                                 probs, _get_rng())
    return ego

def predecessor_tree(g, pred_map):
    """Return a graph from a list of predecessors given by the ``pred_map`` vertex property."""

    _check_prop_scalar(pred_map, "pred_map")
    pg = Graph()
    libgraph_tool_generation.predecessor_graph(g._Graph__graph,
                                               pg._Graph__graph,
                                               _prop("v", g, pred_map))
    return pg


def line_graph(g):
    """Return the line graph of the given graph `g`.

    Notes
    -----
    Given an undirected graph G, its line graph L(G) is a graph such that:

        * Each vertex of L(G) represents an edge of G; and
        * Two vertices of L(G) are adjacent if and only if their corresponding
          edges share a common endpoint ("are adjacent") in G.

    For a directed graph, the second criterion becomes:

        * Two vertices representing directed edges from u to v and from w to x
          in G are connected by an edge from uv to wx in the line digraph when v
          = w.


    Examples
    --------

    >>> g = gt.collection.data["lesmis"]
    >>> lg, vmap = gt.line_graph(g)
    >>> pos = gt.graph_draw(lg, output="lesmis-lg.pdf")

    .. testcleanup::

       conv_png("lesmis-lg.pdf")

    .. figure:: lesmis-lg.png
       :align: center
       :width: 40%

       Line graph of the coappearance of characters in Victor Hugo's novel "Les
       Misérables".

    References
    ----------
    .. [line-wiki] http://en.wikipedia.org/wiki/Line_graph

    """
    lg = Graph(directed=g.is_directed())

    vertex_map = lg.new_vertex_property("int64_t")

    libgraph_tool_generation.line_graph(g._Graph__graph,
                                        lg._Graph__graph,
                                        _prop("v", lg, vertex_map))
    return lg, vertex_map


def graph_union(g1, g2, intersection=None, props=None, include=False,
                internal_props=False):
    """Return the union of graphs ``g1`` and ``g2``, composed of all edges and
    vertices of ``g1`` and ``g2``, without overlap (if ``intersection ==
    None``).

    Parameters
    ----------
    g1 : :class:`~graph_tool.Graph`
       First graph in the union.
    g2 : :class:`~graph_tool.Graph`
       Second graph in the union.
    intersection : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
       Vertex property map owned by `g2` which maps each of its vertices
       to vertex indices belonging to `g1`. Negative values mean no mapping
       exists, and thus both vertices in `g1` and `g2` will be present in the
       union graph.
    props : list of tuples of :class:`~graph_tool.PropertyMap` (optional, default: ``None``)
       Each element in this list must be a tuple of two PropertyMap objects. The
       first element must be a property of `g1`, and the second of `g2`. If either
       value is ``None``, an empty map is created. The values of the property
       maps are propagated into the union graph, and returned.
    include : bool (optional, default: ``False``)
       If ``True``, graph `g2` is inserted in-place into `g1`, which is
       modified. If ``False``, a new graph is created, and both graphs remain
       unmodified.
    internal_props : bool (optional, default: ``False``)
       If ``True``, all internal property maps are propagated, in addition
       to ``props``.

    Returns
    -------
    ug : :class:`~graph_tool.Graph`
        The union graph
    props : list of :class:`~graph_tool.PropertyMap` objects
        List of propagated properties.  This is only returned if `props` is not
        empty.

    Examples
    --------

    .. testcode::
       :hide:

       from numpy.random import random, seed
       from pylab import *
       seed(42)
       gt.seed_rng(42)

    >>> g = gt.triangulation(random((300,2)))[0]
    >>> ug = gt.graph_union(g, g)
    >>> uug = gt.graph_union(g, ug)
    >>> pos = gt.sfdp_layout(g)
    >>> gt.graph_draw(g, pos=pos, adjust_aspect=False, output="graph_original.pdf")
    <...>

    >>> pos = gt.sfdp_layout(ug)
    >>> gt.graph_draw(ug, pos=pos, adjust_aspect=False, output="graph_union.pdf")
    <...>

    >>> pos = gt.sfdp_layout(uug)
    >>> gt.graph_draw(uug, pos=pos, adjust_aspect=False, output="graph_union2.pdf")
    <...>

    .. testcleanup::

       conv_png("graph_original.pdf")
       conv_png("graph_union.pdf")
       conv_png("graph_union2.pdf")

    .. image:: graph_original.png
       :width: 33%
    .. image:: graph_union.png
       :width: 33%
    .. image:: graph_union2.png
       :width: 33%

    """
    pnames = None
    if props is None:
        props = []
    if internal_props:
        pnames = []
        for (k, name), p1 in g1.properties.items():
            if k == 'g':
                continue
            p2 = g2.properties.get((k, name), None)
            props.append((p1, p2))
            pnames.append(name)
        for (k, name), p2 in g2.properties.items():
            if k == 'g' or (k, name) in g1.properties:
                continue
            props.append((None, p2))
            pnames.append(name)
        gprops = [[(name, g1.properties[('g', name)]) for name in g1.graph_properties.keys()],
                  [(name, g2.properties[('g', name)]) for name in g2.graph_properties.keys()]]
    if not include:
        g1 = GraphView(g1, skip_properties=True)
        p1s = []
        for i, (p1, p2) in enumerate(props):
            if p1 is None:
                continue
            if p1.key_type() == "v":
                g1.vp[str(i)] = p1
            elif p1.key_type() == "e":
                g1.ep[str(i)] = p1

        g1 = Graph(g1, prune=True)

        for i, (p1, p2) in enumerate(props):
            if p1 is None:
                continue
            if str(i) in g1.vp:
                props[i] = (g1.vp[str(i)], p2)
                del g1.vp[str(i)]
            else:
                props[i] = (g1.ep[str(i)], p2)
                del g1.ep[str(i)]
    else:
        emask, emask_flip = g1.get_edge_filter()
        emask_flipped = False
        if emask is not None and not emask_flip:
            emask.a = numpy.logical_not(emask.a)
            g1.set_edge_filter(emask, True)
            emask_flipped = True

        vmask, vmask_flip = g1.get_vertex_filter()
        vmask_flipped = False
        if vmask is not None and not vmask_flip:
            vmask.a = numpy.logical_not(vmask.a)
            g1.set_vertex_filter(vmask, True)
            vmask_flipped = True

    if intersection is None:
        intersection = g2.new_vertex_property("int64_t", -1)
    else:
        intersection = intersection.copy("int64_t")

    u1 = GraphView(g1, directed=True, skip_properties=True)
    u2 = GraphView(g2, directed=True, skip_properties=True)

    vmap, emap = libgraph_tool_generation.graph_union(u1._Graph__graph,
                                                      u2._Graph__graph,
                                                      _prop("v", g2,
                                                            intersection))

    if include:
        emask, emask_flip = g1.get_edge_filter()
        if emask is not None and emask_flipped:
            emask.a = numpy.logical_not(emask.a)
            g1.set_edge_filter(emask, False)

        vmask, vmask_flip = g1.get_vertex_filter()
        if vmask is not None and vmask_flipped:
            vmask.a = numpy.logical_not(vmask.a)
            g1.set_vertex_filter(vmask, False)

    n_props = []
    for p1, p2 in props:
        if p1 is None:
            p1 = g1.new_property(p2.key_type(), p2.value_type())
        else:
            p1 = u1.own_property(p1)
        if p2 is None:
            p2 = g2.new_property(p1.key_type(), p1.value_type())
        else:
            p2 = u2.own_property(p2)
        if not include:
            p1 = g1.copy_property(p1)
        if p2.value_type() != p1.value_type():
            p2 = g2.copy_property(p2, value_type=p1.value_type())
        if p1.key_type() == 'v':
            libgraph_tool_generation.\
                  vertex_property_union(u1._Graph__graph, u2._Graph__graph,
                                        vmap, emap,
                                        _prop(p1.key_type(), g1, p1),
                                        _prop(p2.key_type(), g2, p2))
        else:
            libgraph_tool_generation.\
                  edge_property_union(u1._Graph__graph, u2._Graph__graph,
                                      vmap, emap,
                                      _prop(p1.key_type(), g1, p1),
                                      _prop(p2.key_type(), g2, p2))
        n_props.append(p1)

    if pnames is not None:
        for name, p in zip(pnames, n_props):
            g1.properties[(p.key_type(), name)] = p
        if not include:
            for name, p in gprops[0]:
                g1.graph_properties[name] = g1.own_property(p.copy())
        for name, p in gprops[1]:
            if name not in g1.graph_properties:
                g1.graph_properties[name] = g1.own_property(p.copy())
        n_props = []

    if len(n_props) > 0:
        return g1, n_props
    else:
        return g1


@_limit_args({"type": ["simple", "delaunay"]})
def triangulation(points, type="simple", periodic=False):
    r"""
    Generate a 2D or 3D triangulation graph from a given point set.

    Parameters
    ----------
    points : :class:`numpy.ndarray`
        Point set for the triangulation. It may be either a N x d array, where N
        is the number of points, and d is the space dimension (either 2 or 3).
    type : string (optional, default: ``'simple'``)
        Type of triangulation. May be either 'simple' or 'delaunay'.
    periodic : bool (optional, default: ``False``)
        If ``True``, periodic boundary conditions will be used. This is
        parameter is valid only for type="delaunay", and is otherwise ignored.

    Returns
    -------
    triangulation_graph : :class:`~graph_tool.Graph`
        The generated graph.
    pos : :class:`~graph_tool.VertexPropertyMap`
        Vertex property map with the Cartesian coordinates.

    See Also
    --------
    random_graph: random graph generation

    Notes
    -----

    A triangulation [cgal-triang]_ is a division of the convex hull of a point
    set into triangles, using only that set as triangle vertices.

    In simple triangulations (`type="simple"`), the insertion of a point is done
    by locating a face that contains the point, and splitting this face into
    three new faces (the order of insertion is therefore important). If the
    point falls outside the convex hull, the triangulation is restored by
    flips. Apart from the location, insertion takes a time O(1). This bound is
    only an amortized bound for points located outside the convex hull.

    Delaunay triangulations (`type="delaunay"`) have the specific empty sphere
    property, that is, the circumscribing sphere of each cell of such a
    triangulation does not contain any other vertex of the triangulation in its
    interior. These triangulations are uniquely defined except in degenerate
    cases where five points are co-spherical. Note however that the CGAL
    implementation computes a unique triangulation even in these cases.

    Examples
    --------
    .. testcode::
       :hide:

       from numpy.random import random, seed
       from pylab import *
       seed(42)
       gt.seed_rng(42)

    >>> points = random((500, 2)) * 4
    >>> g, pos = gt.triangulation(points)
    >>> weight = g.new_edge_property("double") # Edge weights corresponding to
    ...                                        # Euclidean distances
    >>> for e in g.edges():
    ...    weight[e] = sqrt(sum((array(pos[e.source()]) -
    ...                          array(pos[e.target()]))**2))
    >>> b = gt.betweenness(g, weight=weight)
    >>> b[1].a *= 100
    >>> gt.graph_draw(g, pos=pos, vertex_fill_color=b[0],
    ...               edge_pen_width=b[1], output="triang.pdf")
    <...>

    >>> g, pos = gt.triangulation(points, type="delaunay")
    >>> weight = g.new_edge_property("double")
    >>> for e in g.edges():
    ...    weight[e] = sqrt(sum((array(pos[e.source()]) -
    ...                          array(pos[e.target()]))**2))
    >>> b = gt.betweenness(g, weight=weight)
    >>> b[1].a *= 120
    >>> gt.graph_draw(g, pos=pos, vertex_fill_color=b[0],
    ...               edge_pen_width=b[1], output="triang-delaunay.pdf")
    <...>

    .. testcleanup::

       conv_png("triang.pdf")
       conv_png("triang-delaunay.pdf")

    2D triangulation of random points:

    .. image:: triang.png
       :width: 40%
    .. image:: triang-delaunay.png
       :width: 40%

    *Left:* Simple triangulation. *Right:* Delaunay triangulation. The vertex
    colors and the edge thickness correspond to the weighted betweenness
    centrality.

    References
    ----------
    .. [cgal-triang] http://www.cgal.org/Manual/last/doc_html/cgal_manual/Triangulation_3/Chapter_main.html

    """

    if points.shape[1] not in [2, 3]:
        raise ValueError("points array must have shape N x d, with d either 2 or 3.")
    # copy points to ensure continuity and correct data type
    points = numpy.array(points, dtype='float64')
    if points.shape[1] == 2:
        npoints = numpy.zeros((points.shape[0], 3))
        npoints[:,:2] = points
        points = npoints
    g = Graph(directed=False)
    pos = g.new_vertex_property("vector<double>")
    libgraph_tool_generation.triangulation(g._Graph__graph, points,
                                           _prop("v", g, pos), type,
                                           periodic)
    return g, pos


def lattice(shape, periodic=False):
    r"""
    Generate a N-dimensional square lattice.

    Parameters
    ----------
    shape : list or :class:`numpy.ndarray`
        List of sizes in each dimension.
    periodic : bool (optional, default: ``False``)
        If ``True``, periodic boundary conditions will be used.

    Returns
    -------
    lattice_graph : :class:`~graph_tool.Graph`
        The generated graph.

    See Also
    --------
    triangulation: 2D or 3D triangulation
    random_graph: random graph generation

    Examples
    --------
    .. testcode::
       :hide:

       gt.seed_rng(42)

    >>> g = gt.lattice([10,10])
    >>> pos = gt.sfdp_layout(g, cooling_step=0.95, epsilon=1e-2)
    >>> gt.graph_draw(g, pos=pos, output="lattice.pdf")
    <...>

    >>> g = gt.lattice([10,20], periodic=True)
    >>> pos = gt.sfdp_layout(g, cooling_step=0.95, epsilon=1e-2)
    >>> gt.graph_draw(g, pos=pos, output="lattice_periodic.pdf")
    <...>

    >>> g = gt.lattice([10,10,10])
    >>> pos = gt.sfdp_layout(g, cooling_step=0.95, epsilon=1e-2)
    >>> gt.graph_draw(g, pos=pos, output="lattice_3d.pdf")
    <...>

    .. testcleanup::

       conv_png("lattice.pdf")
       conv_png("lattice_periodic.pdf")
       conv_png("lattice_3d.pdf")

    .. image:: lattice.png
       :width: 33%
    .. image:: lattice_periodic.png
       :width: 33%
    .. image:: lattice_3d.png
       :width: 33%

    *Left:* 10x10 2D lattice. *Middle:* 10x20 2D periodic lattice (torus).
    *Right:* 10x10x10 3D lattice.

    References
    ----------
    .. [lattice] http://en.wikipedia.org/wiki/Square_lattice

    """

    g = Graph(directed=False)
    libgraph_tool_generation.lattice(g._Graph__graph, shape, periodic)
    return g

def complete_graph(N, self_loops=False, directed=False):
    r"""
    Generate complete graph.

    Parameters
    ----------
    N : ``int``
        Number of vertices.
    self_loops : bool (optional, default: ``False``)
        If ``True``, self-loops are included.
    directed : bool (optional, default: ``False``)
        If ``True``, a directed graph is generated.

    Returns
    -------
    complete_graph : :class:`~graph_tool.Graph`
        A complete graph.

    Examples
    --------
    .. doctest:: complete

       >>> g = gt.complete_graph(30)
       >>> pos = gt.sfdp_layout(g, cooling_step=0.95, epsilon=1e-2)
       >>> gt.graph_draw(g, pos=pos, output="complete.pdf")
       <...>

    .. testcleanup:: complete

       conv_png("complete.pdf")

    .. figure:: complete.png
       :width: 20%

       A complete graph with :math:`N=30` vertices.

    References
    ----------
    .. [complete] http://en.wikipedia.org/wiki/Complete_graph

    """

    g = Graph(directed=directed)
    libgraph_tool_generation.complete(g._Graph__graph, N, directed, self_loops)
    return g

def circular_graph(N, k=1, self_loops=False, directed=False):
    r"""
    Generate a circular graph.

    Parameters
    ----------
    N : ``int``
        Number of vertices.
    k : ``int`` (optional, default: ``True``)
        Number of nearest neighbors to be connected.
    self_loops : bool (optional, default: ``False``)
        If ``True``, self-loops are included.
    directed : bool (optional, default: ``False``)
        If ``True``, a directed graph is generated.

    Returns
    -------
    circular_graph : :class:`~graph_tool.Graph`
        A circular graph.

    Examples
    --------

    >>> g = gt.circular_graph(30, 2)
    >>> pos = gt.sfdp_layout(g, cooling_step=0.95)
    >>> gt.graph_draw(g, pos=pos, output="circular.pdf")
    <...>

    .. testcleanup::

       conv_png("circular.pdf")

    .. figure:: circular.png
       :width: 20%

       A circular graph with :math:`N=30` vertices, and :math:`k=2`.

    """

    g = Graph(directed=directed)
    libgraph_tool_generation.circular(g._Graph__graph, N, k, directed, self_loops)
    return g


def geometric_graph(points, radius, ranges=None):
    r"""
    Generate a geometric network form a set of N-dimensional points.

    Parameters
    ----------
    points : list or :class:`numpy.ndarray`
        List of points. This must be a two-dimensional array, where the rows are
        coordinates in a N-dimensional space.
    radius : float
        Pairs of points with an euclidean distance lower than this parameters
        will be connected.
    ranges : list or :class:`numpy.ndarray` (optional, default: ``None``)
        If provided, periodic boundary conditions will be assumed, and the
        values of this parameter it will be used as the ranges in all
        dimensions. It must be a two-dimensional array, where each row will
        cointain the lower and upper bound of each dimension.

    Returns
    -------
    geometric_graph : :class:`~graph_tool.Graph`
        The generated graph.
    pos : :class:`~graph_tool.VertexPropertyMap`
        A vertex property map with the position of each vertex.

    Notes
    -----
    A geometric graph [geometric-graph]_ is generated by connecting points
    embedded in a N-dimensional euclidean space which are at a distance equal to
    or smaller than a given radius.

    See Also
    --------
    triangulation: 2D or 3D triangulation
    random_graph: random graph generation
    lattice : N-dimensional square lattice

    Examples
    --------
    .. testcode::
       :hide:

       from numpy.random import random, seed
       from pylab import *
       seed(42)
       gt.seed_rng(42)

    >>> points = random((500, 2)) * 4
    >>> g, pos = gt.geometric_graph(points, 0.3)
    >>> gt.graph_draw(g, pos=pos, output="geometric.pdf")
    <...>

    >>> g, pos = gt.geometric_graph(points, 0.3, [(0,4), (0,4)])
    >>> pos = gt.graph_draw(g, output="geometric_periodic.pdf")

    .. testcleanup::

       conv_png("geometric.pdf")
       conv_png("geometric_periodic.pdf")

    .. image:: geometric.png
       :width: 40%
    .. image:: geometric_periodic.png
       :width: 40%

    *Left:* Geometric network with random points. *Right:* Same network, but
     with periodic boundary conditions.

    References
    ----------
    .. [geometric-graph] Jesper Dall and Michael Christensen, "Random geometric
       graphs", Phys. Rev. E 66, 016121 (2002), :doi:`10.1103/PhysRevE.66.016121`

    """

    g = Graph(directed=False)
    pos = g.new_vertex_property("vector<double>")
    if type(points) != numpy.ndarray:
        points = numpy.array(points)
    if len(points.shape) < 2:
        raise ValueError("points list must be a two-dimensional array!")
    if ranges is not None:
        periodic = True
        if type(ranges) != numpy.ndarray:
            ranges = numpy.array(ranges, dtype="float")
        else:
            ranges = array(ranges, dtype="float")
    else:
        periodic = False
        ranges = ()

    libgraph_tool_generation.geometric(g._Graph__graph, points, float(radius),
                                       ranges, periodic,
                                       _prop("v", g, pos))
    return g, pos


def price_network(N, m=1, c=None, gamma=1, directed=True, seed_graph=None):
    r"""A generalized version of Price's --- or Barabási-Albert if undirected
    --- preferential attachment network model.

    Parameters
    ----------
    N : int
        Size of the network.
    m : int (optional, default: ``1``)
        Out-degree of newly added vertices.
    c : float (optional, default: ``1 if directed == True else 0``)
        Constant factor added to the probability of a vertex receiving an edge
        (see notes below).
    gamma : float (optional, default: ``1``)
        Preferential attachment exponent (see notes below).
    directed : bool (optional, default: ``True``)
        If ``True``, a Price network is generated. If ``False``, a
        Barabási-Albert network is generated.
    seed_graph : :class:`~graph_tool.Graph` (optional, default: ``None``)
        If provided, this graph will be used as the starting point of the
        algorithm.

    Returns
    -------
    price_graph : :class:`~graph_tool.Graph`
        The generated graph.

    Notes
    -----

    The (generalized) [price]_ network is either a directed or undirected graph
    (the latter is called a Barabási-Albert network), generated dynamically by
    at each step adding a new vertex, and connecting it to :math:`m` other
    vertices, chosen with probability :math:`\pi` defined as:

    .. math::

        \pi \propto k^\gamma + c

    where :math:`k` is the (in-)degree of the vertex (or simply the degree in
    the undirected case).

    Note that for directed graphs we must have :math:`c \ge 0`, and for
    undirected graphs, :math:`c > -\min(k_{\text{min}}, m)^{\gamma}`, where
    :math:`k_{\text{min}}` is the smallest degree in the seed graph.

    If :math:`\gamma=1`, the tail of resulting in-degree distribution of the
    directed case is given by

    .. math::

        P_{k_\text{in}} \sim k_\text{in}^{-(2 + c/m)},

    or for the undirected case

    .. math::

        P_{k} \sim k^{-(3 + c/m)}.

    However, if :math:`\gamma \ne 1`, the in-degree distribution is not
    scale-free (see [dorogovtsev-evolution]_ for details).

    Note that if `seed_graph` is not given, the algorithm will *always* start
    with one node if :math:`c > 0`, or with two nodes with an edge between them
    otherwise. If :math:`m > 1`, the degree of the newly added vertices will be
    vary dynamically as :math:`m'(t) = \min(m, V(t))`, where :math:`V(t)` is the
    number of vertices added so far. If this behaviour is undesired, a proper
    seed graph with :math:`V \ge m` vertices must be provided.

    This algorithm runs in :math:`O(V\log V)` time.

    See Also
    --------
    triangulation: 2D or 3D triangulation
    random_graph: random graph generation
    lattice : N-dimensional square lattice
    geometric_graph : N-dimensional geometric network

    Examples
    --------
    .. testcode::
       :hide:

       gt.seed_rng(42)

    >>> g = gt.price_network(20000)
    >>> gt.graph_draw(g, pos=gt.sfdp_layout(g, cooling_step=0.99),
    ...               vertex_fill_color=g.vertex_index, vertex_size=2,
    ...               vcmap=matplotlib.cm.plasma,
    ...               edge_pen_width=1, output="price-network.pdf")
    <...>
    >>> g = gt.price_network(20000, c=0.1)
    >>> gt.graph_draw(g, pos=gt.sfdp_layout(g, cooling_step=0.99),
    ...               vertex_fill_color=g.vertex_index, vertex_size=2,
    ...               vcmap=matplotlib.cm.plasma,
    ...               edge_pen_width=1, output="price-network-broader.pdf")
    <...>

    .. testcleanup::

       conv_png("price-network.pdf")
       conv_png("price-network-broader.pdf")

    .. figure:: price-network.png
       :align: center
       :width: 60%

       Price network with :math:`N=2\times 10^4` nodes and :math:`c=1`.  The
       colors represent the order in which vertices were added.

    .. figure:: price-network-broader.png
       :align: center
       :width: 60%

       Price network with :math:`N=2\times 10^4` nodes and :math:`c=0.1`.  The
       colors represent the order in which vertices were added.


    References
    ----------

    .. [yule] Yule, G. U. "A Mathematical Theory of Evolution, based on the
       Conclusions of Dr. J. C. Willis, F.R.S.". Philosophical Transactions of
       the Royal Society of London, Ser. B 213: 21-87, 1925,
       :doi:`10.1098/rstb.1925.0002`
    .. [price] Derek De Solla Price, "A general theory of bibliometric and other
       cumulative advantage processes", Journal of the American Society for
       Information Science, Volume 27, Issue 5, pages 292-306, September 1976,
       :doi:`10.1002/asi.4630270505`
    .. [barabasi-albert] Barabási, A.-L., and Albert, R., "Emergence of
       scaling in random networks", Science, 286, 509, 1999,
       :doi:`10.1126/science.286.5439.509`
    .. [dorogovtsev-evolution] S. N. Dorogovtsev and J. F. F. Mendes, "Evolution
       of networks", Advances in Physics, 2002, Vol. 51, No. 4, 1079-1187,
       :doi:`10.1080/00018730110112519`

    """

    if c is None:
        c = 1 if directed else 0

    if seed_graph is None:
        g = Graph(directed=directed)
        if c > 0:
            g.add_vertex()
        else:
            g.add_vertex(2)
            g.add_edge(g.vertex(1), g.vertex(0))
        N -= g.num_vertices()
    else:
        g = seed_graph

    if ((directed and c < 0) or
        (not directed and c <= -min(g.degree_property_map("out").fa.min(), m) ** gamma)):
        raise ValueError("Parameter 'c' is too small, and yields negative probabilities")

    libgraph_tool_generation.price(g._Graph__graph, N, gamma, c, m, _get_rng())

    return g

def condensation_graph(g, prop, vweight=None, eweight=None, avprops=None,
                       aeprops=None, self_loops=False, parallel_edges=False):
    r"""
    Obtain the condensation graph, where each vertex with the same 'prop' value
    is condensed in one vertex.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modelled.
    prop : :class:`~graph_tool.VertexPropertyMap`
        Vertex property map with the community partition.
    vweight : :class:`~graph_tool.VertexPropertyMap` (optional, default: None)
        Vertex property map with the optional vertex weights.
    eweight : :class:`~graph_tool.EdgePropertyMap` (optional, default: None)
        Edge property map with the optional edge weights.
    avprops : list of :class:`~graph_tool.VertexPropertyMap` (optional, default: None)
        If provided, the sum of each property map in this list for
        each vertex in the condensed graph will be computed and returned.
    aeprops : list of :class:`~graph_tool.EdgePropertyMap` (optional, default: None)
        If provided, the sum of each property map in this list for
        each edge in the condensed graph will be computed and returned.
    self_loops : ``bool`` (optional, default: ``False``)
        If ``True``, self-loops due to intra-block edges are also included in
        the condensation graph.
    parallel_edges : ``bool`` (optional, default: ``False``)
        If ``True``, parallel edges will be included in the condensation graph,
        such that the total number of edges will be the same as in the original
        graph.

    Returns
    -------
    condensation_graph : :class:`~graph_tool.Graph`
        The community network
    prop : :class:`~graph_tool.VertexPropertyMap`
        The community values.
    vcount : :class:`~graph_tool.VertexPropertyMap`
        A vertex property map with the vertex count for each community.
    ecount : :class:`~graph_tool.EdgePropertyMap`
        An edge property map with the inter-community edge count for each edge.
    va : list of :class:`~graph_tool.VertexPropertyMap`
        A list of vertex property maps with summed values of the properties
        passed via the ``avprops`` parameter.
    ea : list of :class:`~graph_tool.EdgePropertyMap`
        A list of edge property maps with summed values of the properties
        passed via the ``avprops`` parameter.

    Notes
    -----
    Each vertex in the condensation graph represents one community in the
    original graph (vertices with the same 'prop' value), and the edges
    represent existent edges between vertices of the respective communities in
    the original graph.

    Examples
    --------

    .. testsetup:: condensation_graph

       gt.seed_rng(43)
       np.random.seed(42)

    Let's first obtain the best block partition with ``B=5``.

    .. doctest:: condensation_graph

       >>> g = gt.collection.data["polbooks"]
       >>> # fit a SBM
       >>> state = gt.BlockState(g)
       >>> gt.mcmc_equilibrate(state, wait=1000)
       (...)
       >>> b = state.get_blocks()
       >>> b = gt.perfect_prop_hash([b])[0]
       >>> gt.graph_draw(g, pos=g.vp["pos"], vertex_fill_color=b, vertex_shape=b,
       ...               output="polbooks_blocks_B5.pdf")
       <...>

    Now we get the condensation graph:

    .. doctest:: condensation_graph

       >>> bg, bb, vcount, ecount, avp, aep = \
       ...     gt.condensation_graph(g, b, avprops=[g.vp["pos"]],
       ...                           self_loops=True)
       >>> pos = avp[0]
       >>> for v in bg.vertices():
       ...     pos[v].a /= vcount[v]
       >>> gt.graph_draw(bg, pos=avp[0], vertex_fill_color=bb, vertex_shape=bb,
       ...               vertex_size=gt.prop_to_size(vcount, mi=40, ma=100),
       ...               edge_pen_width=gt.prop_to_size(ecount, mi=2, ma=10),
       ...               fit_view=.8, output="polbooks_blocks_B5_cond.pdf")
       <...>

    .. testcleanup:: condensation_graph

       conv_png("polbooks_blocks_B5.pdf")
       conv_png("polbooks_blocks_B5_cond.pdf")

    .. figure:: polbooks_blocks_B5.png
       :align: center
       :width: 60%

       Block partition of a political books network with :math:`B=5`.

    .. figure:: polbooks_blocks_B5_cond.png
       :align: center
       :width: 60%

       Condensation graph of the obtained block partition.

    """
    gp = Graph(directed=g.is_directed())
    if vweight is None:
        vcount = gp.new_vertex_property("int32_t")
    else:
        vcount = gp.new_vertex_property(vweight.value_type())
    if eweight is None:
        ecount = gp.new_edge_property("int32_t")
    else:
        ecount = gp.new_edge_property(eweight.value_type())

    if prop is g.vertex_index:
        prop = prop.copy(value_type="int32_t")
    cprop = gp.new_vertex_property(prop.value_type())

    if avprops is None:
        avprops = []
    avp = []
    r_avp = []
    for p in avprops:
        if p is g.vertex_index:
            p = p.copy(value_type="int")
        if "string" in p.value_type():
            raise ValueError("Cannot compute sum of string properties!")
        temp = g.new_vertex_property(p.value_type())
        cp = gp.new_vertex_property(p.value_type())
        avp.append((_prop("v", g, p), _prop("v", g, temp), _prop("v", gp, cp)))
        r_avp.append(cp)

    if aeprops is None:
        aeprops = []
    aep = []
    r_aep = []
    for p in aeprops:
        if p is g.edge_index:
            p = p.copy(value_type="int")
        if "string" in p.value_type():
            raise ValueError("Cannot compute sum of string properties!")
        temp = g.new_edge_property(p.value_type())
        cp = gp.new_edge_property(p.value_type())
        aep.append((_prop("e", g, p), _prop("e", g, temp), _prop("e", gp, cp)))
        r_aep.append(cp)

    libgraph_tool_generation.community_network(g._Graph__graph,
                                               gp._Graph__graph,
                                               _prop("v", g, prop),
                                               _prop("v", gp, cprop),
                                               _prop("v", gp, vcount),
                                               _prop("e", gp, ecount),
                                               _prop("v", g, vweight),
                                               _prop("e", g, eweight),
                                               self_loops,
                                               parallel_edges)

    u = GraphView(g, directed=True, reversed=False)
    libgraph_tool_generation.community_network_vavg(u._Graph__graph,
                                                    gp._Graph__graph,
                                                    _prop("v", g, prop),
                                                    _prop("v", gp, cprop),
                                                    _prop("v", g, vweight),
                                                    avp)

    libgraph_tool_generation.community_network_eavg(g._Graph__graph,
                                                    gp._Graph__graph,
                                                    _prop("v", g, prop),
                                                    _prop("v", gp, cprop),
                                                    _prop("e", g, eweight),
                                                    aep, self_loops,
                                                    parallel_edges)
    return gp, cprop, vcount, ecount, r_avp, r_aep

def contract_parallel_edges(g, weight=None):
    r"""Contract all parallel edges into simple edges.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modified.
    weight : :class:`~graph_tool.EdgePropertyMap`, optional (default: ``None``)
        Edge multiplicities.

    Returns
    -------
    weight : :class:`~graph_tool.EdgePropertyMap`
        Edge multiplicities.

    See Also
    --------
    expand_parallel_edges: expand edge multiplicities into parallel edges.

    Notes
    -----
    This algorithm runs in time :math:`O(N + E)` where :math:`N` and :math:`E`
    are the number of nodes and edges in the graph, respectively.

    Examples
    --------

    >>> u = gt.collection.data["polblogs"].copy()
    >>> u.set_directed(False)
    >>> g = u.copy()
    >>> w = gt.contract_parallel_edges(g)
    >>> gt.expand_parallel_edges(g, w)
    >>> gt.similarity(g, u)
    1.0
    """

    if weight is None:
        weight = g.new_ep("int", val=1)
    libgraph_tool_generation.\
        contract_parallel_edges(g._Graph__graph, _prop("e", g, weight))
    return weight

def remove_parallel_edges(g):
    """Remove all parallel edges from the graph. Only one edge from each
    parallel edge set is left."""
    libgraph_tool_generation.\
        contract_parallel_edges(g._Graph__graph, _prop("e", g, None))

def remove_self_loops(g):
    """Remove all self-loops edges from the graph."""
    eprop = label_self_loops(g)
    remove_labeled_edges(g, eprop)

def expand_parallel_edges(g, weight):
    r"""Expand edge multiplicities into parallel edges.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modified.
    weight : :class:`~graph_tool.EdgePropertyMap`
        Edge multiplicities.

    See Also
    --------
    contract_parallel_edges: contract all parallel edges into simple edges.

    Notes
    -----

    This algorithm runs in time :math:`O(N + E)` where :math:`N` is the number
    of nodes and :math:`E` is the final number of edges in the graph.

    Examples
    --------
    >>> u = gt.collection.data["polblogs"].copy()
    >>> u.set_directed(False)
    >>> g = u.copy()
    >>> w = gt.contract_parallel_edges(g)
    >>> gt.expand_parallel_edges(g, w)
    >>> gt.similarity(g, u)
    1.0

    """
    libgraph_tool_generation.\
        expand_parallel_edges(g._Graph__graph, _prop("e", g, weight))

def remove_labeled_edges(g, label):
    """Remove every edge `e` such that `label[e] != 0`."""
    u = GraphView(g, directed=True, reversed=g.is_reversed(),
                  skip_properties=True)
    libgraph_tool_generation.\
          remove_labeled_edges(u._Graph__graph, _prop("e", g, label))


def label_parallel_edges(g, mark_only=False, eprop=None):
    r"""Label edges which are parallel, i.e, have the same source and target
    vertices. For each parallel edge set :math:`PE`, the labelling starts from 0
    to :math:`|PE|-1`. If `mark_only==True`, all parallel edges are simply
    marked with the value 1. If the `eprop` parameter is given (a
    :class:`~graph_tool.EdgePropertyMap`), the labelling is stored there."""
    if eprop is None:
        if mark_only:
            eprop = g.new_edge_property("bool")
        else:
            eprop = g.new_edge_property("int32_t")
    libgraph_tool_generation.\
          label_parallel_edges(g._Graph__graph, _prop("e", g, eprop),
                               mark_only)
    return eprop


def label_self_loops(g, mark_only=False, eprop=None):
    """Label edges which are self-loops, i.e, the source and target vertices are
    the same. For each self-loop edge set :math:`SL`, the labelling starts from 0
    to :math:`|SL|-1`. If `mark_only == True`, self-loops are labeled with 1
    and others with 0. If the `eprop` parameter is given
    (a :class:`~graph_tool.EdgePropertyMap`), the labelling is stored there."""

    if eprop is None:
        if mark_only:
            eprop = g.new_edge_property("bool")
        else:
            eprop = g.new_edge_property("int32_t")
    libgraph_tool_generation.\
          label_self_loops(g._Graph__graph, _prop("e", g, eprop),
                           mark_only)
    return eprop

class Sampler(libgraph_tool_generation.Sampler):
    def __init__(self, values, probs):
        libgraph_tool_generation.Sampler.__init__(self, values, probs)

    def sample(self):
        return libgraph_tool_generation.Sampler.sample(self, _get_rng())

class DynamicSampler(libgraph_tool_generation.DynamicSampler):
    def __init__(self, values=None, probs=None):
        if values is None:
            values = probs = []
        libgraph_tool_generation.DynamicSampler.__init__(self, values, probs)

    def sample(self):
        return libgraph_tool_generation.DynamicSampler.sample(self, _get_rng())
