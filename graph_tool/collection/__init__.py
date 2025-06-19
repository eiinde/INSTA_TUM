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
``graph_tool.collection``
-------------------------

This module contains an assortment of useful networks.

Interface to the Netzschleuder online network repository
========================================================

.. data:: ns

    Dictionary containing :class:`~graph_tool.Graph` objects, indexed by the
    name of the dataset, fetched from the `Netzschleuder
    <https://networks.skewed.de>`__ repository. For dataset entries with more
    than one network, they are accessed either via a string
    ``"<entry>/<network>"`` or a tuple ``("<entry>", "<network>")``. This is a
    "lazy" dictionary, i.e. it only downloads the graphs when the items are
    accessed for the first time. The description and summary information for
    each graph are given in the :data:`ns_info` dictionary, or alternatively in
    the graph properties which accompanies each graph object.


    Examples
    ++++++++
    >>> g = gt.collection.ns["advogato"]
    >>> print(g.gp.description)
    A network of trust relationships among users on Advogato, an online community of open source software developers. Edge direction indicates that node i trusts node j, and edge weight denotes one of four increasing levels of declared trust from i to j: observer (0.4), apprentice (0.6), journeyer (0.8), and master (1.0).


.. data:: ns_info

    Dictionary containing descriptions and other summary information for
    datasets available in the `Netzschleuder <https://networks.skewed.de>`__
    repository. For dataset entries with more than one network, they are
    accessed either via a string ``"<entry>/<network>"`` or a tuple
    ``("<entry>", "<network>")``. The information is downloaded on-the-fly via
    the available JSON API.

Built-in collection of empirical network data
=============================================

.. data:: data

    Dictionary containing :class:`~graph_tool.Graph` objects, indexed by the
    name of the graph. This is a "lazy" dictionary, i.e. it only loads the
    graphs from disk when the items are accessed for the first time.  The
    description for each graph is given in the :data:`descriptions` dictionary,
    or alternatively in the ``"description"`` graph property which accompanies
    each graph object.

    Examples
    ++++++++
    >>> g = gt.collection.data["karate"]
    >>> print(g)
    <Graph object, undirected, with 34 vertices and 78 edges, 1 internal vertex property, 2 internal graph properties, at 0x...>
    >>> print(g.gp.readme)
    The file karate.gml contains the network of friendships between the 34
    members of a karate club at a US university, as described by Wayne Zachary
    in 1977.  If you use these data in your work, please cite W. W. Zachary, An
    information flow model for conflict and fission in small groups, Journal of
    Anthropological Research 33, 452-473 (1977).
    <BLANKLINE>


.. data:: descriptions

    Dictionary with a short description and source information on each graph
    contained in :data:`data`.

    A summary, with some extra information, is available in the following table.

    .. table::

        ===================  ===========  ===========  ========  ================================================
        Name                 N            E            Directed  Description
        ===================  ===========  ===========  ========  ================================================
        adjnoun              112          425          False     Word adjacencies: adjacency network of common
                                                                 adjectives and nouns in the novel David
                                                                 Copperfield by Charles Dickens. Please cite M.
                                                                 E. J. Newman, Phys. Rev. E 74, 036104 (2006).
                                                                 Retrieved from `Mark Newman's website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        as-22july06          22963        48436        False     Internet: a symmetrized snapshot of the
                                                                 structure of the Internet at the level of
                                                                 autonomous systems, reconstructed from BGP
                                                                 tables posted by the `University of Oregon Route
                                                                 Views Project <http://routeviews.org/>`_. This
                                                                 snapshot was created by Mark Newman from data
                                                                 for July 22, 2006 and is not previously
                                                                 published. Retrieved from `Mark Newman's website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        astro-ph             16706        121251       False     Astrophysics collaborations: weighted network of
                                                                 coauthorships between scientists posting
                                                                 preprints on the Astrophysics E-Print Archive
                                                                 between Jan 1, 1995 and December 31, 1999.
                                                                 Please cite M. E. J. Newman, Proc. Natl. Acad.
                                                                 Sci. USA 98, 404-409 (2001). Retrieved from
                                                                 `Mark Newman's website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        celegansneural       297          2359         True      Neural network: A directed, weighted network
                                                                 representing the neural network of C. Elegans.
                                                                 Data compiled by D. Watts and S. Strogatz and
                                                                 made available on the web `here
                                                                 <http://cdg.columbia.edu/cdg/datasets>`__. Please
                                                                 cite D. J. Watts and S. H. Strogatz, Nature 393,
                                                                 440-442 (1998). Original experimental data taken
                                                                 from J. G. White, E. Southgate, J. N. Thompson,
                                                                 and S. Brenner, Phil. Trans. R. Soc. London 314,
                                                                 1-340 (1986). Retrieved from `Mark Newman's
                                                                 website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        cond-mat             16726        47594        False     Condensed matter collaborations 1999: weighted
                                                                 network of coauthorships between scientists
                                                                 posting preprints on the Condensed Matter
                                                                 E-Print Archive between Jan 1, 1995 and December
                                                                 31, 1999. Please cite M. E. J. Newman, The
                                                                 structure of scientific collaboration networks,
                                                                 Proc. Natl. Acad. Sci. USA 98, 404-409 (2001).
                                                                 Retrieved from `Mark Newman's website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        cond-mat-2003        31163        120029       False     Condensed matter collaborations 2003: updated
                                                                 network of coauthorships between scientists
                                                                 posting preprints on the Condensed Matter
                                                                 E-Print Archive. This version includes all
                                                                 preprints posted between Jan 1, 1995 and June
                                                                 30, 2003. The largest component of this network,
                                                                 which contains 27519 scientists, has been used
                                                                 by several authors as a test-bed for
                                                                 community-finding algorithms for large networks;
                                                                 see for example J. Duch and A. Arenas, Phys.
                                                                 Rev. E 72, 027104 (2005). These data can be
                                                                 cited as M. E. J. Newman, Proc. Natl. Acad. Sci.
                                                                 USA 98, 404-409 (2001). Retrieved from `Mark
                                                                 Newman's website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        cond-mat-2005        40421        175693       False     Condensed matter collaborations 2005: updated
                                                                 network of coauthorships between scientists
                                                                 posting preprints on the Condensed Matter
                                                                 E-Print Archive. This version includes all
                                                                 preprints posted between Jan 1, 1995 and March
                                                                 31, 2005. Please cite M. E. J. Newman, Proc.
                                                                 Natl. Acad. Sci. USA 98, 404-409 (2001).
                                                                 Retrieved from `Mark Newman's website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        dolphins             62           159          False     Dolphin social network: an undirected social
                                                                 network of frequent associations between 62
                                                                 dolphins in a community living off Doubtful
                                                                 Sound, New Zealand. Please cite D. Lusseau, K.
                                                                 Schneider, O. J. Boisseau, P. Haase, E. Slooten,
                                                                 and S. M. Dawson, Behavioral Ecology and
                                                                 Sociobiology 54, 396-405 (2003). Retrieved from
                                                                 `Mark Newman's website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        email-Enron          36692        367662       False     Enron email communication network covers all the
                                                                 email communication within a dataset of around
                                                                 half million emails. This data was originally
                                                                 made public, and posted to the web, by the
                                                                 Federal Energy Regulatory Commission during its
                                                                 investigation. Nodes of the network are email
                                                                 addresses and if an address i sent at least one
                                                                 email to address j, the graph contains an
                                                                 undirected edge from i to j. Note that non-Enron
                                                                 email addresses act as sinks and sources in the
                                                                 network as we only observe their communication
                                                                 with the Enron email addresses. The Enron email
                                                                 data was `originally released
                                                                 <http://www.cs.cmu.edu/~enron/>`_ by William
                                                                 Cohen at CMU. This version was retrieved from
                                                                 the SNAP database at
                                                                 http://snap.stanford.edu/data/email-Enron.html.
                                                                 Please cite: J. Leskovec, K. Lang, A. Dasgupta,
                                                                 M. Mahoney. Community Structure in Large
                                                                 Networks: Natural Cluster Sizes and the Absence
                                                                 of Large Well-Defined Clusters. Internet
                                                                 Mathematics 6(1) 29--123, 2009,  B. Klimmt, Y.
                                                                 Yang. Introducing the Enron corpus. CEAS
                                                                 conference, 2004.
        football             115          615          False     American College football: network of American
                                                                 football games between Division IA colleges
                                                                 during regular season Fall 2000. Please cite M.
                                                                 Girvan and M. E. J. Newman, Proc. Natl. Acad.
                                                                 Sci. USA 99, 7821-7826 (2002), and T.S. Evans,
                                                                 "Clique Graphs and Overlapping Communities",
                                                                 J.Stat.Mech. (2010) P12037 [arXiv:1009.0638]. 
                                                                 Retrieved from `Mark Newman's website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_,
                                                                 with corrections by T. S. Evans, available 
                                                                 `here <http://figshare.com/articles/American_College_Football_Network_Files/93179>`__.
        hep-th               8361         15751        False     High-energy theory collaborations: weighted
                                                                 network of coauthorships between scientists
                                                                 posting preprints on the High-Energy Theory
                                                                 E-Print Archive between Jan 1, 1995 and December
                                                                 31, 1999. Please cite M. E. J. Newman, Proc.
                                                                 Natl. Acad. Sci. USA 98, 404-409 (2001).
                                                                 Retrieved from `Mark Newman's website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        karate               34           78           False     Zachary's karate club: social network of
                                                                 friendships between 34 members of a karate club
                                                                 at a US university in the 1970s. Please cite W.
                                                                 W. Zachary, An information flow model for
                                                                 conflict and fission in small groups, Journal of
                                                                 Anthropological Research 33, 452-473 (1977).
                                                                 Retrieved from `Mark Newman's website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        lesmis               77           254          False     Les Miserables: coappearance network of
                                                                 characters in the novel Les Miserables. Please
                                                                 cite D. E. Knuth, The Stanford GraphBase: A
                                                                 Platform for Combinatorial Computing,
                                                                 Addison-Wesley, Reading, MA (1993). Retrieved
                                                                 from `Mark Newman's website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        netscience           1589         2742         False     Coauthorships in network science: coauthorship
                                                                 network of scientists working on network theory
                                                                 and experiment, as compiled by M. Newman in May
                                                                 2006. A figure depicting the largest component
                                                                 of this network can be found `here
                                                                 <http://www-personal.umich.edu/~mejn/centrality/>`__.
                                                                 These data can be cited as M. E. J. Newman,
                                                                 Phys. Rev. E 74, 036104 (2006). Retrieved from
                                                                 `Mark Newman's website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        pgp-strong-2009      39796        301498       True      Strongly connected component of the PGP web of
                                                                 trust circa November 2009. The full data is
                                                                 available at http://key-server.de/dump/. Please
                                                                 cite: Richters O, Peixoto TP (2011) Trust
                                                                 Transitivity in Social Networks. PLoS ONE 6(4):
                                                                 e18384. :doi:`10.1371/journal.pone.0018384`.
        polblogs             1490         19090        True      Political blogs: A directed network of
                                                                 hyperlinks between weblogs on US politics,
                                                                 recorded in 2005 by Adamic and Glance. Please
                                                                 cite L. A. Adamic and N. Glance, "The political
                                                                 blogosphere and the 2004 US Election", in
                                                                 Proceedings of the WWW-2005 Workshop on the
                                                                 Weblogging Ecosystem (2005). Retrieved from
                                                                 `Mark Newman's website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        polbooks             105          441          False     Books about US politics: A network of books
                                                                 about US politics published around the time of
                                                                 the 2004 presidential election and sold by the
                                                                 online bookseller Amazon.com. Edges between
                                                                 books represent frequent copurchasing of books
                                                                 by the same buyers. The network was compiled by
                                                                 V. Krebs and is unpublished, but can found on
                                                                 Krebs' `web site <http://www.orgnet.com/>`_.
                                                                 Retrieved from `Mark Newman's website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        power                4941         6594         False     Power grid: An undirected, unweighted network
                                                                 representing the topology of the Western States
                                                                 Power Grid of the United States. Data compiled
                                                                 by D. Watts and S. Strogatz and made available
                                                                 on the web `here
                                                                 <http://cdg.columbia.edu/cdg/datasets>`__. Please
                                                                 cite D. J. Watts and S. H. Strogatz, Nature 393,
                                                                 440-442 (1998). Retrieved from `Mark Newman's
                                                                 website
                                                                 <http://www-personal.umich.edu/~mejn/netdata/>`_.
        serengeti-foodweb    161          592          True      Plant and mammal food web from the Serengeti
                                                                 savanna ecosystem in Tanzania. Please cite:
                                                                 Baskerville EB, Dobson AP, Bedford T, Allesina
                                                                 S, Anderson TM, et al. (2011) Spatial Guilds in
                                                                 the Serengeti Food Web Revealed by a Bayesian
                                                                 Group Model. PLoS Comput Biol 7(12): e1002321.
                                                                 :doi:`10.1371/journal.pcbi.1002321`
        ===================  ===========  ===========  ========  ================================================


Functions returning small graphs
================================

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   LCF_graph
   bull_graph
   chvatal_graph
   cubical_graph
   desargues_graph
   diamond_graph
   dodecahedral_graph
   frucht_graph
   heawood_graph
   hoffman_singleton_graph
   house_graph
   icosahedral_graph
   krackhardt_kite_graph
   moebius_kantor_graph
   octahedral_graph
   pappus_graph
   petersen_graph
   sedgewick_maze_graph
   tetrahedral_graph
   truncated_cube_graph
   truncated_tetrahedron_graph
   tutte_graph

Small graph atlas
=================

.. data:: atlas

    Lazy list of of all graphs with up to seven nodes named in the
    Graph Atlas [atlas]_.

    The graphs are listed in increasing according to

    1. number of nodes,
    2. number of edges,
    3. degree sequence (for example 111223 < 112222),
    4. number of automorphisms,

    in that order, with three exceptions:

    - graphs 55 and 56, with degree sequences 001111 and 000112,
    - graphs 1007 and 1008, with degree sequences 3333444 and 3333336,
    - graphs 1012 and 1213, with degree sequences 1244555 and 1244456.

    These errors are kept in this list so that the indices match those of
    [atlas]_.

    References
    ++++++++++
    .. [atlas] Ronald C. Read and Robin J. Wilson, “An Atlas of Graphs". Oxford
       University Press, 1998.
"""

import os.path
import textwrap
import gzip
import io
from .. import load_graph

__all__ = ["data", "descriptions", "ns", "ns_info", "atlas", "LCF_graph",
           "bull_graph", "chvatal_graph", "cubical_graph", "desargues_graph",
           "diamond_graph", "dodecahedral_graph", "frucht_graph",
           "heawood_graph", "hoffman_singleton_graph", "house_graph",
           "icosahedral_graph", "krackhardt_kite_graph", "moebius_kantor_graph",
           "octahedral_graph", "pappus_graph", "petersen_graph",
           "sedgewick_maze_graph", "tetrahedral_graph", "truncated_cube_graph",
           "truncated_tetrahedron_graph", "tutte_graph" ]

base_dir = os.path.dirname(__file__)

descriptions = {
    'adjnoun': "Word adjacencies: adjacency network of common adjectives and nouns in the novel David Copperfield by Charles Dickens. Please cite M. E. J. Newman, Phys. Rev. E 74, 036104 (2006). Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'as-22july06': "Internet: a symmetrized snapshot of the structure of the Internet at the level of autonomous systems, reconstructed from BGP tables posted by the `University of Oregon Route Views Project <http://routeviews.org/>`_. This snapshot was created by Mark Newman from data for July 22, 2006 and is not previously published. Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'astro-ph': "Astrophysics collaborations: weighted network of coauthorships between scientists posting preprints on the Astrophysics E-Print Archive between Jan 1, 1995 and December 31, 1999. Please cite M. E. J. Newman, Proc. Natl. Acad. Sci. USA 98, 404-409 (2001). Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'celegansneural': "Neural network: A directed, weighted network representing the neural network of C. Elegans. Data compiled by D. Watts and S. Strogatz and made available on the web `here <http://cdg.columbia.edu/cdg/datasets>`_. Please cite D. J. Watts and S. H. Strogatz, Nature 393, 440-442 (1998). Original experimental data taken from J. G. White, E. Southgate, J. N. Thompson, and S. Brenner, Phil. Trans. R. Soc. London 314, 1-340 (1986). Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'cond-mat': "Condensed matter collaborations 1999: weighted network of coauthorships between scientists posting preprints on the Condensed Matter E-Print Archive between Jan 1, 1995 and December 31, 1999. Please cite M. E. J. Newman, The structure of scientific collaboration networks, Proc. Natl. Acad. Sci. USA 98, 404-409 (2001). Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'cond-mat-2003': "Condensed matter collaborations 2003: updated network of coauthorships between scientists posting preprints on the Condensed Matter E-Print Archive. This version includes all preprints posted between Jan 1, 1995 and June 30, 2003. The largest component of this network, which contains 27519 scientists, has been used by several authors as a test-bed for community-finding algorithms for large networks; see for example J. Duch and A. Arenas, Phys. Rev. E 72, 027104 (2005). These data can be cited as M. E. J. Newman, Proc. Natl. Acad. Sci. USA 98, 404-409 (2001). Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'cond-mat-2005': "Condensed matter collaborations 2005: updated network of coauthorships between scientists posting preprints on the Condensed Matter E-Print Archive. This version includes all preprints posted between Jan 1, 1995 and March 31, 2005. Please cite M. E. J. Newman, Proc. Natl. Acad. Sci. USA 98, 404-409 (2001). Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'dolphins': "Dolphin social network: an undirected social network of frequent associations between 62 dolphins in a community living off Doubtful Sound, New Zealand. Please cite D. Lusseau, K. Schneider, O. J. Boisseau, P. Haase, E. Slooten, and S. M. Dawson, Behavioral Ecology and Sociobiology 54, 396-405 (2003). Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'email-Enron': 'Enron email communication network covers all the email communication within a dataset of around half million emails. This data was originally made public, and posted to the web, by the Federal Energy Regulatory Commission during its investigation. Nodes of the network are email addresses and if an address i sent at least one email to address j, the graph contains an undirected edge from i to j. Note that non-Enron email addresses act as sinks and sources in the network as we only observe their communication with the Enron email addresses. The Enron email data was `originally released <http://www.cs.cmu.edu/~enron/>`_ by William Cohen at CMU. This version was retrieved from the SNAP database at http://snap.stanford.edu/data/email-Enron.html. Please cite: J. Leskovec, K. Lang, A. Dasgupta, M. Mahoney. Community Structure in Large Networks: Natural Cluster Sizes and the Absence of Large Well-Defined Clusters. Internet Mathematics 6(1) 29--123, 2009,  B. Klimmt, Y. Yang. Introducing the Enron corpus. CEAS conference, 2004.',
    'football': "American College football: network of American football games between Division IA colleges during regular season Fall 2000. Please cite M. Girvan and M. E. J. Newman, Proc. Natl. Acad. Sci. USA 99, 7821-7826 (2002). Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'hep-th': "High-energy theory collaborations: weighted network of coauthorships between scientists posting preprints on the High-Energy Theory E-Print Archive between Jan 1, 1995 and December 31, 1999. Please cite M. E. J. Newman, Proc. Natl. Acad. Sci. USA 98, 404-409 (2001). Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'karate': "Zachary's karate club: social network of friendships between 34 members of a karate club at a US university in the 1970s. Please cite W. W. Zachary, An information flow model for conflict and fission in small groups, Journal of Anthropological Research 33, 452-473 (1977). Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'lesmis': "Les Miserables: coappearance network of characters in the novel Les Miserables. Please cite D. E. Knuth, The Stanford GraphBase: A Platform for Combinatorial Computing, Addison-Wesley, Reading, MA (1993). Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'netscience': "Coauthorships in network science: coauthorship network of scientists working on network theory and experiment, as compiled by M. Newman in May 2006. A figure depicting the largest component of this network can be found `here <http://www-personal.umich.edu/~mejn/centrality/>`_. These data can be cited as M. E. J. Newman, Phys. Rev. E 74, 036104 (2006). Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'pgp-strong-2009': 'Strongly connected component of the PGP web of trust circa November 2009. The full data is available at http://key-server.de/dump/. Please cite: Richters O, Peixoto TP (2011) Trust Transitivity in Social Networks. PLoS ONE 6(4): e18384. :doi:`10.1371/journal.pone.0018384`.',
    'polblogs': 'Political blogs: A directed network of hyperlinks between weblogs on US politics, recorded in 2005 by Adamic and Glance. Please cite L. A. Adamic and N. Glance, "The political blogosphere and the 2004 US Election", in Proceedings of the WWW-2005 Workshop on the Weblogging Ecosystem (2005). Retrieved from `Mark Newman\'s website <http://www-personal.umich.edu/~mejn/netdata/>`_.',
    'polbooks': "Books about US politics: A network of books about US politics published around the time of the 2004 presidential election and sold by the online bookseller Amazon.com. Edges between books represent frequent copurchasing of books by the same buyers. The network was compiled by V. Krebs and is unpublished, but can found on Krebs' `web site <http://www.orgnet.com/>`_. Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'power': "Power grid: An undirected, unweighted network representing the topology of the Western States Power Grid of the United States. Data compiled by D. Watts and S. Strogatz and made available on the web `here <http://cdg.columbia.edu/cdg/datasets>`_. Please cite D. J. Watts and S. H. Strogatz, Nature 393, 440-442 (1998). Retrieved from `Mark Newman's website <http://www-personal.umich.edu/~mejn/netdata/>`_.",
    'serengeti-foodweb': 'Plant and mammal food web from the Serengeti savanna ecosystem in Tanzania. Please cite: Baskerville EB, Dobson AP, Bedford T, Allesina S, Anderson TM, et al. (2011) Spatial Guilds in the Serengeti Food Web Revealed by a Bayesian Group Model. PLoS Comput Biol 7(12): e1002321. :doi:`10.1371/journal.pcbi.1002321`'
}

def get_data_path(name):
    r"""Return the full path of the corresponding dataset."""
    return base_dir + "/" + name + ".gt.gz"

class LazyDataDict(dict):
    def __contains__(self, k):
        if not super().__contains__(k):
            fname = get_data_path(k)
            return os.path.exists(fname)
        return True
    def __getitem__(self, k):
        if not super().__contains__(k):
            fname = get_data_path(k)
            if not os.path.exists(fname):
                raise KeyError(k)
            g = load_graph(fname)
            dict.__setitem__(self, k, g)
            return g
        return dict.__getitem__(self, k)
    def keys(self):
        return descriptions.keys()
    def items(self):
        for k in self.keys():
            self[k]  # force loading of lazy items
        return dict.items(self)

data = LazyDataDict()

def _update_descriptions():
    for k, g in data.items():
        descriptions[k] = g.gp["description"]

def _print_table():
    print("===================  ===========  ===========  ========  ================================================")
    print("Name                 N            E            Directed  Description")
    print("===================  ===========  ===========  ========  ================================================")
    for k in sorted(data.keys()):
        g = data[k]
        print("  ".join((k.ljust(19), str(g.num_vertices()).ljust(11),
                         str(g.num_edges()).ljust(11),
                         str(g.is_directed()).ljust(8))), end="  ")
        d = textwrap.wrap(descriptions[k], 48, break_long_words=False, break_on_hyphens=False)
        print(d[0])
        for line in d[1:]:
            print(" " * 57 + line)
    print("===================  ===========  ===========  ========  ================================================")


class LazyList(list):
    def _populate(self):
        if super().__len__() == 0:
            with gzip.open(base_dir + "/atlas.dat.gz", "rb") as f:
                while True:
                    s = int.from_bytes(f.read(4), 'little')
                    b = f.read(s)
                    if len(b) == 0:
                        break
                    g = load_graph(io.BytesIO(b), fmt="gt")
                    self.append(g)

    def __len__(self):
        self._populate()
        return super().__len__()

    def __getitem__(self, i):
        self._populate()
        return super().__getitem__(i)

    def __iter__(self):
        self._populate()
        return super().__iter__()

atlas = LazyList()

from . netzschleuder import *
from . small import *
