// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2024 Tiago de Paula Peixoto <tiago@skewed.de>
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 3 of the License, or (at your option) any
// later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#ifndef GRAPH_BLOCKMODEL_ENTROPY_HH
#define GRAPH_BLOCKMODEL_ENTROPY_HH

#include "../support/util.hh"

namespace graph_tool
{

// ====================
// Entropy calculation
// ====================

enum deg_dl_kind
{
    ENT,
    UNIFORM,
    DIST
};

struct entropy_args_t
{
    bool dense = false;
    bool multigraph = true;
    bool exact = true;
    bool adjacency = true;
    bool recs = true;
    bool deg_entropy = true;
    bool partition_dl = true;
    bool degree_dl = true;
    deg_dl_kind  degree_dl_kind = deg_dl_kind::DIST;
    bool edges_dl = true;
    bool recs_dl = true;
    double beta_dl = 1.;
    bool Bfield = false;
};

// Sparse entropy terms
// ====================

// exact microcanonical deg-corr entropy
template <class Graph>
[[gnu::always_inline]]
inline double eterm_exact(size_t r, size_t s, size_t mrs, const Graph&)
{
    double val = lgamma_fast(mrs + 1);

    if (is_directed_::apply<Graph>::type::value || r != s)
    {
        return -val;
    }
    else
    {
#ifndef __clang__
        constexpr
#endif
        double log_2 = log(2);
        return -val - mrs * log_2;
    }
}

template <class Graph>
[[gnu::always_inline]]
inline double vterm_exact(size_t mrp, size_t mrm, size_t wr, bool deg_corr,
                          const Graph&)
{
    if (deg_corr)
    {
        if constexpr (is_directed_::apply<Graph>::type::value)
            return lgamma_fast(mrp + 1) + lgamma_fast(mrm + 1);
        else
            return lgamma_fast(mrp + 1);
    }
    else
    {
        if constexpr (is_directed_::apply<Graph>::type::value)
            return (mrp + mrm) * safelog_fast(wr);
        else
            return mrp * safelog_fast(wr);
    }
}

// "edge" term of the entropy
template <class Graph>
[[gnu::always_inline]]
inline double eterm(size_t r, size_t s, size_t mrs, const Graph&)
{
    if (!is_directed_::apply<Graph>::type::value && r == s)
        mrs *= 2;

    double val = xlogx_fast(mrs);

    if (is_directed_::apply<Graph>::type::value || r != s)
        return -val;
    else
        return -val / 2;
}

// "vertex" term of the entropy
template <class Graph>
[[gnu::always_inline]]
inline double vterm(size_t mrp, size_t mrm, size_t wr, bool deg_corr,
                    Graph&)
{
    double one = 0.5;

    if constexpr (is_directed_::apply<Graph>::type::value)
        one = 1;

    if (deg_corr)
        return one * (xlogx_fast(mrm) + xlogx_fast(mrp));
    else
        return one * (mrm * safelog_fast(wr) + mrp * safelog_fast(wr));
}



// Dense entropy
// =============

// "edge" term of the entropy
template <class Graph>
[[gnu::pure]]
inline double eterm_dense(size_t r, size_t s, uint64_t ers, uint64_t wr_r,
                          uint64_t wr_s, bool multigraph, const Graph& g)
{
    uint64_t nrns; // avoid overflow for nr > 2^32

    if (ers == 0)
        return 0.;

    assert(wr_r + wr_s > 0);

    if (r != s || graph_tool::is_directed(g))
    {
        nrns = wr_r * wr_s;
    }
    else
    {
        if (multigraph)
            nrns = (wr_r * (wr_r + 1)) / 2;
        else
            nrns = (wr_r * (wr_r - 1)) / 2;
    }

    double S;
    if (multigraph)
        S = lbinom_fast<false>(nrns + ers - 1, ers); // do not use lbinom_fast<true>!
    else
        S = lbinom_fast<false>(nrns, ers);
    return S;
}

// Edges description length
template <class Graph>
double get_edges_dl(size_t B, size_t E, Graph& g)
{
    size_t NB = (graph_tool::is_directed(g)) ? B * B : (B * (B + 1)) / 2;
    return lbinom_fast<false>(NB + E - 1, E);
}

} // namespace graph_tool

#endif // GRAPH_BLOCKMODEL_ENTROPY_HH
