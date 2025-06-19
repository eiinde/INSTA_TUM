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

#ifndef GRAPH_MAXIMAL_CLIQUES_HH
#define GRAPH_MAXIMAL_CLIQUES_HH

#include <array>
#include <deque>

#include "hash_map_wrap.hh"

namespace graph_tool
{

using namespace boost;

template <class Graph, class Visitor>
void max_cliques(Graph& g, Visitor&& vis)
{
    typedef gt_hash_set<size_t> vset_t;
    //typedef std::set<size_t> vset_t;

    auto get_pivot_adj = [&](const auto& P, const auto& X, auto& u_adj)
    {
        auto u = graph_traits<Graph>::null_vertex();
        size_t ku = 0;
        std::array<const vset_t*, 2> Ss({&P, &X});
        for (auto S : Ss)
        {
            for (auto v : *S)
            {
                size_t k = 0;
                for (auto w : out_neighbors_range(v, g))
                {
                    if (w == v)
                        continue;
                    if (P.find(w) != P.end())
                        ++k;
                }
                if (k >= ku)
                {
                    u = v;
                    ku = k;
                }
            }
        }

        for (auto w : out_neighbors_range(u, g))
        {
            if (w == u)
                continue;
            u_adj.insert(w);
        }
    };

    std::deque<std::tuple<vset_t, vset_t, vset_t, vset_t, vset_t::iterator>>
        stack;
    stack.emplace_back();
    auto& P_ = get<1>(stack.back());
    for (auto v : vertices_range(g))
        P_.insert(v);
    get_pivot_adj(get<1>(stack.back()),
                  get<2>(stack.back()),
                  get<3>(stack.back()));
    get<4>(stack.back()) = P_.begin();

    while (!stack.empty())
    {
        auto& top = stack.back();
        vset_t& R = get<0>(top);
        vset_t& P = get<1>(top);
        vset_t& X = get<2>(top);
        vset_t& u_adj = get<3>(top);
        auto& viter = get<4>(top);

        if (P.empty())
        {
            if (X.empty() && R.size() > 1)
                vis(R);

            stack.pop_back();
            continue;
        }

        while (viter != P.end() && u_adj.find(*viter) != u_adj.end())
            ++viter;

        if (viter == P.end())
        {
            stack.pop_back();
            continue;
        }

        auto v = *viter;

        stack.emplace_back();
        auto& ntop = stack.back();
        auto& nR = get<0>(ntop);
        auto& nP = get<1>(ntop);
        auto& nX = get<2>(ntop);
        auto& nu_adj = get<3>(ntop);
        auto& nviter = get<4>(ntop);

        if (!R.empty())
            nR.insert(R.begin(), R.end());
        nR.insert(v);
        for (auto w : out_neighbors_range(v, g))
        {
            if (w == v)
                continue;
            if (P.find(w) != P.end())
                nP.insert(w);
            if (X.find(w) != X.end())
                nX.insert(w);
        }

        if (!(nP.empty() && nX.empty()))
            get_pivot_adj(nP, nX, nu_adj);

        nviter = nP.begin();

        X.insert(v);
        P.erase(viter++);
    }
}

} // graph_tool namespace

#endif // GRAPH_MAXIMAL_CLIQUES_HH
