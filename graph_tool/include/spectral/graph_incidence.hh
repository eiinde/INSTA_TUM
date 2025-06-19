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

#ifndef GRAPH_INCIDENCE_HH
#define GRAPH_INCIDENCE_HH

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"

namespace graph_tool
{
using namespace boost;

struct get_incidence
{
    template <class Graph, class VIndex, class EIndex>
    void operator()(Graph& g, VIndex vindex, EIndex eindex,
                    multi_array_ref<double,1>& data,
                    multi_array_ref<int32_t,1>& i,
                    multi_array_ref<int32_t,1>& j) const
    {
        int pos = 0;
        for (auto v : vertices_range(g))
        {
            for (const auto& e : out_edges_range(v, g))
            {
                if constexpr (graph_tool::is_directed_::apply<Graph>::type::value)
                    data[pos] = -1;
                else
                    data[pos] = 1;
                i[pos] = get(vindex, v);
                j[pos] = get(eindex, e);
                ++pos;
            }

            if constexpr (graph_tool::is_directed_::apply<Graph>::type::value)
            {
                for (const auto& e : in_edges_range(v, g))
                {
                    data[pos] = 1;
                    i[pos] = get(vindex, v);
                    j[pos] = get(eindex, e);
                    ++pos;
                }
            }
        }
    }
};

template <class Graph, class Vindex, class EIndex, class V>
void inc_matvec(Graph& g, Vindex vindex, EIndex eindex, V& x, V& ret, bool transpose)
{
    if (!transpose)
    {
        parallel_vertex_loop
            (g,
             [&](auto v)
             {
                 auto& y = ret[get(vindex, v)];
                 for (const auto& e : out_edges_range(v, g))
                 {
                    auto u = eindex[e];
                    if constexpr (graph_tool::is_directed_::apply<Graph>::type::value)
                        y -= x[u];
                    else
                        y += x[u];
                 }

                 if constexpr (graph_tool::is_directed_::apply<Graph>::type::value)
                 {
                    for (const auto& e : in_edges_range(v, g))
                    {
                        auto u = eindex[e];
                        y += x[u];
                    }
                 }
             });
    }
    else
    {
        parallel_edge_loop
            (g,
             [&](const auto& e)
             {
                 auto u = eindex[e];
                 if constexpr (graph_tool::is_directed_::apply<Graph>::type::value)
                     ret[u] = x[get(vindex, target(e, g))] - x[get(vindex, source(e, g))];
                 else
                     ret[u] = x[get(vindex, target(e, g))] + x[get(vindex, source(e, g))];
             });
    }
}

template <class Graph, class Vindex, class Eindex, class M>
void inc_matmat(Graph& g, Vindex vindex, Eindex eindex, M& x, M& ret, bool transpose)
{
    size_t k = x.shape()[1];
    if (!transpose)
    {
        parallel_vertex_loop
            (g,
             [&](auto v)
             {
                 auto y = ret[get(vindex, v)];
                 for (const auto& e : out_edges_range(v, g))
                 {
                    auto u = eindex[e];
                    for (size_t i = 0; i < k; ++i)
                    {
                        if constexpr (graph_tool::is_directed_::apply<Graph>::type::value)
                            y[i] -= x[u][i];
                        else
                            y[i] += x[u][i];
                    }
                 }

                 if constexpr (graph_tool::is_directed_::apply<Graph>::type::value)
                 {
                    for (const auto& e : in_edges_range(v, g))
                    {
                        auto u = eindex[e];
                        for (size_t i = 0; i < k; ++i)
                            y[i] += x[u][i];
                    }
                 }
             });
    }
    else
    {
        parallel_edge_loop
            (g,
             [&](const auto& e)
             {
                 auto u = eindex[e];
                 auto s = get(vindex, source(e, g));
                 auto t = get(vindex, target(e, g));
                 for (size_t i = 0; i < k; ++i)
                 {
                     if constexpr (graph_tool::is_directed_::apply<Graph>::type::value)
                         ret[u][i] = x[t][i] - x[s][i];
                     else
                         ret[u][i] = x[t][i] + x[s][i];
                 }
             });
    }
}


} // namespace graph_tool

#endif // GRAPH_INCIDENCE_HH
