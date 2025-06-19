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

#ifndef GRAPH_ADJACENCY_MATRIX_HH
#define GRAPH_ADJACENCY_MATRIX_HH

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"

namespace graph_tool
{
using namespace boost;

struct get_adjacency
{
    template <class Graph, class Index, class Weight>
    void operator()(Graph& g, Index index, Weight weight,
                    multi_array_ref<double,1>& data,
                    multi_array_ref<int,1>& i,
                    multi_array_ref<int,1>& j) const
    {
        int pos = 0;
        for (const auto& e : edges_range(g))
        {
            data[pos] = get(weight, e);
            i[pos] = get(index, target(e, g));
            j[pos] = get(index, source(e, g));

            ++pos;
            if (!graph_tool::is_directed(g))
            {
                data[pos] = get(weight, e);
                i[pos] = get(index, source(e, g));
                j[pos] = get(index, target(e, g));
                ++pos;
            }
        }
    }
};

template <class Graph, class Vindex, class Weight, class V>
void adj_matvec(Graph& g, Vindex index, Weight w, V& x, V& ret)
{
    parallel_vertex_loop
        (g,
         [&](auto v)
         {
             size_t i = get(index, v);
             std::remove_reference_t<decltype(ret[i])> y = 0;
             if constexpr (!std::is_same_v<Weight, UnityPropertyMap<double, GraphInterface::edge_t>>)
             {
                 for (auto e : in_or_out_edges_range(v, g))
                     y += get(w, e) * x[get(index, target(e, g))];
             }
             else
             {
                 for (auto u : in_or_out_neighbors_range(v, g))
                     y += x[get(index, u)];
             }
             ret[i] = y;
         });
}

template <class Graph, class Vindex, class Weight, class M>
void adj_matmat(Graph& g, Vindex index, Weight w, M& x, M& ret)
{
    size_t k = x.shape()[1];
    parallel_vertex_loop
        (g,
         [&](auto v)
         {
             size_t i = get(index, v);
             auto y = ret[i];
             if constexpr (!std::is_same_v<Weight, UnityPropertyMap<double, GraphInterface::edge_t>>)
             {
                 for (auto e : in_or_out_edges_range(v, g))
                 {
                     auto w_e = get(w, e);
                     for (size_t l = 0; l < k; ++l)
                         y[l] += w_e * x[get(index, target(e, g))][l];
                 }
             }
             else
             {
                 for (auto u : in_or_out_neighbors_range(v, g))
                     for (size_t l = 0; l < k; ++l)
                         y[l] += x[get(index, u)][l];
             }

         });
}

} // namespace graph_tool

#endif // GRAPH_ADJACENCY_MATRIX_HH
