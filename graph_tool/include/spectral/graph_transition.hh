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

#ifndef GRAPH_TRANSITION_HH
#define GRAPH_TRANSITION_HH

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"

namespace graph_tool
{

using namespace boost;

template <class Graph, class Weight>
typename property_traits<Weight>::value_type
sum_degree(Graph& g, typename graph_traits<Graph>::vertex_descriptor v,
           const Weight& w)
{
    typename property_traits<Weight>::value_type sum = 0;
    for(const auto& e : out_edges_range(v, g))
        sum += get(w, e);
    return sum;
}

template <class Graph, class Type>
size_t
sum_degree(Graph& g, typename graph_traits<Graph>::vertex_descriptor v,
           const UnityPropertyMap<Type,GraphInterface::edge_t>&)
{
    return out_degreeS()(v, g);
}

struct get_transition
{
    template <class Graph, class Index, class Weight>
    void operator()(const Graph& g, Index index, Weight weight,
                    multi_array_ref<double,1>& data,
                    multi_array_ref<int32_t,1>& i,
                    multi_array_ref<int32_t,1>& j) const
    {
        int pos = 0;
        for (auto v: vertices_range(g))
        {
            auto k = sum_degree(g, v, weight);
            for (const auto& e: out_edges_range(v, g))
            {
                data[pos] = double(weight[e]) / k;
                j[pos] = get(index, source(e, g));
                i[pos] = get(index, target(e, g));
                ++pos;
            }
        }
    }
};

template <bool transpose, class Graph, class Vindex, class Weight, class Deg, class V>
void trans_matvec(Graph& g, Vindex vindex, Weight w, Deg id, V& x, V& ret)
{
    parallel_vertex_loop
        (g,
         [&](auto v)
         {
             std::remove_reference_t<decltype(ret[v])> y = 0;
             if constexpr (!std::is_same_v<Weight, UnityPropertyMap<double, GraphInterface::edge_t>>)
             {
                 for (auto e : in_or_out_edges_range(v, g))
                 {
                     auto u = target(e, g);
                     auto w_e = get(w, e);
                     auto ui = get(vindex, u);
                     if constexpr (!transpose)
                         y += w_e * x[ui] * id[u];
                     else
                         y += w_e * x[ui];
                 }
             }
             else
             {
                 for (auto u : in_or_out_neighbors_range(v, g))
                 {
                     auto ui = get(vindex, u);
                     if constexpr (!transpose)
                         y += x[ui] * id[u];
                     else
                         y += x[ui];
                 }
             }
             if constexpr (transpose)
                 y *= id[v];
             ret[get(vindex, v)] = y;
         });
}


template <bool transpose, class Graph, class Vindex, class Weight, class Deg, class Mat>
void trans_matmat(Graph& g, Vindex vindex, Weight w, Deg id, Mat& x, Mat& ret)
{
    size_t M = x.shape()[1];
    parallel_vertex_loop
        (g,
         [&](auto v)
         {
             auto vi = get(vindex, v);
             auto y = ret[vi];
             if constexpr (!std::is_same_v<Weight, UnityPropertyMap<double, GraphInterface::edge_t>>)
             {
                 for (auto e : in_or_out_edges_range(v, g))
                 {
                     auto u = target(e, g);
                     auto w_e = get(w, e);
                     auto ui = get(vindex, u);
                     for (size_t i = 0; i < M; ++i)
                     {
                         if constexpr (!transpose)
                             y[i] += w_e * x[ui][i] * id[u];
                         else
                             y[i] += w_e * x[ui][i];
                     }
                 }
             }
             else
             {
                 for (auto u : in_or_out_neighbors_range(v, g))
                 {
                     auto ui = get(vindex, u);
                     for (size_t i = 0; i < M; ++i)
                     {
                         if constexpr (!transpose)
                             y[i] += x[ui][i] * id[u];
                         else
                             y[i] += x[ui][i];
                     }
                 }
             }
             if constexpr (transpose)
             {
                 for (size_t i = 0; i < M; ++i)
                     y[i] *= id[v];
             }
         });
}

} // namespace graph_tool

#endif // GRAPH_TRANSITION_HH
