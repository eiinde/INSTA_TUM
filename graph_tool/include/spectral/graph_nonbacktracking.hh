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

#ifndef GRAPH_NONBACKTRACKING_MATRIX_HH
#define GRAPH_NONBACKTRACKING_MATRIX_HH

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"

namespace graph_tool
{
using namespace boost;

template <class Graph, class Index>
void get_nonbacktracking(Graph& g, Index index,
                         std::vector<int64_t>& i,
                         std::vector<int64_t>& j)
{
    for (auto u : vertices_range(g))
    {
        for (auto e1 : out_edges_range(u, g))
        {
            auto v = target(e1, g);
            int64_t idx1 = index[e1];
            if (!graph_tool::is_directed(g))
                idx1 = (idx1 << 1) + (u > v);
            for (auto e2 : out_edges_range(v, g))
            {
                auto w = target(e2, g);
                if (w == u)
                    continue;
                int64_t idx2 = index[e2];
                if (!graph_tool::is_directed(g))
                    idx2 = (idx2 << 1) + (v > w);
                i.push_back(idx1);
                j.push_back(idx2);
            }
        }
    }
}

template <bool transpose, class Graph, class EIndex, class V>
void nbt_matvec(Graph& g, EIndex eindex, V& x, V& ret)
{
    auto get_idx =
        [&](auto& e, bool reverse = false)
        {
            int64_t idx = get(eindex, e);
            if constexpr (!graph_tool::is_directed_::apply<Graph>::type::value)
            {
                size_t u = source(e, g);
                size_t v = target(e, g);
                if (reverse)
                    std::swap(u, v);
                if constexpr (!transpose)
                    idx = (idx << 1) + (u > v);
                else
                    idx = (idx << 1) + (v > u);
            }
            return idx;
        };


    parallel_edge_loop
        (g,
         [&](const auto& e)
         {
             size_t u = source(e, g);
             size_t v = target(e, g);

             int64_t idx = get_idx(e);

             for (const auto& e2 : out_edges_range(v, g))
             {
                 auto w = target(e2, g);
                 if (w == u || w == v)
                     continue;
                 int64_t idx2 = get_idx(e2);
                 ret[idx] += x[idx2];
             }

             idx = get_idx(e, true);

             for (const auto& e2 : out_edges_range(u, g))
             {
                 auto w = target(e2, g);
                 if (w == u || w == v)
                     continue;
                 int64_t idx2 = get_idx(e2);
                 ret[idx] += x[idx2];
             }

         });
}

template <bool transpose, class Graph, class EIndex, class V>
void nbt_matmat(Graph& g, EIndex eindex, V& x, V& ret)
{
    auto get_idx =
        [&](auto& e, bool reverse = false)
        {
            int64_t idx = get(eindex, e);
            if constexpr (!graph_tool::is_directed_::apply<Graph>::type::value)
            {
                size_t u = source(e, g);
                size_t v = target(e, g);
                if (reverse)
                    std::swap(u, v);
                if constexpr (!transpose)
                    idx = (idx << 1) + (u > v);
                else
                    idx = (idx << 1) + (v > u);
            }
            return idx;
        };


    size_t k = x.shape()[1];
    parallel_edge_loop
        (g,
         [&](const auto& e)
         {
             size_t u = source(e, g);
             size_t v = target(e, g);

             int64_t idx = get_idx(e);

             for (const auto& e2 : out_edges_range(v, g))
             {
                 auto w = target(e2, g);
                 if (w == u || w == v)
                     continue;
                 int64_t idx2 = get_idx(e2);
                 for (size_t i = 0; i < k; ++i)
                     ret[idx][i] += x[idx2][i];
             }

             idx = get_idx(e, true);

             for (const auto& e2 : out_edges_range(u, g))
             {
                 auto w = target(e2, g);
                 if (w == u || w == v)
                     continue;
                 int64_t idx2 = get_idx(e2);
                 for (size_t i = 0; i < k; ++i)
                     ret[idx][i] += x[idx2][i];
             }

         });
}

template <class Graph>
void get_compact_nonbacktracking(Graph& g,
                                 std::vector<int64_t>& i,
                                 std::vector<int64_t>& j,
                                 std::vector<double>& x)
{
    for (auto e : edges_range(g))
    {
        auto u = source(e, g);
        auto v = target(e, g);
        i.push_back(u);
        j.push_back(v);
        x.push_back(1);

        i.push_back(v);
        j.push_back(u);
        x.push_back(1);
    }

    auto N = num_vertices(g);

    for (auto u : vertices_range(g))
    {
        int32_t k = out_degree(u, g);
        auto idx = u + N;

        i.push_back(u);
        j.push_back(idx);
        x.push_back(-1);

        i.push_back(idx);
        j.push_back(u);
        x.push_back(k-1);
    }
}

template <bool transpose, class Graph, class VIndex, class V>
void cnbt_matvec(Graph& g, VIndex vindex, V& x, V& ret)
{
    size_t N = HardNumVertices()(g);
    parallel_vertex_loop
        (g,
         [&](const auto& v)
         {
             size_t i = get(vindex, v);
             auto& y = ret[i];
             size_t k = 0;
             for (const auto& e : out_edges_range(v, g))
             {
                 auto u = target(e, g);
                 size_t j = get(vindex, u);
                 y += x[j];
                 ++k;
             }

             if (k > 0)
             {
                 if constexpr (!transpose)
                 {
                     ret[i] -= x[i + N];
                     ret[i + N] = (k - 1) * x[i];
                 }
                 else
                 {
                     ret[i + N] -= x[i];
                     ret[i] = (k - 1) * x[i + N];
                 }
             }
         });
}

template <bool transpose, class Graph, class VIndex, class V>
void cnbt_matmat(Graph& g, VIndex vindex, V& x, V& ret)
{
    size_t k = x.shape()[1];
    size_t N = HardNumVertices()(g);
    parallel_vertex_loop
        (g,
         [&](const auto& v)
         {
             size_t i = get(vindex, v);
             auto y = ret[i];
             size_t d = 0;
             for (const auto& e : out_edges_range(v, g))
             {
                 auto u = target(e, g);
                 size_t j = get(vindex, u);
                 for (size_t l = 0; l < k; ++l)
                     y[l] += x[j][l];
                 ++d;
             }

             if (d > 0)
             {
                 --d;
                 if constexpr (!transpose)
                 {
                     for (size_t l = 0; l < k; ++l)
                     {
                         ret[i][l] -= x[i + N][l];
                         ret[i + N][l] = d * x[i][l];
                     }
                 }
                 else
                 {
                     for (size_t l = 0; l < k; ++l)
                     {
                         ret[i + N][l] -= x[i][l];
                         ret[i][l] = d * x[i + N][l];
                     }
                 }
             }
         });
}


} // namespace graph_tool

#endif // GRAPH_NONBACKTRACKING_MATRIX_HH
