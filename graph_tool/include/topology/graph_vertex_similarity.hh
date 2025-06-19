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

#ifndef GRAPH_VERTEX_SIMILARITY_HH
#define GRAPH_VERTEX_SIMILARITY_HH

#include "graph_util.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class Graph, class Vertex, class Mark, class Weight>
auto common_neighbors(Vertex u, Vertex v, Mark& mark, Weight& weight, Graph& g)
{
    typename property_traits<Weight>::value_type count = 0, ku = 0, kv = 0;
    for (auto e : out_edges_range(u, g))
    {
        auto w = weight[e];
        mark[target(e, g)] += w;
        ku += w;
    }
    for (auto e : out_edges_range(v, g))
    {
        auto w = weight[e];
        auto dw = std::min(w, mark[target(e, g)]);
        mark[target(e, g)] -= dw;
        count += dw;
        kv += w;
    }
    for (auto w : adjacent_vertices_range(u, g))
        mark[w] = 0;
    return std::make_tuple(count, ku, kv);
}

template <class Graph, class Vertex, class Mark, class Weight>
double dice(Vertex u, Vertex v, Mark& mark, Weight& weight, Graph& g)
{
    typename property_traits<Weight>::value_type count = 0, ku = 0, kv = 0;
    std::tie(count, ku, kv) = common_neighbors(u, v, mark, weight, g);
    return 2 * count / double(ku + kv);
}

template <class Graph, class Vertex, class Mark, class Weight>
double salton(Vertex u, Vertex v, Mark& mark, Weight& weight, Graph& g)
{
    typename property_traits<Weight>::value_type count = 0, ku = 0, kv = 0;
    std::tie(count, ku, kv) = common_neighbors(u, v, mark, weight, g);
    return count / sqrt(ku * kv);
}

template <class Graph, class Vertex, class Mark, class Weight>
double hub_promoted(Vertex u, Vertex v, Mark& mark, Weight& weight, Graph& g)
{
    typename property_traits<Weight>::value_type count = 0, ku = 0, kv = 0;
    std::tie(count, ku, kv) = common_neighbors(u, v, mark, weight, g);
    return count / double(std::max(ku, kv));
}

template <class Graph, class Vertex, class Mark, class Weight>
double hub_suppressed(Vertex u, Vertex v, Mark& mark, Weight& weight, Graph& g)
{
    typename property_traits<Weight>::value_type count = 0, ku = 0, kv = 0;
    std::tie(count, ku, kv) = common_neighbors(u, v, mark, weight, g);
    return count / double(std::min(ku, kv));
}

template <class Graph, class Vertex, class Mark, class Weight>
double leicht_holme_newman(Vertex u, Vertex v, Mark& mark, Weight& weight, Graph& g)
{
    typename property_traits<Weight>::value_type count = 0, ku = 0, kv = 0;
    std::tie(count, ku, kv) = common_neighbors(u, v, mark, weight, g);
    return count / double(ku * kv);
}

template <class Graph, class Vertex, class Mark, class Weight>
double jaccard(Vertex u, Vertex v, Mark& mark, Weight& weight, Graph& g)
{
    typename property_traits<Weight>::value_type count = 0, total = 0;
    for (auto e : out_edges_range(u, g))
    {
        auto w = weight[e];
        mark[target(e, g)] += w;
        total += w;
    }

    for (auto e : out_edges_range(v, g))
    {
       auto w = weight[e];
       auto dw = std::min(w, mark[target(e, g)]);
       count += dw;
       mark[target(e, g)] -= dw;
       total += w - dw;
    }

    for (auto w : adjacent_vertices_range(u, g))
        mark[w] = 0;
    return count / double(total);
}

template <class Graph, class Vertex, class Mark, class Weight>
double inv_log_weighted(Vertex u, Vertex v, Mark& mark, Weight& weight, Graph& g)
{
    double count = 0;
    for (auto e : out_edges_range(u, g))
        mark[target(e, g)] += weight[e];
    for (auto e : out_edges_range(v, g))
    {
       auto w = weight[e];
       auto dw = std::min(w, mark[target(e, g)]);
       if (mark[target(e, g)] > 0)
       {
           if (graph_tool::is_directed(g))
               count += dw / log(in_degreeS()(target(e, g), g, weight));
           else
               count += dw / log(out_degreeS()(target(e, g), g, weight));
       }
       mark[target(e, g)] -= dw;
    }
    for (auto w : adjacent_vertices_range(u, g))
        mark[w] = 0;
    return count;
}

template <class Graph, class Vertex, class Mark, class Weight>
double r_allocation(Vertex u, Vertex v, Mark& mark, Weight& weight, Graph& g)
{
    double count = 0;
    for (auto e : out_edges_range(u, g))
        mark[target(e, g)] += weight[e];
    for (auto e : out_edges_range(v, g))
    {
       auto w = weight[e];
       auto dw = std::min(w, mark[target(e, g)]);
       if (mark[target(e, g)] > 0)
       {
           if (graph_tool::is_directed(g))
               count += dw / double(in_degreeS()(target(e, g), g, weight));
           else
               count += dw / double(out_degreeS()(target(e, g), g, weight));
       }
       mark[target(e, g)] -= dw;
    }
    for (auto w : adjacent_vertices_range(u, g))
        mark[w] = 0;
    return count;
}

template <class Graph, class VMap, class Sim, class Weight>
void all_pairs_similarity(Graph& g, VMap s, Sim&& f, Weight& weight)
{
    vector<typename property_traits<Weight>::value_type>
        mask(num_vertices(g));
    #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh()) \
        firstprivate(mask)
    parallel_vertex_loop_no_spawn
        (g,
         [&](auto v)
         {
             s[v].resize(num_vertices(g));
             for (auto w : vertices_range(g))
                 s[v][w] = f(v, w, mask, weight);
         });
}

template <class Graph, class Vlist, class Slist, class Sim, class Weight>
void some_pairs_similarity(Graph& g, Vlist& vlist, Slist& slist, Sim&& f,
                           Weight& weight)
{
    vector<typename property_traits<Weight>::value_type>
        mark(num_vertices(g));
    #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh()) \
        firstprivate(mark)
    parallel_loop_no_spawn
        (vlist,
         [&](size_t i, const auto& val)
         {
             size_t u = val[0];
             size_t v = val[1];
             slist[i] = f(u, v, mark, weight);
         });
}

} // graph_tool namespace

#endif // GRAPH_VERTEX_SIMILARITY_HH
