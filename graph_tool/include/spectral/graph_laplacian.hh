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

#ifndef GRAPH_LAPLACIAN_HH
#define GRAPH_LAPLACIAN_HH

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"

namespace graph_tool
{
using namespace boost;

enum deg_t
{
    IN_DEG,
    OUT_DEG,
    TOTAL_DEG
};

template <class Graph, class Weight, class EdgeSelector>
typename property_traits<Weight>::value_type
sum_degree(Graph& g, typename graph_traits<Graph>::vertex_descriptor v,
           Weight w, EdgeSelector)
{
    typename property_traits<Weight>::value_type sum = 0;
    typename EdgeSelector::type e, e_end;
    for(tie(e, e_end) = EdgeSelector::get_edges(v, g); e != e_end; ++e)
        sum += get(w, *e);
    return sum;
}

template <class Graph, class EdgeSelector, class Val>
double
sum_degree(Graph& g, typename graph_traits<Graph>::vertex_descriptor v,
           UnityPropertyMap<Val, GraphInterface::edge_t>, all_edges_iteratorS<Graph>)
{
    return total_degreeS()(v, g);
}

template <class Graph, class EdgeSelector, class Val>
double
sum_degree(Graph& g, typename graph_traits<Graph>::vertex_descriptor v,
           UnityPropertyMap<Val, GraphInterface::edge_t>, in_edge_iteratorS<Graph>)
{
    return in_degreeS()(v, g);
}

template <class Graph, class EdgeSelector, class Val>
double
sum_degree(Graph& g, typename graph_traits<Graph>::vertex_descriptor v,
           UnityPropertyMap<Val, GraphInterface::edge_t>, out_edge_iteratorS<Graph>)
{
    return out_degreeS()(v, g);
}

struct get_laplacian
{
    template <class Graph, class Index, class Weight>
    void operator()(const Graph& g, Index index, Weight weight, deg_t deg,
                    double r,
                    multi_array_ref<double,1>& data,
                    multi_array_ref<int32_t,1>& i,
                    multi_array_ref<int32_t,1>& j) const
    {
        int pos = 0;
        for (const auto& e : edges_range(g))
        {
            if (source(e, g) == target(e, g))
                continue;

            data[pos] = -get(weight, e) * r;
            i[pos] = get(index, target(e, g));
            j[pos] = get(index, source(e, g));

            ++pos;
            if (!graph_tool::is_directed(g))
            {
                data[pos] = -get(weight, e) * r;
                i[pos] = get(index, source(e, g));
                j[pos] = get(index, target(e, g));
                ++pos;
            }
        }

        double rr = r*r - 1;
        for (auto v : vertices_range(g))
        {
            double k = 0;
            switch (deg)
            {
            case OUT_DEG:
                k = sum_degree(g, v, weight, out_edge_iteratorS<Graph>());
                break;
            case IN_DEG:
                k = sum_degree(g, v, weight, in_edge_iteratorS<Graph>());
                break;
            case TOTAL_DEG:
                k = sum_degree(g, v, weight, all_edges_iteratorS<Graph>());
            }
            data[pos] = k + rr;
            i[pos] = j[pos] = get(index, v);
            ++pos;
        }

    }
};

template <class Graph, class Vindex, class Weight, class Deg, class V>
void lap_matvec(Graph& g, Vindex index, Weight w, Deg d, double r, V& x, V& ret)
{
    double rr = r*r - 1;
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
                     if (u == v)
                         continue;
                     auto w_e = get(w, e);
                     y += r * w_e * x[get(index, u)];
                 }
             }
             else
             {
                 for (auto u : in_or_out_neighbors_range(v, g))
                 {
                     if (u == v)
                         continue;
                     y += r * x[get(index, u)];
                 }
             }
             ret[get(index, v)] = (d[v] + rr) * x[get(index, v)] - y;
         });
}

template <class Graph, class Vindex, class Weight, class Deg, class Mat>
void lap_matmat(Graph& g, Vindex index, Weight w, Deg d, double r, Mat& x, Mat& ret)
{
    double rr = r*r - 1;
    size_t M = x.shape()[1];
    parallel_vertex_loop
        (g,
         [&](auto v)
         {
             auto vi = get(index, v);
             auto y = ret[vi];
             if constexpr (!std::is_same_v<Weight, UnityPropertyMap<double, GraphInterface::edge_t>>)
             {
                 for (auto e : in_or_out_edges_range(v, g))
                 {
                     auto u = target(e, g);
                     if (u == v)
                         continue;
                     auto w_e = get(w, e);
                     auto ui = get(index, u);
                     for (size_t i = 0; i < M; ++i)
                         y[i] += r * w_e * x[ui][i];
                 }
             }
             else
             {
                 for (auto u : in_or_out_neighbors_range(v, g))
                 {
                     if (u == v)
                         continue;
                     auto ui = get(index, u);
                     for (size_t i = 0; i < M; ++i)
                         y[i] += r * x[ui][i];
                 }
             }
             for (size_t i = 0; i < M; ++i)
                 ret[vi][i] = (d[v] + rr) * x[vi][i] - y[i];
         });
}


struct get_norm_laplacian
{
    template <class Graph, class Index, class Weight>
    void operator()(const Graph& g, Index index, Weight weight, deg_t deg,
                    multi_array_ref<double,1>& data,
                    multi_array_ref<int32_t,1>& i,
                    multi_array_ref<int32_t,1>& j) const
    {
        int pos = 0;
        std::vector<double> degs(num_vertices(g));
        for (auto v : vertices_range(g))
        {
            double k = 0;
            switch (deg)
            {
            case OUT_DEG:
                k = sum_degree(g, v, weight, out_edge_iteratorS<Graph>());
                break;
            case IN_DEG:
                k = sum_degree(g, v, weight, in_edge_iteratorS<Graph>());
                break;
            case TOTAL_DEG:
                k = sum_degree(g, v, weight, all_edges_iteratorS<Graph>());
            }
            degs[v] = sqrt(k);
        }

        for (auto v : vertices_range(g))
        {
            double ks = degs[v];
            for(const auto& e : out_edges_range(v, g))
            {
                if (source(e, g) == target(e, g))
                    continue;
                double kt = degs[target(e, g)];
                if (ks * kt > 0)
                    data[pos] = -get(weight, e) / (ks * kt);
                i[pos] = get(index, target(e, g));
                j[pos] = get(index, source(e, g));

                ++pos;
            }

            if (ks > 0)
                data[pos] = 1;
            i[pos] = j[pos] = get(index, v);
            ++pos;
        }

    }
};

template <class Graph, class Vindex, class Weight, class Deg, class V>
void nlap_matvec(Graph& g, Vindex index, Weight w, Deg id, V& x, V& ret)
{
    parallel_vertex_loop
        (g,
         [&](auto v)
         {
             auto vi = get(index, v);
             std::remove_reference_t<decltype(ret[v])> y = 0;
             if constexpr (!std::is_same_v<Weight, UnityPropertyMap<double, GraphInterface::edge_t>>)
             {
                 for (auto e : in_or_out_edges_range(v, g))
                 {
                     auto u = target(e, g);
                     if (u == v)
                         continue;
                     auto w_e = get(w, e);
                     y += w_e * x[get(index, u)] * id[u];
                 }
             }
             else
             {
                 for (auto u : in_or_out_neighbors_range(v, g))
                 {
                     if (u == v)
                         continue;
                     y += x[get(index, u)] * id[u];
                 }
             }
             if (id[v] > 0)
                 ret[vi] = x[vi] - y * id[v];
         });
}

template <class Graph, class Vindex, class Weight, class Deg, class Mat>
void nlap_matmat(Graph& g, Vindex index, Weight w, Deg id, Mat& x, Mat& ret)
{
    size_t M = x.shape()[1];
    parallel_vertex_loop
        (g,
         [&](auto v)
         {
             auto vi = get(index, v);
             auto y = ret[vi];
             if constexpr (!std::is_same_v<Weight, UnityPropertyMap<double, GraphInterface::edge_t>>)
             {
                 for (auto e : in_or_out_edges_range(v, g))
                 {
                     auto u = target(e, g);
                     if (u == v)
                         continue;
                     auto w_e = get(w, e);
                     auto ui = get(index, u);
                     for (size_t i = 0; i < M; ++i)
                         y[i] += w_e * x[ui][i] * id[u];
                 }
             }
             else
             {
                 for (auto u : in_or_out_neighbors_range(v, g))
                 {
                     if (u == v)
                         continue;
                     auto ui = get(index, u);
                     for (size_t i = 0; i < M; ++i)
                         y[i] += x[ui][i] * id[u];
                 }
             }

             if (id[v] > 0)
             {
                 for (size_t i = 0; i < M; ++i)
                     y[i] = x[vi][i] - y[i] * id[v];
             }
         });
}


} // namespace graph_tool

#endif // GRAPH_LAPLACIAN_HH
