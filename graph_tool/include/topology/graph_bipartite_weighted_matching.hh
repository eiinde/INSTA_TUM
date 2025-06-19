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

#ifndef GRAPH_BIPARTITE_WEIGHTED_HH
#define GRAPH_BIPARTITE_WEIGHTED_HH

#include <stack>
#include <boost/graph/breadth_first_search.hpp>
#include "idx_map.hh"

namespace graph_tool
{
using namespace boost;

template <class Graph, class Partition, class Weight, class Mate>
void maximum_bipartite_weighted_perfect_matching(Graph& g, Partition&& partition,
                                                 Weight&& weight, Mate&& mate)
{
    typedef typename graph_traits<Graph>::vertex_descriptor vertex_t;
    typedef typename property_traits<std::remove_reference_t<Weight>>::value_type weight_t;
    adj_list<> G;
    G.set_keep_epos(true);

    typename vprop_map_t<vertex_t>::type::unchecked_t
        vmap(num_vertices(g)), rvmap(num_vertices(g));

    typename vprop_map_t<weight_t>::type::unchecked_t y(num_vertices(g));

    vertex_t null_vertex = graph_traits<Graph>::null_vertex();

    std::vector<vertex_t> S, T;
    auto S_val = partition[*vertices(g).first];
    for (auto v : vertices_range(g))
    {
        if (partition[v] == S_val)
            S.push_back(v);
        else
            T.push_back(v);
        auto u = add_vertex(G);
        vmap[v] = u;
        rvmap[u] = v;
    }

    weight_t max_weight = std::numeric_limits<weight_t>::lowest();
    for (auto e : edges_range(g))
        max_weight = std::max(max_weight, get(weight, e));
    for (auto v : vertices_range(g))
        y[v] = max_weight + 1;

    auto is_tight =
        [&](const auto& e)
        {
            auto res = (y[source(e, g)] + y[target(e, g)]) - get(weight, e);
            if constexpr (std::is_floating_point_v<weight_t>)
                return std::abs(res) < sqrt(10 * std::numeric_limits<weight_t>::epsilon());
            return res == 0;
        };

    for (auto e : edges_range(g))
    {
        auto u = vmap[source(e, g)];
        auto v = vmap[target(e, g)];
        if (partition[source(e, g)] != S_val)
            std::swap(u, v);
        if (is_tight(e))
            add_edge(u, v, G); // S -> T
    }

    typename vprop_map_t<boost::default_color_type>::type::unchecked_t
        Z(num_vertices(G));
    typename vprop_map_t<vertex_t>::type pred(num_vertices(G));

    size_t M = 0;

    assert(S.size() == T.size());
    while (M < std::min(S.size(), T.size()))
    {
        for (auto v : vertices_range(G))
        {
            Z[v] = color_traits<default_color_type>::white();
            pred[v] = v;
        }

        std::stack<vertex_t> Q;
        auto vis = make_bfs_visitor(record_predecessors(pred, on_tree_edge()));
        for (auto v_g : S)
        {
            auto v = vmap[v_g];
            if (in_degree(v, G) > 0 ||
                Z[v] == color_traits<default_color_type>::black())
                continue;
            breadth_first_visit(G, v, Q, vis, Z);
        }

        bool reversed_path = false;
        for (auto v : T)
        {
            auto v_G = vmap[v];
            if (out_degree(v_G, G) > 0)
                continue;
            //v is in R_T
            if (Z[v_G] == color_traits<default_color_type>::black())
            {
                //there is an augmenting path from S to T; reverse it in G and break;
                vertex_t u = v_G;
                while (pred[u] != u)
                {
                    auto w = pred[u];
                    auto e = edge(w, u, G);
                    assert(e.second);
                    remove_edge(e.first, G);
                    add_edge(u, w, G);
                    u = w;
                }
                M++;
                reversed_path = true;
                break;
            }
        }

        if (reversed_path)
            continue;
        // R_T ∩ Z is empty

        weight_t delta = std::numeric_limits<weight_t>::max();
        for (auto v : S)
        {
            if (Z[vmap[v]] != color_traits<default_color_type>::black())
                continue;
            for (auto e : out_edges_range(v, g))
            {
                auto u = target(e, g);
                if (Z[vmap[u]] == color_traits<default_color_type>::black())
                    continue;
                weight_t res = (y[v] + y[u]) - get(weight, e);
                if (res < delta)
                    delta = res;
            }
        }

        assert(delta > 0);

        for (auto v : S)
        {
            if (Z[vmap[v]] != color_traits<default_color_type>::black())
                continue;
            y[v] -= delta;
        }

        for (auto v : T)
        {
            if (Z[vmap[v]] != color_traits<default_color_type>::black())
                continue;
            y[v] += delta;
        }

        std::vector<graph_traits<adj_list<>>::edge_descriptor> redges;
        for (auto e_g : edges_range(G))
        {
            auto u = rvmap[source(e_g, G)];
            auto v = rvmap[target(e_g, G)];
            auto e = edge(u, v, g);
            if (!is_tight(e.first))
                redges.push_back(e_g);
        }

        for (auto& e : redges)
            remove_edge(e, G);

        for (auto e : edges_range(g))
        {
            auto u = vmap[source(e, g)];
            auto v = vmap[target(e, g)];
            if (partition[source(e, g)] != S_val)
                std::swap(u, v);
            if (is_tight(e) && !edge(u, v, G).second && !edge(v, u, G).second)
                add_edge(u, v, G); // S -> T
        }

#ifndef NDEBUG
        for (auto e : edges_range(G))
            assert(is_tight(edge(rvmap[source(e, G)], rvmap[target(e, G)],g).first));
        for (auto e : edges_range(g))
            if (is_tight(e))
                assert(edge(vmap[source(e, g)], vmap[target(e, g)], G).second ||
                       edge(vmap[target(e, g)], vmap[source(e, g)], G).second);

        for (auto e : edges_range(g))
            assert(y[source(e, g)] + y[target(e,g)] >= get(weight, e) ||
                   abs(y[source(e, g)] + y[target(e,g)] - get(weight, e)) < 1e-8);
#endif // NDEBUG
    }

#ifndef NDEBUG
    size_t m = 0;
    for (auto e : edges_range(g))
    {
        auto u = vmap[source(e, g)];
        auto v = vmap[target(e, g)];
        if (partition[source(e, g)] != S_val)
            std::swap(u, v);
        auto ne = edge(v, u, G);
        if (ne.second)
        {
            ++m;
            assert(out_degree(v, G) == 1);
            assert(in_degree(u, G) == 1);
            assert(is_tight(e));
        }
        else
        {
            assert(edge(u, v, G).second == is_tight(e));
        }
    }
    for (auto v : S)
    {
        auto u = vmap[v];
        assert(in_degree(u, G) == 1);
    }
    for (auto v : T)
    {
        auto u = vmap[v];
        assert(out_degree(u, G) == 1);
    }
    assert(m == M);
#endif // NDEBUG

    for (auto v : vertices_range(g))
        mate[v] = null_vertex;

    for (auto v : T)
    {
        auto u = vmap[v];
        for (auto w : out_neighbors_range(u, G))
        {
            auto w_v = rvmap[w];
            mate[v] = w_v;
            mate[w_v] = v;
            break;
        }
    }
}

template <class Graph, class Partition, class Weight, class Mate>
void maximum_bipartite_weighted_imperfect_matching(Graph& g,
                                                   Partition&& partition,
                                                   Weight&& weight, Mate&& mate)
{
    typedef typename graph_traits<Graph>::vertex_descriptor vertex_t;
    typedef typename property_traits<std::remove_reference_t<Weight>>::value_type oweight_t;
    typedef typename std::conditional<std::is_integral_v<oweight_t>,
                                      std::make_signed_t<std::conditional_t
                                                         <std::is_integral_v<oweight_t>,
                                                          oweight_t, int>>,
                                      oweight_t>::type weight_t;

    adj_list<> u_base;
    undirected_adaptor<adj_list<>> u(u_base);

    typedef typename property_traits<std::remove_reference_t<Partition>>::value_type pval_t;
    typename vprop_map_t<pval_t>::type u_partition;
    typename eprop_map_t<weight_t>::type u_weight;
    typename vprop_map_t<vertex_t>::type u_mate;
    typename vprop_map_t<bool>::type is_augmented;

    typename vprop_map_t<vertex_t>::type vmap, vmap2, rvmap;

    std::vector<vertex_t> S, T;
    pval_t S_val = partition[*vertices(g).first];
    pval_t T_val = S_val;
    for (auto v : vertices_range(g))
    {
        if (partition[v] == S_val)
        {
            S.push_back(v);
        }
        else
        {
            T.push_back(v);
            T_val = partition[v];
        }
        auto w = add_vertex(u);
        u_partition[w] = partition[v];
        vmap[v] = w;
        rvmap[w] = v;
    }

    for (auto v : vertices_range(g))
    {
        auto w = add_vertex(u);
        u_partition[w] = (partition[v] == S_val) ? T_val : S_val;
        vmap2[v] = w;
        rvmap[w] = v;
        is_augmented[w] = true;
    }

    weight_t max_weight = std::numeric_limits<weight_t>::min();
    if constexpr (std::is_integral_v<weight_t>)
        max_weight = 0;
    for (auto e : edges_range(g))
    {
        max_weight = std::max(max_weight,
            weight_t(std::abs(weight_t(get(weight, e)))));

        auto ne = add_edge(vmap[source(e, g)],
                           vmap[target(e, g)], u);
        u_weight[ne.first] = get(weight, e);

        ne = add_edge(vmap2[source(e, g)],
                      vmap2[target(e, g)], u);
        u_weight[ne.first] = get(weight, e);
    }

    if (S.size() < T.size())
        S.swap(T);

    // large to large
    for (auto v : S)
    {
        auto w1 = vmap[v];
        auto w2 = vmap2[v];
        auto e = add_edge(w1, w2, u);
        u_weight[e.first] = 0;
    }

    // small to small
    for (auto v : T)
    {
        auto w1 = vmap[v];
        auto w2 = vmap2[v];
        auto e = add_edge(w1, w2, u);
        u_weight[e.first] = -4 * ((max_weight + 1) * T.size());
    }

    maximum_bipartite_weighted_perfect_matching(u, u_partition.get_unchecked(),
                                                u_weight.get_unchecked(),
                                                u_mate.get_unchecked(num_vertices(u)));

    for (auto v : vertices_range(g))
    {
        auto w = vmap[v];
        auto x = u_mate[w];
        assert(x != graph_traits<Graph>::null_vertex());
        if (is_augmented[x])
            mate[v] = graph_traits<Graph>::null_vertex();
        else
            mate[v] = rvmap[x];
    }
}

template <class Graph, class Partition, class Weight, class Mate>
void maximum_bipartite_weighted_matching(Graph& g,
                                         Partition&& partition,
                                         Weight&& weight, Mate&& mate)
{
    typedef typename graph_traits<Graph>::vertex_descriptor vertex_t;
    typedef typename property_traits<std::remove_reference_t<Weight>>::value_type weight_t;

    adj_list<> u_base;
    undirected_adaptor<adj_list<>> u(u_base);

    typedef typename property_traits<std::remove_reference_t<Partition>>::value_type pval_t;
    typename vprop_map_t<pval_t>::type u_partition;
    typename eprop_map_t<weight_t>::type u_weight;
    typename vprop_map_t<vertex_t>::type u_mate;
    typename vprop_map_t<bool>::type is_augmented;

    typename vprop_map_t<vertex_t>::type vmap, rvmap;

    std::vector<vertex_t> S, T;
    pval_t S_val = partition[*vertices(g).first];
    pval_t T_val = S_val;
    for (auto v : vertices_range(g))
    {
        if (partition[v] == S_val)
        {
            S.push_back(v);
        }
        else
        {
            T.push_back(v);
            T_val = partition[v];
        }
        auto w = add_vertex(u);
        u_partition[w] = partition[v];
        vmap[v] = w;
        rvmap[w] = v;
    }

    for (auto e : edges_range(g))
    {
        auto ne = add_edge(vmap[source(e, g)],
                           vmap[target(e, g)], u);
        u_weight[ne.first] = get(weight, e);
    }

    if (S.size() > T.size())
    {
        S.swap(T);
        std::swap(S_val, T_val);
    }

    for (auto v : S)
    {
        auto w = add_vertex(u);
        is_augmented[w] = true;
        u_partition[w] = T_val;
        auto e = add_edge(vmap[v], w, u);
        u_weight[e.first] = 0;
    }

    maximum_bipartite_weighted_imperfect_matching(u,
                                                  u_partition.get_unchecked(),
                                                  u_weight.get_unchecked(),
                                                  u_mate.get_unchecked(num_vertices(u)));

    for (auto v : vertices_range(g))
    {
        auto w = vmap[v];
        auto x = u_mate[w];
        if (x == graph_traits<Graph>::null_vertex() || is_augmented[x])
            mate[v] = graph_traits<Graph>::null_vertex();
        else
            mate[v] = rvmap[x];
    }
}

}

#endif //GRAPH_BIPARTITE_WEIGHTED_HH
