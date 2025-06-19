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

#ifndef GRAPH_SIMILARITY_HH
#define GRAPH_SIMILARITY_HH

#include "hash_map_wrap.hh"
#include "idx_map.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

template <bool normed, class Keys, class Set1, class Set2>
auto set_difference(Keys& ks, Set1& s1, Set2& s2, double norm, bool asym)
{
    typename Set1::value_type::second_type s = 0;

    auto ndispatch = [&](auto x)
                     {
                         if constexpr (normed)
                             return std::pow(x, norm);
                         else
                             return x;
                     };

    auto get_map =
        [&](auto& m, auto&& k)
        {
            auto iter = m.find(k);
            if (iter == m.end())
                return decltype(iter->second)(0);
            return iter->second;
        };

    for (auto& k : ks)
    {
        auto x1 = get_map(s1, k);
        auto x2 = get_map(s2, k);
        if (x1 > x2)
            s += ndispatch(x1 - x2);
        else if (!asym)
            s += ndispatch(x2 - x1);
    }

    return s;
}

template <class Vertex, class WeightMap, class LabelMap,
          class Graph1, class Graph2, class Keys, class Adj>
auto vertex_difference(Vertex v1, Vertex v2, WeightMap& ew1, WeightMap& ew2,
                       LabelMap& l1, LabelMap& l2, const Graph1& g1,
                       const Graph2& g2, bool asym, Keys& keys, Adj& adj1,
                       Adj& adj2, double norm)
{
    if (v1 != graph_traits<Graph1>::null_vertex())
    {
        for (auto e : out_edges_range(v1, g1))
        {
            auto w = ew1[e];
            auto k = get(l1, target(e, g1));
            adj1[k] += w;
            keys.insert(k);
        }
    }

    if (v2 != graph_traits<Graph2>::null_vertex())
    {
        for (auto e : out_edges_range(v2, g2))
        {
            auto w = ew2[e];
            auto k = get(l2, target(e, g2));
            adj2[k] += w;
            keys.insert(k);
        }
    }

    if (norm == 1)
        return set_difference<false>(keys, adj1, adj2, 1, asym);
    else
        return set_difference<true>(keys, adj1, adj2, norm, asym);
}

template <class Graph1, class Graph2, class WeightMap, class LabelMap>
auto get_similarity(const Graph1& g1, const Graph2& g2, WeightMap ew1,
                    WeightMap ew2, LabelMap l1, LabelMap l2, double norm,
                    bool asym)
{
    typedef typename property_traits<LabelMap>::value_type label_t;
    typedef typename property_traits<WeightMap>::value_type val_t;

    typedef typename graph_traits<Graph1>::vertex_descriptor vertex_t;
    std::unordered_map<label_t, vertex_t> lmap1;
    std::unordered_map<label_t, vertex_t> lmap2;

    for (auto v : vertices_range(g1))
        lmap1[get(l1, v)] = v;
    for (auto v : vertices_range(g2))
        lmap2[get(l2, v)] = v;

    val_t s = 0;
    for (auto& lv1 : lmap1)
    {
        vertex_t v1 = lv1.second;
        vertex_t v2;

        auto li2 = lmap2.find(lv1.first);
        if (li2 == lmap2.end())
            v2 = graph_traits<Graph2>::null_vertex();
        else
            v2 = li2->second;

        std::unordered_set<label_t> keys;
        std::unordered_map<label_t, val_t> adj1, adj2;

        s += vertex_difference(v1, v2, ew1, ew2, l1, l2, g1, g2, asym, keys,
                               adj1, adj2, norm);
    }

    if (!asym)
    {
        for (auto& lv2 : lmap2)
        {
            vertex_t v2 = lv2.second;
            vertex_t v1;

            auto li1 = lmap1.find(lv2.first);
            if (li1 == lmap1.end())
                v1 = graph_traits<Graph2>::null_vertex();
            else
                continue;

            std::unordered_set<label_t> keys;
            std::unordered_map<label_t, val_t> adj1, adj2;

            s += vertex_difference(v1, v2, ew1, ew2, l1, l2, g1, g2, false,
                                   keys, adj1, adj2, norm);
        }
    }
    return s;
}

template <class Graph1, class Graph2, class WeightMap, class LabelMap>
auto get_similarity_fast(const Graph1& g1, const Graph2& g2, WeightMap ew1,
                         WeightMap ew2, LabelMap l1, LabelMap l2, double norm,
                         bool asym)
{
    typedef typename property_traits<WeightMap>::value_type val_t;
    typedef typename graph_traits<Graph1>::vertex_descriptor vertex_t;
    typedef typename property_traits<LabelMap>::value_type label_t;

    vector<vertex_t> lmap1, lmap2;

    for (auto v : vertices_range(g1))
    {
        size_t i = get(l1, v);
        if (lmap1.size() <= i)
            lmap1.resize(i * i + 1, graph_traits<Graph1>::null_vertex());
        lmap1[i] = v;
    }

    for (auto v : vertices_range(g2))
    {
        size_t i = get(l2, v);
        if (lmap2.size() <= i)
            lmap2.resize(i * i + 1, graph_traits<Graph2>::null_vertex());
        lmap2[i] = v;
    }

    size_t N = std::max(lmap1.size(), lmap2.size());
    lmap1.resize(N, graph_traits<Graph1>::null_vertex());
    lmap2.resize(N, graph_traits<Graph2>::null_vertex());

    idx_set<label_t, false, false> keys(N);
    idx_map<label_t, val_t, false, false> adj1(N), adj2(N);

    val_t s = 0;
    #pragma omp parallel if (num_vertices(g1) > get_openmp_min_thresh()) \
        reduction(+:s) firstprivate(keys, adj1, adj2)
    parallel_loop_no_spawn
        (lmap1,
         [&](size_t i, auto v1)
         {
             auto v2 = lmap2[i];
             if (v1 == graph_traits<Graph1>::null_vertex() &&
                 v2 == graph_traits<Graph2>::null_vertex())
                 return;
             keys.clear();
             adj1.clear();
             adj2.clear();
             s += vertex_difference(v1, v2, ew1, ew2, l1, l2, g1, g2, asym,
                                    keys, adj1, adj2, norm);
         });

    if (!asym)
    {
        #pragma omp parallel if (num_vertices(g2) > get_openmp_min_thresh())  \
            reduction(+:s) firstprivate(keys, adj1, adj2)
        parallel_loop_no_spawn
            (lmap2,
             [&](size_t i, auto v2)
             {
                 auto v1 = lmap1[i];
                 if (v1 != graph_traits<Graph1>::null_vertex() ||
                     v2 == graph_traits<Graph2>::null_vertex())
                     return;
                 keys.clear();
                 adj1.clear();
                 adj2.clear();
                 s += vertex_difference(v1, v2, ew1, ew2, l1, l2, g1, g2, false,
                                        keys, adj1, adj2, norm);
             });
    }

    return s;
}

} // graph_tool namespace

#endif // GRAPH_SIMILARITY_HH
