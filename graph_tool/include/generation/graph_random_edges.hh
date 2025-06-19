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

#ifndef GRAPH_RANDOM_EDGES_HH
#define GRAPH_RANDOM_EDGES_HH

#include "graph.hh"
#include "graph_util.hh"
#include "random.hh"
#include "sampler.hh"
#include "dynamic_sampler.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class Graph, class EWeight, class RNG>
void add_random_edges(Graph& g, size_t E, bool parallel, bool self_loops,
                      bool filtered, EWeight eweight, RNG& rng)
{
    auto dispatch = [&](auto& vsampler)
    {
        size_t m = 0;
        while (m < E)
        {
            size_t u = vsampler(rng);
            size_t v = vsampler(rng);
            if (!self_loops && u == v)
                continue;
            if constexpr (std::is_same_v<EWeight,
                                         UnityPropertyMap<int, GraphInterface::edge_t>>)
            {
                if (!parallel && edge(u, v, g).second)
                    continue;
                add_edge(u, v, g);
            }
            else
            {
                auto [e, exists] = edge(u, v, g);
                if (!parallel && exists && eweight[e] > 0)
                    continue;
                if (!exists)
                    e = add_edge(u, v, g).first;
                eweight[e]++;
            }
            m++;
        }
    };

    if (filtered)
    {
        std::vector<size_t> vs(vertices(g).first, vertices(g).second);
        auto vsampler = [&](auto& rng)
        {
            return uniform_sample(vs, rng);
        };
        dispatch(vsampler);
    }
    else
    {
        std::uniform_int_distribution<size_t> vsampler(0, num_vertices(g) - 1);
        dispatch(vsampler);
    }
}

template <class Graph, class EWeight, class RNG>
void remove_random_edges(Graph& g, size_t E, EWeight eweight, bool counts,
                         RNG& rng)
{
    if constexpr (std::is_same_v<EWeight,
                                 UnityPropertyMap<int, GraphInterface::edge_t>>)
    {
        std::vector<typename graph_traits<Graph>::edge_descriptor> edges;
        for (auto e : edges_range(g))
            edges.push_back(e);

        size_t m = 0;
        for (auto iter = edges.begin(); iter != edges.end(); ++iter)
        {
            if (m == E)
                break;
            auto item = uniform_sample_iter(iter, edges.end(), rng);
            std::swap(*item, *iter);
            remove_edge(*iter, g);
            m++;
        }
    }
    else
    {
        std::vector<typename graph_traits<Graph>::edge_descriptor> edges;
        std::vector<double> eprob;
        size_t M = 0;
        for (auto e : edges_range(g))
        {
            auto w = eweight[e];
            if (w <= 0)
                continue;
            edges.push_back(e);
            eprob.push_back(counts ? ceil(w) : w);
            if (counts)
                M += w;
            else
                M++;
        }

        DynamicSampler<typename graph_traits<Graph>::edge_descriptor>
            esampler(edges, eprob);
        for (size_t i = 0; i < std::min(M, E); ++i)
        {
            size_t j = esampler.sample_idx(rng);
            auto& e = edges[j];
            auto& w = eweight[e];
            if (counts)
            {
                esampler.update(j, ceil(w) - 1, false);
                w--;
                if (w <= 0)
                    remove_edge(e, g);
            }
            else
            {
                esampler.update(j, 0, false);
                remove_edge(e, g);
            }
        }
    }
}

} // graph_tool namespace

#endif // GRAPH_RANDOM_EDGES_HH
