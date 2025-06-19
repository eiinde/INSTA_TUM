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

#ifndef GRAPH_CONTRACT_EDGES_HH
#define GRAPH_CONTRACT_EDGES_HH

#include "graph.hh"
#include "graph_util.hh"
#include "random.hh"
#include "sampler.hh"
#include "idx_map.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class Graph, class EWeight>
void contract_parallel_edges(Graph& g, EWeight eweight)
{
    typedef typename graph_traits<Graph>::edge_descriptor edge_t;
    idx_map<size_t, edge_t, false, false> emap(num_vertices(g));
    idx_set<size_t> loops;
    auto eidx = get(edge_index_t(), g);
    std::vector<edge_t> remove;

    for (auto v : vertices_range(g))
    {
        emap.clear();
        remove.clear();
        loops.clear();
        for (auto e : out_edges_range(v, g))
        {
            auto u = target(e, g);

            if (!graph_tool::is_directed(g) && u < v)
                continue;

            auto iter = emap.find(u);
            if (iter == emap.end())
            {
                emap[u] = e;
                if (u == v)
                    loops.insert(eidx[e]);
            }
            else
            {
                if (loops.find(eidx[e]) != loops.end()) // self-loops
                    continue;
                if constexpr (is_convertible_v<typename boost::property_traits<EWeight>::category,
                                               boost::writable_property_map_tag>)
                {
                    auto w = eweight[iter->second];
                    put(eweight, iter->second, eweight[e] + w);
                }
                remove.push_back(e);
                if (u == v)
                    loops.insert(eidx[e]);
            }
        }

        for (auto& e : remove)
            remove_edge(e, g);
    }
}

template <class Graph, class EWeight>
void expand_parallel_edges(Graph& g, EWeight eweight)
{
    typedef typename graph_traits<Graph>::edge_descriptor edge_t;
    std::vector<edge_t> edges;
    idx_set<size_t> loops;
    auto eidx = get(edge_index_t(), g);
    for (auto v : vertices_range(g))
    {
        edges.clear();
        if (!graph_tool::is_directed(g))
            loops.clear();

        for (auto e : out_edges_range(v, g))
        {
            auto u = target(e, g);

            if (!graph_tool::is_directed(g))
            {
                if (v > u)
                    continue;
                if (v == u && loops.find(eidx[e]) != loops.end()) // self-loops
                    continue;
            }

            edges.push_back(e);

            if (!graph_tool::is_directed(g) && u == v)
                loops.insert(eidx[e]);
        }

        for (auto& e : edges)
        {
            size_t w = eweight[e];
            if (w == 0)
            {
                remove_edge(e, g);
            }
            else
            {
                auto u = target(e, g);
                for (size_t m = 0; m < w - 1; ++m)
                    add_edge(v, u, g);
            }
        }
    }
}

} // graph_tool namespace

#endif // GRAPH_CONTRACT_EDGES_HH
