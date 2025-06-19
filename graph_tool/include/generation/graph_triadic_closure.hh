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

#ifndef GRAPH_TRIADIC_CLOSURE_HH
#define GRAPH_TRIADIC_CLOSURE_HH

#include <tuple>

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "hash_map_wrap.hh"
#include "sampler.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class Graph, class EMap, class ETMap, class EV, class RNG>
void gen_triadic_closure(Graph& g, EMap curr, ETMap ego, EV& Es, bool probs,
                         RNG& rng)
{
    std::vector<uint8_t> mark(num_vertices(g));
    std::vector<std::vector<std::tuple<size_t, size_t>>> edges(num_vertices(g));

    #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh())   \
        firstprivate(mark)
    parallel_vertex_loop_no_spawn
        (g,
         [&](auto v)
         {
             if (Es[v] == 0)
                 return;

             for (auto e1 : out_edges_range(v, g))
             {
                 auto w = target(e1, g);
                 if (w == v)
                     continue;

                 for (auto e2 : out_edges_range(w, g))
                     mark[target(e2, g)] = 1;

                 for (auto e2 : out_edges_range(v, g))
                 {
                     if (!curr[e1] && !curr[e2])
                         continue;

                     auto u = target(e2, g);

                     if (u >= w || mark[u] > 0)
                         continue;

                     edges[v].emplace_back(u, w);
                 }

                 for (auto e2 : out_edges_range(w, g))
                     mark[target(e2, g)] = 0;
             }
         });

    for (auto v : vertices_range(g))
    {
        auto x = Es[v];
        if (x == 0)
            continue;

        size_t E;
        if (probs)
        {
            std::binomial_distribution<size_t> sample(edges[v].size(), x);
            E = sample(rng);
        }
        else
        {
            E = x;
        }

        for (const auto& e : random_permutation_range(edges[v], rng))
        {
            if (E == 0)
                break;
            auto ne = add_edge(get<0>(e), get<1>(e), g);
            ego[ne.first] = v;
            E--;
        }
    }
}


} // graph_tool namespace

#endif // GRAPH_TRIADIC_CLOSURE_HH
