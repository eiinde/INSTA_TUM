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

#ifndef GRAPH_PAGERANK_HH
#define GRAPH_PAGERANK_HH

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

struct get_pagerank
{
    template <class Graph, class VertexIndex, class RankMap, class PerMap,
              class Weight>
    void operator()(Graph& g, VertexIndex vertex_index, RankMap rank,
                    PerMap pers, Weight weight, double damping, double epsilon,
                    size_t max_iter, size_t& iter) const
    {
        typedef typename property_traits<RankMap>::value_type rank_type;

        RankMap r_temp(vertex_index, num_vertices(g));
        RankMap deg(vertex_index, num_vertices(g));

        // init degs
        std::vector<size_t> sinks;
        for (auto v : vertices_range(g))
        {
            auto k = out_degreeS()(v, g, weight);
            put(deg, v, k);
            if (k == 0)
                sinks.push_back(v);
        }

        rank_type delta = epsilon + 1;
        rank_type d = damping;
        iter = 0;
        while (delta >= epsilon)
        {
            delta = 0;

            double p_sink = 0;
            #pragma omp parallel if (sinks.size() > get_openmp_min_thresh())      \
                reduction(+:p_sink)
            parallel_loop_no_spawn
                (sinks,
                 [&](auto, auto v){ p_sink += get(rank, v); });

            #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh())   \
                reduction(+:delta)
            parallel_vertex_loop_no_spawn
                (g,
                 [&](auto v)
                 {
                     rank_type r = p_sink * get(pers, v);
                     for (const auto& e : in_or_out_edges_range(v, g))
                     {
                         auto s = source(e, g);
                         r += (get(rank, s) * get(weight, e)) / get(deg, s);
                     }

                     put(r_temp, v, (1.0 - d) * get(pers, v) + d * r);

                     delta += abs(get(r_temp, v) - get(rank, v));
                 });
            swap(r_temp, rank);
            ++iter;
            if (max_iter > 0 && iter == max_iter)
                break;
        }

        if (iter % 2 != 0)
        {
            parallel_vertex_loop
                (g,
                 [&](auto v)
                 {
                     put(rank, v, get(r_temp, v));
                 });
        }
    }
};

}
#endif // GRAPH_PAGERANK_HH
