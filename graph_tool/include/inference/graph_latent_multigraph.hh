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

#ifndef GRAPH_LATENT_MULTIGRAPH_HH
#define GRAPH_LATENT_MULTIGRAPH_HH

#include <iostream>

#include "graph_tool.hh"

namespace graph_tool
{

using namespace std;
using namespace boost;

template <class Graph, class WMap, class TMap>
void  get_latent_multigraph(Graph& g, WMap w, TMap theta_out, TMap theta_in,
                            double epsilon, size_t max_niter, bool verbose)
{
    auto wc = w.get_checked();
    for (auto v : vertices_range(g))
    {
        auto tout = theta_out[v];
        auto tin = theta_in[v];
        auto e = add_edge(v, v, g).first;
        if (graph_tool::is_directed(g))
            wc[e] = tout * tin;
        else
            wc[e] = (tout * tin) / 2;
    }

    double delta = 1 + epsilon;
    size_t niter = 0;
    while (delta > epsilon && (niter < max_niter || max_niter == 0))
    {
        double M = 0;
        delta = 0;
        #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh())   \
            reduction(+:M) reduction(max:delta)
        parallel_edge_loop_no_spawn
            (g,
             [&](auto e)
             {
                 auto u = source(e, g);
                 auto v = target(e, g);
                 auto l = theta_out[u] * theta_in[v];
                 auto nw = ((u != v) ? l / (1-exp(-l)) :
                            (graph_tool::is_directed(g) ? l : l / 2));
                 auto& ew = w[e];
                 delta = std::max(delta, abs(nw - ew));
                 ew = nw;
                 M += ew;
             });

        #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh())   \
            reduction(max:delta)
        parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 if (graph_tool::is_directed(g))
                 {
                     auto& tout = theta_out[v];
                     auto d = out_degreeS()(v, g, w);
                     auto nt = d / sqrt(M);
                     delta = std::max(delta, abs(tout - nt));
                     tout = nt;

                     auto& tin = theta_in[v];
                     d = in_degreeS()(v, g, w);
                     nt = d / sqrt(M);
                     delta = std::max(delta, abs(tin - nt));
                     tin = nt;
                 }
                 else
                 {
                     auto& t = theta_out[v];
                     auto d = out_degreeS()(v, g, w);
                     auto nt = d / sqrt(2 * M);
                     delta = std::max(delta, abs(t - nt));
                     t = nt;
                 }
             });
        niter++;
        if (verbose)
            cout << delta << endl;
    }
};

} // graph_tool namespace

#endif //GRAPH_LATENT_MULTIGRAPH_HH
