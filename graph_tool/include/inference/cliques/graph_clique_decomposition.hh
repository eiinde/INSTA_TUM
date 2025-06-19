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

#ifndef GRAPH_CLIQUE_DECOMPOSITION_HH
#define GRAPH_CLIQUE_DECOMPOSITION_HH

#include "hash_map_wrap.hh"

using namespace graph_tool;
using namespace boost;
using namespace std;


double L_over(size_t N, size_t d, size_t Ed, size_t D, double E)
{
    double lb = lbinom_fast(N, d);
    return lgamma_fast(Ed + 1) - (Ed + 1) * log1p((D - 1) / E) - Ed * lb - log(E / (D - 1));
}

template <class Graph, class Vxprop, class Vprop, class Cprop, class Vec, class RNG>
std::tuple<double, size_t>
iter_mh(Graph& g, Vxprop x, Cprop c, Vprop is_fac, Vprop is_max, Vec& Ed, int N,
        int E, double beta, size_t niter, RNG& rng)
{
    double dL = 0;
    size_t nflips = 0;

    size_t D = 0;

    gt_hash_map<std::vector<int>, size_t> vars;
    gt_hash_map<std::tuple<int,int>, size_t> facs;
    vector<size_t> mvars;
    std::vector<size_t> removed;
    for (auto v : vertices_range(g))
    {
        if (is_fac[v])
        {
            facs[{c[v][0], c[v][1]}] = v;
        }
        else
        {
            if (is_max[v])
                mvars.push_back(v);
            if (out_degree(v, g) > 0)
                vars[c[v]] = v;
            else
                removed.push_back(v);
            D = std::max(c[v].size(), D);
        }
    }

    vector<int> nc;
    auto c_c = c.get_checked();
    auto is_fac_c = is_fac.get_checked();
    auto x_c = x.get_checked();
    auto is_max_c = is_max.get_checked();

    auto get_v =
        [&](auto& nc)
        {
            auto iter = vars.find(nc);
            if (iter != vars.end())
                return iter->second;

            size_t v;
            if (removed.empty())
            {
                v = add_vertex(g);
            }
            else
            {
                v = removed.back();
                removed.pop_back();
            }

            vars[nc] = v;

            for (size_t i = 0; i < nc.size(); ++i)
                for (size_t j = i+1; j < nc.size(); ++j)
                    add_edge(v, facs[{nc[i], nc[j]}], g);

            c_c[v] = nc;
            is_fac_c[v] = 0;
            x_c[v] = 0;
            is_max_c[v] = 0;
            return v;
        };

    auto cleanup_v =
        [&](auto v)
        {
            if (x[v] == 0 && !is_max[v])
            {
                clear_vertex(v, g);
                removed.push_back(v);
                vars.erase(c[v]);
            }
        };

    std::vector<int> uc;
    std::vector<size_t> us;

    for (size_t i = 0; i < niter; ++i)
    {
        std::shuffle(mvars.begin(), mvars.end(), rng);
        for (auto mv : mvars)
        {
            uniform_int_distribution<> d_sample(2, c[mv].size());
            int dv = d_sample(rng);
            nc = c[mv];
            std::shuffle(nc.begin(), nc.end(), rng);
            nc.erase(nc.begin() + dv, nc.end());
            std::sort(nc.begin(), nc.end());

            auto v = get_v(nc);

            int dx = 0;
            if (x[v] == 0)
            {
                dx = 1;
            }
            else
            {
                bernoulli_distribution add(.5);
                dx = add(rng) ? 1 : -1;
            }

            if (x[v] + dx == 0)
            {
                int x_min = numeric_limits<int>::max();
                for (auto u : out_neighbors_range(v, g))
                    x_min = std::min(x_min, x[u]);

                if (x_min == 1)
                {
                    cleanup_v(v);
                    continue;
                }
            }

            double pb = L_over(N, dv, Ed[dv], D, E) - lgamma_fast(x[v] + 1);
            double pa = L_over(N, dv, Ed[dv] + dx, D, E) - lgamma_fast(x[v] + dx + 1);

            double a = 0;
            a -= (x[v] == 0) ? 0 : -log(2);
            a += (x[v] + dx == 0) ? 0 : -log(2);

            bernoulli_distribution accept(exp(min(beta * (pa - pb) + a, 0.)));
            if ((std::isinf(beta) && (pa > pb)) || accept(rng))
            {
                x[v] += dx;
                for (auto u : out_neighbors_range(v, g))
                    x[u] += dx;
                Ed[dv] += dx;

                dL += pa - pb;
                nflips++;
            }

            cleanup_v(v);
        }
    }

    return std::make_tuple(-dL, nflips);
}

#endif //GRAPH_CLIQUE_DECOMPOSITION_HH
