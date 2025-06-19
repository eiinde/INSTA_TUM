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

#ifndef GRAPH_BLOCKMODEL_UNCERTAIN_UTIL_HH
#define GRAPH_BLOCKMODEL_UNCERTAIN_UTIL_HH

#include "config.h"

#include "../support/util.hh"

#include "../blockmodel/graph_blockmodel_entropy.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

struct uentropy_args_t:
        public entropy_args_t
{
    uentropy_args_t(const entropy_args_t& ea)
        : entropy_args_t(ea) {}

    bool latent_edges = true;
    bool density = false;
    bool sbm = true;
    double aE = std::numeric_limits<double>::quiet_NaN();
};

template <class State, class... X>
double get_edge_prob(State& state, size_t u, size_t v, const uentropy_args_t& ea,
                     double epsilon, X... x)
{
    auto e = state.get_u_edge(u, v);
    size_t ew = 0;
    [[maybe_unused]] int old_x = 0;
    if (e != state._null_edge)
    {
        ew = state._eweight[e];
        if constexpr (sizeof...(X) > 0)
            old_x = state._xc[e];
    }

    if (ew > 0)
        state.remove_edge(u, v, ew);

    double S = 0;
    double delta = 1. + epsilon;
    size_t ne = 0;
    double L = -std::numeric_limits<double>::infinity();
    while (delta > epsilon || ne < 2)
    {
        double dS;
        if constexpr (sizeof...(X) < 2)
        {
            dS = state.add_edge_dS(u, v, 1, x..., ea);
            state.add_edge(u, v, 1, x...);
        }
        else
        {
            std::tuple<X...> rg(x...);
            if (ne > 0)
            {
                dS = state.add_edge_dS(u, v, 1, get<0>(rg), ea);
                state.add_edge(u, v, 1, get<0>(rg));
            }
            else
            {
                double dS_x = state.add_edge_dS(u, v, 1, get<0>(rg), ea);
                uentropy_args_t mea(ea);
                mea.latent_edges = false;
                dS = state.add_edge_dS(u, v, 1, get<0>(rg), mea);
                state.add_edge(u, v, 1, get<0>(rg));
                dS_x -= dS;

                double Lx = -std::numeric_limits<double>::infinity();
                for (size_t m = get<0>(rg) + 1; m < get<1>(rg); ++m)
                {
                    dS_x += state.update_edge_dS(u, v, m, ea);
                    state.update_edge(u, v, m);
                    Lx = log_sum_exp(Lx, -dS_x);
                }
                state.update_edge(u, v, get<0>(rg));

                L += Lx;
            }
        }
        S += dS;
        double old_L = L;
        L = log_sum_exp(L, -S);
        ne++;
        delta = abs(L-old_L);
    }

    L = (L > 0) ? -log1p(exp(-L)) : L - log1p(exp(L));

    if constexpr (sizeof...(X) > 0)
    {
        state.remove_edge(u, v, ne);
        if (ew > 0)
            state.add_edge(u, v, ew, old_x);
    }
    else
    {
        if (ne > ew)
            state.remove_edge(u, v, ne - ew);
        else if (ne < ew)
            state.add_edge(u, v, ew - ne);
    }

    return L;
}

template <class State>
void get_edges_prob(State& state, python::object edges, python::object probs,
                    const uentropy_args_t& ea, double epsilon)
{
    multi_array_ref<uint64_t,2> es = get_array<uint64_t,2>(edges);
    multi_array_ref<double,1> eprobs = get_array<double,1>(probs);
    for (size_t i = 0; i < eprobs.shape()[0]; ++i)
        eprobs[i] = get_edge_prob(state, es[i][0], es[i][1], ea, epsilon);
}

template <class State>
void get_xedges_prob(State& state, python::object edges, python::object probs,
                     const uentropy_args_t& ea, double epsilon)
{
    multi_array_ref<double,2> es = get_array<double,2>(edges);
    multi_array_ref<double,1> eprobs = get_array<double,1>(probs);
    for (size_t i = 0; i < eprobs.shape()[0]; ++i)
    {
        if (es.shape()[1] > 2)
            eprobs[i] = get_edge_prob(state, es[i][0], es[i][1], ea, epsilon, es[i][2]);
        else
            eprobs[i] = get_edge_prob(state, es[i][0], es[i][1], ea, epsilon, 0, state._xvals.size());
    }
}

template <class State, class Graph, class EProp>
void set_state(State& state, Graph& g, EProp w)
{
    std::vector<std::pair<size_t, size_t>> us;
    for (auto v : vertices_range(state._u))
    {
        us.clear();
        for (auto e : out_edges_range(v, state._u))
        {
            auto w = target(e, state._u);
            if (w == v)
                continue;
            us.emplace_back(w, state._eweight[e]);
        }
        for (auto& uw : us)
            state.remove_edge(v, uw.first, uw.second);

        auto& e = state.template get_u_edge<false>(v, v);
        if (e == state._null_edge)
            continue;
        size_t x = state._eweight[e];
        state.remove_edge(v, v, x);
    }

    for (auto e : edges_range(g))
        state.add_edge(source(e, g), target(e, g), w[e]);
}


} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_UNCERTAIN_UTIL_HH
