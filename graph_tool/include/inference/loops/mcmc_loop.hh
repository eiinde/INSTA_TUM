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

#ifndef MCMC_LOOP_HH
#define MCMC_LOOP_HH

#include "config.h"

#include <iostream>

#include <tuple>

#include "hash_map_wrap.hh"
#include "parallel_rng.hh"
#include "graph_python_interface.hh"

#ifdef _OPENMP
#include <omp.h>
#endif
namespace graph_tool
{

template <class RNG>
bool metropolis_accept(double dS, double mP, double beta, RNG& rng)
{
    if (std::isinf(beta))
    {
        return dS < 0;
    }
    else
    {
        double a = -dS * beta + mP;
        if (a > 0)
        {
            return true;
        }
        else
        {
            std::uniform_real_distribution<> sample;
            return sample(rng) < exp(a);
        }
    }
}


template <class MCMCState, class RNG>
auto mcmc_sweep(MCMCState& state, RNG& rng)
{
    GILRelease gil;

    auto& vlist = state.get_vlist();
    auto beta = state.get_beta();

    typedef std::remove_const_t<decltype(state._null_move)> move_t;
    constexpr bool single_step =
        std::is_same_v<std::remove_reference_t<decltype(state.move_proposal(vlist.front(), rng))>,
                       move_t>;

    double S = 0;
    size_t nattempts = 0;
    size_t nmoves = 0;

    for (size_t iter = 0; iter < state.get_niter(); ++iter)
    {
        if (state.is_sequential() && !state.is_deterministic())
            std::shuffle(vlist.begin(), vlist.end(), rng);

        size_t nsteps = 1;
        auto get_N =
            [&]
            {
                if constexpr (single_step)
                    return vlist.size();
                else
                    return state.get_N();
            };

        state.init_iter(rng);

        for (size_t vi = 0; vi < get_N(); ++vi)
        {
            auto& v = (state.is_sequential()) ?
                vlist[vi] : uniform_sample(vlist, rng);

            if (state.skip_node(v))
                continue;

            if (state._verbose > 1)
            {
                auto&& r = state.node_state(v);
                cout << v << ": " << r;
            }

            auto&& ret = state.move_proposal(v, rng);

            auto get_s =
                [&]() -> move_t&
                {
                    if constexpr (single_step)
                    {
                        return ret;
                    }
                    else
                    {
                        nsteps = get<1>(ret);
                        return get<0>(ret);
                    }
                };

            move_t& s = get_s();

            if (s == state._null_move)
            {
                if (state._verbose > 1)
                    cout << " (null proposal)" << endl;
                continue;
            }

            double dS, mP;
            std::tie(dS, mP) = state.virtual_move_dS(v, s);

            nattempts += nsteps;

            bool accept = false;
            if (metropolis_accept(dS, mP, beta, rng))
            {
                state.perform_move(v, s);
                nmoves += nsteps;
                S += dS;
                accept = true;
            }

            state.step(v, s);

            if (state._verbose > 1)
                cout << " -> " << s << " " << accept << " " << dS << " " << mP << " " << -dS * beta + mP << " " << S << endl;
        }

        if (state.is_sequential() && state.is_deterministic())
            std::reverse(vlist.begin(), vlist.end());
    }
    return make_tuple(S, nattempts, nmoves);
}

} // graph_tool namespace

#endif //MCMC_LOOP_HH
