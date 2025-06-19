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

#ifndef GIBBS_LOOP_HH
#define GIBBS_LOOP_HH

#include "config.h"

#include <tuple>

#include "hash_map_wrap.hh"
#include "parallel_rng.hh"
#include "graph_python_interface.hh"

#ifdef _OPENMP
#include <omp.h>
#endif
namespace graph_tool
{

template <class GibbsState, class RNG>
auto gibbs_sweep(GibbsState& state, RNG& rng)
{
    GILRelease gil;

    auto& vlist = state._vlist;
    auto beta = state._beta;

    vector<double> probs;
    vector<double> deltas;
    vector<size_t> idx;

    double S = 0;
    size_t nmoves = 0;
    size_t nattempts = 0;

    for (size_t iter = 0; iter < state._niter; ++iter)
    {
        if (!state._deterministic)
            std::shuffle(vlist.begin(), vlist.end(), rng);

        for (auto v : vlist)
        {
             if (!state._sequential)
                 v = uniform_sample(vlist, rng);

             if (state.node_weight(v) == 0)
                 continue;

             auto& moves = state.get_moves(v);

             nattempts += moves.size();

             probs.resize(moves.size());
             deltas.resize(moves.size());
             idx.resize(moves.size());

             double dS_min = numeric_limits<double>::max();
             for (size_t j = 0; j < moves.size(); ++j)
             {
                 size_t s = moves[j];
                 double dS = state.virtual_move_dS(v, s, rng);
                 dS_min = std::min(dS, dS_min);
                 deltas[j] = dS;
                 idx[j] = j;
             }

             if (!std::isinf(beta))
             {
                 for (size_t j = 0; j < moves.size(); ++j)
                 {
                     if (std::isinf(deltas[j]))
                         probs[j] = 0;
                     else
                         probs[j] = exp((-deltas[j] + dS_min) * beta);
                 }
             }
             else
             {
                 for (size_t j = 0; j < moves.size(); ++j)
                     probs[j] = (deltas[j] == dS_min) ? 1 : 0;
             }

             Sampler<size_t> sampler(idx, probs);

             size_t j = sampler.sample(rng);

             assert(probs[j] > 0);

             size_t s = moves[j];
             size_t r = state.node_state(v);

             if (s == r)
                 continue;

             state.perform_move(v, s);
             nmoves += state.node_weight(v);
             S += deltas[j];
        }

        if (state._sequential && state._deterministic)
            std::reverse(vlist.begin(), vlist.end());

    }
    return std::make_tuple(S, nattempts, nmoves);
}

} // graph_tool namespace

#endif //GIBBS_LOOP_HH
