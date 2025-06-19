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

#ifndef GRAPH_BLOCKMODEL_MULTICANONICAL_HH
#define GRAPH_BLOCKMODEL_MULTICANONICAL_HH

#include "config.h"

#include <vector>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include <boost/mpl/vector.hpp>

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MULTICANONICAL_BLOCK_STATE_params(State)                               \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((hist, &, std::vector<size_t>&, 0))                                       \
    ((dens, &, std::vector<double>&, 0))                                       \
    ((S_min, , double, 0))                                                     \
    ((S_max, , double, 0))                                                     \
    ((f, , double, 0))                                                         \
    ((S, , double, 0))                                                         \
    ((verbose,, int, 0))


template <class State>
struct Multicanonical
{
    GEN_STATE_BASE(MulticanonicalBlockStateBase,
                   MULTICANONICAL_BLOCK_STATE_params(State))

    template <class... Ts>
    class MulticanonicalBlockState
        : public MulticanonicalBlockStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MulticanonicalBlockStateBase<Ts...>,
                         MULTICANONICAL_BLOCK_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MULTICANONICAL_BLOCK_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MulticanonicalBlockState(ATs&&... as)
           : MulticanonicalBlockStateBase<Ts...>(as...),
            _i(get_bin(_S))
        {
        }

        int _i;
        double _dS;

        typedef decltype(_state._null_move) move_t;

        move_t _null_move = _state._null_move;

        int get_bin(double S)
        {
            return std::floor((_hist.size() - 1) *
                              ((S - _S_min) / (_S_max - _S_min)));
        };

        bool skip_node(size_t v)
        {
            return _state.skip_node(v);
        }

        size_t node_state(size_t v)
        {
            return _state.node_state(v);
        }

        template <class RNG>
        auto move_proposal(size_t v, RNG& rng)
        {
            return _state.move_proposal(v, rng);
        }

        auto virtual_move_dS(size_t v, move_t nr)
        {
            auto dS = _state.virtual_move_dS(v, nr);
            double nS = _S + get<0>(dS);
            if (nS < _S_min || nS >= _S_max)
            {
                get<0>(dS) = numeric_limits<double>::infinity();
            }
            else
            {
                int j = get_bin(nS);
                get<1>(dS) += _dens[_i] - _dens[j];
            }
            _dS = get<0>(dS);
            return dS;
        }

        void perform_move(size_t v, move_t nr)
        {
            _state.perform_move(v, nr);
            _S += _dS;
            _i = get_bin(_S);
        }

        bool is_deterministic()
        {
            return _state.is_deterministic();
        }

        bool is_sequential()
        {
            return _state.is_sequential();
        }

        auto& get_vlist()
        {
            return _state.get_vlist();
        }

        size_t get_N()
        {
            return _state.get_N();
        }

        double get_beta()
        {
            return 1;
        }

        size_t get_niter()
        {
            return _state.get_niter();
        }

        void step(size_t, move_t)
        {
            _hist[_i]++;
            _dens[_i] += _f;
        }

        template <class RNG>
        void init_iter(RNG&)
        {
        }
    };
};

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_MULTICANONICAL_HH
