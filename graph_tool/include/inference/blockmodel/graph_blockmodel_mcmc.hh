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

#ifndef GRAPH_BLOCKMODEL_MCMC_HH
#define GRAPH_BLOCKMODEL_MCMC_HH

#include "config.h"

#include <vector>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include <boost/mpl/vector.hpp>

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_BLOCK_STATE_params(State)                                         \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((vlist,&, std::vector<size_t>&, 0))                                       \
    ((beta,, double, 0))                                                       \
    ((c,, double, 0))                                                          \
    ((d,, double, 0))                                                          \
    ((oentropy_args,, python::object, 0))                                      \
    ((allow_vacate,, bool, 0))                                                 \
    ((sequential,, bool, 0))                                                   \
    ((deterministic,, bool, 0))                                                \
    ((verbose,, int, 0))                                                       \
    ((niter,, size_t, 0))


template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCBlockStateBase, MCMC_BLOCK_STATE_params(State))

    template <class... Ts>
    class MCMCBlockState
        : public MCMCBlockStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MCMCBlockStateBase<Ts...>,
                         MCMC_BLOCK_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_BLOCK_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCBlockState(ATs&&... as)
           : MCMCBlockStateBase<Ts...>(as...),
            _g(_state._g),
            _m_entries(num_vertices(_state._bg)),
            _entropy_args(python::extract<typename State::_entropy_args_t&>(_oentropy_args))
        {
            GILRelease gil_release;
            _state.init_mcmc(*this);
        }

        typename state_t::g_t& _g;
        typename state_t::m_entries_t _m_entries;
        constexpr static size_t _null_move = null_group;
        typename State::_entropy_args_t& _entropy_args;

        size_t node_state(size_t v)
        {
            return _state._b[v];
        }

        bool skip_node(size_t v)
        {
            return _state.node_weight(v) == 0;
        }

        template <class RNG>
        size_t move_proposal(size_t v, RNG& rng)
        {
            if (!_allow_vacate && _state.is_last(v))
                return _null_move;
            size_t s = _state.sample_block(v, _c, _d, rng);
            if (s == node_state(v))
                return _null_move;
            return s;
        }

        std::tuple<double, double>
        virtual_move_dS(size_t v, size_t nr)
        {
            size_t r = _state._b[v];
            if (r == nr)
                return std::make_tuple(0., 0.);

            double dS = _state.virtual_move(v, r, nr, _entropy_args,
                                            _m_entries);
            double a = 0;
            if (!std::isinf(_beta))
            {
                double pf = _state.get_move_prob(v, r, nr, _c, _d, false,
                                                 _m_entries);
                double pb = _state.get_move_prob(v, nr, r, _c, _d, true,
                                                 _m_entries);
                a = pb - pf;
            }
            return std::make_tuple(dS, a);
        }

        void perform_move(size_t v, size_t nr)
        {
            _state.move_vertex(v, nr, _m_entries);
        }

        bool is_deterministic()
        {
            return _deterministic;
        }

        bool is_sequential()
        {
            return _sequential;
        }

        auto& get_vlist()
        {
            return _vlist;
        }

        double get_beta()
        {
            return _beta;
        }

        size_t get_niter()
        {
            return _niter;
        }

        void step(size_t, size_t)
        {
        }

        template <class RNG>
        void init_iter(RNG&)
        {
        }
    };
};


} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_MCMC_HH
