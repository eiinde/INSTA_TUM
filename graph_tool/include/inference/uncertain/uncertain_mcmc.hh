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

#ifndef GRAPH_BLOCKMODEL_UNCERTAIN_MCMC_HH
#define GRAPH_BLOCKMODEL_UNCERTAIN_MCMC_HH

#include "config.h"

#include <vector>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include "sample_edge_sbm.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef std::vector<size_t> vlist_t;

#define MCMC_UNCERTAIN_STATE_params(State)                                     \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((entropy_args,, uentropy_args_t, 0))                                      \
    ((edges_only,, bool, 0))                                                   \
    ((verbose,, int, 0))                                                       \
    ((niter,, size_t, 0))


template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCUncertainStateBase, MCMC_UNCERTAIN_STATE_params(State))

    template <class... Ts>
    class MCMCUncertainState
        : public MCMCUncertainStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MCMCUncertainStateBase<Ts...>,
                         MCMC_UNCERTAIN_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_UNCERTAIN_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCUncertainState(ATs&&... as)
            : MCMCUncertainStateBase<Ts...>(as...),
              _edge_sampler(_state._block_state, _edges_only ? 1. : 0.25),
              _vlist(num_vertices(_state._u))
        {
        }

        SBMEdgeSampler<typename State::block_state_t> _edge_sampler;

        std::tuple<size_t, size_t> _e;
        std::vector<size_t> _vlist;
        int _null_move = 0;

        std::tuple<size_t, size_t> get_edge()
        {
            return _e;
        }

        size_t node_state(size_t u, size_t v)
        {
            auto&& e = _state.get_u_edge(u, v);
            if (e == _state._null_edge)
                return 0;
            return _state._eweight[e];
        }

        size_t node_state(size_t)
        {
            size_t u, v;
            std::tie(u, v) = get_edge();
            return node_state(u, v);
        }

        template <class T>
        bool skip_node(T&)
        {
            return false;
        }

        template <class RNG>
        int move_proposal(size_t, RNG& rng)
        {
            _e = _edge_sampler.sample(rng);

            int m = node_state(get<0>(_e), get<1>(_e));

            std::geometric_distribution<int> sample_m(1./(m + 2));

            return sample_m(rng) - m;
        }

        std::tuple<double, double>
        virtual_move_dS(size_t, int dm)
        {
            if (dm == 0)
                return {0., 0.};

            size_t u, v;
            std::tie(u, v) = get_edge();

            double dS = 0;
            if (dm < 0)
                dS = _state.remove_edge_dS(u, v, -dm, _entropy_args);
            else
                dS = _state.add_edge_dS(u, v, dm, _entropy_args);

            size_t m = node_state(u, v);
            double a = (_edge_sampler.log_prob(u, v, m, dm) -
                        _edge_sampler.log_prob(u, v, m, 0));

            size_t nm = m + dm;

            a -= nm * safelog_fast(m + 1) - (nm + 1) * safelog_fast(m + 2);
            a += m * safelog_fast(nm + 1) - (m + 1) * safelog_fast(nm + 2);

            return std::make_tuple(dS, a);
        }

        void perform_move(size_t, int dm)
        {
            if (dm == 0)
                return;

            size_t u, v;
            std::tie(u, v) = get_edge();
            size_t m = node_state(u, v);

            if (dm < 0)
            {
                _edge_sampler.update_edge(u, v, m, dm);
                _state.remove_edge(u, v, -dm);
            }
            else
            {
                _state.add_edge(u, v, dm);
                _edge_sampler.update_edge(u, v, m, dm);
            }
        }

        bool is_deterministic()
        {
            return false;
        }

        bool is_sequential()
        {
            return false;
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

        template <class T>
        void step(T&, int)
        {
        }

        template <class RNG>
        void init_iter(RNG&)
        {
        }
    };
};


} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_UNCERTAIN_MCMC_HH
