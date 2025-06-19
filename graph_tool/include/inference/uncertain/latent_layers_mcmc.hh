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

namespace std
{
template <class T1, class T2>
ostream& operator<<(ostream& out, const tuple<T1, T2>& move)
{
    out << "[" << get<0>(move) << ", " << get<1>(move) << "]";
    return out;
}
}

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef std::vector<size_t> vlist_t;

#define MCMC_LATENT_LAYERS_STATE_params(State)                                 \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((entropy_args,, uentropy_args_t, 0))                                      \
    ((verbose,, int, 0))                                                       \
    ((niter,, size_t, 0))


template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCLatentLayersBase, MCMC_LATENT_LAYERS_STATE_params(State))

    template <class... Ts>
    class MCMCLatentLayersState
        : public MCMCLatentLayersBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MCMCLatentLayersBase<Ts...>,
                         MCMC_LATENT_LAYERS_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_LATENT_LAYERS_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCLatentLayersState(ATs&&... as)
            : MCMCLatentLayersBase<Ts...>(as...),
              _block_state(_state._block_state[0].get()._ebstate),
              _measured(_state._measured),
              _edge_sampler(_block_state, _measured ? .25 : 1),
              _vlist(num_vertices(_state._g))
        {
            GILRelease gil_release;

            for (auto e : edges_range(_state._g))
            {
                auto u = source(e, _state._g);
                auto v = target(e, _state._g);
                if (_state._eweight[e] > 0 || (u == v && _state._self_loops))
                    _edges.emplace_back(u, v);
            }

            if (_state._self_loops)
            {
                for (auto v : vertices_range(_state._g))
                {
                    if (edge(v, v, _state._g).second)
                        continue;
                    _edges.emplace_back(v, v);
                }
            }
        }

        typedef decltype(_state._block_state[0].get()._ebstate) block_state_t;
        block_state_t _block_state;
        bool _measured;

        std::vector<std::tuple<size_t, size_t>> _edges;

        std::tuple<size_t, size_t, size_t> _e;

        SBMEdgeSampler<block_state_t> _edge_sampler;

        std::vector<size_t> _vlist;
        std::tuple<int, bool> _null_move = {0, false};

        std::tuple<size_t, size_t, size_t> get_edge()
        {
            return _e;
        }

        size_t node_state(size_t l, size_t u, size_t v)
        {
            auto&& e = _state.get_u_edge(l, u, v);
            if (e == _state._null_edge)
                return 0;
            return _state._block_state[l].get()._eweight[e];
        }

        size_t total_node_state(size_t u, size_t v)
        {
            auto&& e = _state.get_edge(u, v);
            return _state._eweight[e];
        }

        size_t node_state(size_t)
        {
            size_t l, u, v;
            std::tie(l, u, v) = get_edge();
            return node_state(l, u, v);
        }

        template <class T>
        bool skip_node(T&)
        {
            return false;
        }

        template <class RNG>
        std::tuple<int, bool> move_proposal(size_t, RNG& rng)
        {
            auto e = _measured ? _edge_sampler.sample(rng) :
                uniform_sample(_edges, rng);

            std::uniform_int_distribution<size_t> lsample(0, _state._block_state.size()-1);
            size_t l = lsample(rng);
            _e = {l, get<0>(e), get<1>(e)};
            _state._block_state[0].get().internal_move_proposal(get<0>(e), get<1>(e), rng);
            size_t m = node_state(get<0>(_e), get<1>(_e), get<2>(_e));

            std::bernoulli_distribution coin(.5);
            if (coin(rng))
            {
                std::uniform_int_distribution<size_t> lsample(0, _state._block_state.size()-2);
                auto nl = lsample(rng);
                if (nl == l)
                    nl = _state._block_state.size() - 1;
                return {nl, true};
            }
            else
            {
                if (l == 0)
                {
                    std::geometric_distribution<int> sample_m(1./(m + 2));
                    return {sample_m(rng) - m, false};
                }
                else
                {
                    if (m > 0 && coin(rng))
                    {
                        return {-1, false};
                    }
                    else
                    {
                        return {1, false};
                    }
                }
            }
        }

        std::tuple<double, double>
        virtual_move_dS(size_t, const std::tuple<int, bool>& d)
        {
            int dm;
            bool lmove;
            std::tie(dm, lmove) = d;

            if (!lmove && dm == 0)
                return {0, 0};

            size_t l, u, v;
            std::tie(l, u, v) = get_edge();

            double dS = 0;
            double a = 0;

            if (lmove)
            {
                size_t nl = dm;
                size_t m = node_state(l, u, v);

                if (m == 0 || node_state(nl, u, v) > 0)
                {
                    dS = std::numeric_limits<double>::infinity();
                }
                else
                {
                    size_t k1 = 0, k2 = 0;
                    for (size_t i = 0; i < m; ++i)
                    {
                        double ddS = _state.remove_edge_dS(l, u, v, _entropy_args);
                        if (std::isinf(ddS))
                        {
                            dS = std::numeric_limits<double>::infinity();
                            break;
                        }
                        dS += ddS;
                        _state.remove_edge(l, u, v);
                        ++k1;
                    }

                    if (!std::isinf(dS))
                    {
                        for (size_t i = 0; i < m; ++i)
                        {
                            double ddS = _state.add_edge_dS(nl, u, v, _entropy_args);
                            if (std::isinf(ddS))
                            {
                                dS = std::numeric_limits<double>::infinity();
                                break;
                            }
                            dS += ddS;
                            _state.add_edge(nl, u, v);
                            ++k2;
                        }
                    }

                    for (size_t i = 0; i < k2; ++i)
                        _state.remove_edge(nl, u, v);

                    for (size_t i = 0; i < k1; ++i)
                        _state.add_edge(l, u, v);

                    if (_measured)
                    {
                        if (l == 0)
                            a += (_edge_sampler.log_prob(u, v, m, -int(m)) -
                                  _edge_sampler.log_prob(u, v, m, 0));

                        if (nl == 0)
                        {
                            size_t dm = m;
                            size_t m = node_state(nl, u, v);
                            a += (_edge_sampler.log_prob(u, v, m, dm) -
                                  _edge_sampler.log_prob(u, v, m, 0));
                        }
                    }
                }
            }
            else
            {
                if (dm < 0)
                {
                    size_t m_tot = (u == v || _measured) ? 1 - dm : total_node_state(u, v);
                    if (m_tot + dm < 1)
                    {
                        dS = std::numeric_limits<double>::infinity();
                    }
                    else
                    {
                        dS = _state.remove_edge_dS(l, u, v, _entropy_args);
                        size_t j = 0;
                        for (int i = 0; i < -dm-1; ++i)
                        {
                            if (std::isinf(dS))
                                break;
                            _state.remove_edge(l, u, v);
                            dS += _state.remove_edge_dS(l, u, v, _entropy_args);
                            j++;
                        }
                        for (size_t i = 0; i < j; ++i)
                            _state.add_edge(l, u, v);
                    }
                }
                else
                {
                    dS = _state.add_edge_dS(l, u, v, _entropy_args);
                    size_t j = 0;
                    for (int i = 0; i < dm-1; ++i)
                    {
                        if (std::isinf(dS))
                            break;
                        _state.add_edge(l, u, v);
                        dS += _state.add_edge_dS(l, u, v, _entropy_args);
                        j++;
                    }
                    for (size_t i = 0; i < j; ++i)
                        _state.remove_edge(l, u, v);
                }

                size_t m = node_state(l, u, v);

                if (l == 0)
                {
                    if (_measured)
                        a += (_edge_sampler.log_prob(u, v, m, dm) -
                              _edge_sampler.log_prob(u, v, m, 0));

                    size_t nm = m + dm;
                    a -= nm * safelog_fast(m + 1) - (nm + 1) * safelog_fast(m + 2);
                    a += m * safelog_fast(nm + 1) - (m + 1) * safelog_fast(nm + 2);
                }
                else
                {
                    if (m > 0)
                        a -= -log(2);
                    if (m + dm > 0)
                        a += -log(2);
                }
            }

            return std::make_tuple(dS, a);
        }

        void perform_move(size_t, const std::tuple<int, bool>& d)
        {
            int dm;
            bool lmove;
            std::tie(dm, lmove) = d;

            if (!lmove && dm == 0)
                return;

            size_t l, u, v;
            std::tie(l, u, v) = get_edge();
            size_t m = node_state(l, u, v);

            if (lmove)
            {
                size_t nl = dm;
                if (_measured)
                {
                    if (l == 0)
                        _edge_sampler.update_edge(u, v, m, -1);
                }
                for (size_t i = 0; i < m; ++i)
                    _state.remove_edge(l, u, v);
                for (size_t i = 0; i < m; ++i)
                    _state.add_edge(nl, u, v);

                if (_measured)
                {
                    if (nl == 0)
                        _edge_sampler.update_edge(u, v, m, 1);
                }
            }
            else
            {
                if (_measured && l == 0 && dm < 0)
                    _edge_sampler.update_edge(u, v, m, dm);

                if (dm < 0)
                {
                    for (int i = 0; i < -dm; ++i)
                        _state.remove_edge(l, u, v);
                }
                else
                {
                    for (int i = 0; i < dm; ++i)
                        _state.add_edge(l, u, v);
                }

                if (_measured && l == 0 && dm > 0)
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

        template <class T, class M>
        void step(T&, const M&)
        {
        }

        template <class RNG>
        void init_iter(RNG&)
        {
        }
    };
};


} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_LATENT_LAYERS_MCMC_HH
