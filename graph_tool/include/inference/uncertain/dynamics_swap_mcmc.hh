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

#ifndef DYNAMICS_MCMC_SWAP_HH
#define DYNAMICS_MCMC_SWAP_HH

#include "config.h"

#include <vector>
#include <mutex>

#include "graph_tool.hh"
#include "../../support/graph_state.hh"
#include "dynamics.hh"
#include "segment_sampler.hh"
#include "../../../generation/sampler.hh"
#include "openmp.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef multi_array_ref<int64_t,2> elist_t;

#define MCMC_DYNAMICS_STATE_params(State)                                      \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((elist,, elist_t, 0))                                                     \
    ((pmove,, double, 0))                                                      \
    ((ptmove,, double, 0))                                                     \
    ((pswap,, double, 0))                                                      \
    ((entropy_args,, dentropy_args_t, 0))                                      \
    ((verbose,, int, 0))                                                       \
    ((sequential,, bool, 0))                                                   \
    ((deterministic,, bool, 0))                                                \
    ((parallel,, bool, 0))                                                     \
    ((niter,, size_t, 0))


enum class move_t { move = 0, tmove, swap, null };

ostream& operator<<(ostream& s, move_t& v);

template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCDynamicsStateBase, MCMC_DYNAMICS_STATE_params(State))

    template <class... Ts>
    class MCMCDynamicsState
        : public MCMCDynamicsStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MCMCDynamicsStateBase<Ts...>,
                         MCMC_DYNAMICS_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_DYNAMICS_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCDynamicsState(ATs&&... as)
            : MCMCDynamicsStateBase<Ts...>(as...),
              _vlist(num_edges(_state._u)),
              _candidates(num_vertices(_state._u)),
              _vmutex(num_vertices(_state._u)),
              _vlocked(num_vertices(_state._u), false)
        {
            std::iota(_vlist.begin(), _vlist.end(), 0);
            _state._eweight.reserve(2 * _elist.shape()[0]);
            _state._x.reserve(2 * _elist.shape()[0]);
            _dS.resize(get_num_threads());

            std::vector<move_t> moves = {move_t::move, move_t::tmove, move_t::swap};
            std::vector<double> probs = {_pmove, _ptmove, _pswap};
            _move_sampler = Sampler<move_t, mpl::false_>(moves, probs);

            for (auto e : edges_range(_state._u))
                _edges.emplace_back(source(e, _state._u),
                                    target(e, _state._u));
            _swaps.resize(_edges.size());

            for (size_t i = 0; i < _elist.shape()[0]; ++i)
            {
                auto [u, v] = std::tie(_elist[i][0], _elist[i][1]);
                _candidates[u].push_back(v);
                _candidates[v].push_back(u);
            }
        }

        Sampler<move_t, mpl::false_> _move_sampler;

        typedef typename State::xval_t xval_t;

        std::vector<size_t> _vlist;
        std::vector<std::tuple<size_t, size_t>> _edges;
        std::vector<std::vector<size_t>> _candidates;

        struct swap_t
        {
            size_t u;
            size_t v;
            size_t s; // can be null
            size_t t;
            size_t pos_uv;
            size_t pos_st;
        };

        std::vector<swap_t> _swaps;

        constexpr static move_t _null_move = move_t::null;

        std::vector<std::recursive_mutex> _vmutex;
        std::vector<uint8_t> _vlocked;

        std::vector<std::tuple<move_t,double>> _dS;

        std::tuple<size_t, xval_t> edge_state(size_t u, size_t v)
        {
            std::tuple<size_t, xval_t> ret;
            #pragma omp critical (swap_dS)
            ret = _state.edge_state(u, v);
            return ret;
        }

        template <class RNG>
        bool stage_proposal(size_t pos, RNG& rng)
        {
            auto& [u, v, s, t, pos_uv, pos_st] = _swaps[pos];

            std::bernoulli_distribution coin(.5);

            auto iter = uniform_sample_iter(_edges.begin(), _edges.end(), rng);
            std::tie(u, v) = *iter;
            pos_uv = iter - _edges.begin();
            if (!graph_tool::is_directed(_state._u) && coin(rng))
                std::swap(u, v);

            auto [m, x] = edge_state(u, v);

            auto& [move, dS] = _dS[get_thread_num()];
            dS = 0;

            move = _move_sampler(rng);
            switch (move)
            {
            case move_t::move:
                {
                    auto iter = uniform_sample_iter(_edges.begin(), _edges.end(), rng);
                    std::tie(s, t) = *iter;
                    pos_st = iter - _edges.begin();
                    auto [m2, x2] = edge_state(s, t);
                    if (x == x2)
                    {
                        u = v = s = t = numeric_limits<size_t>::max();
                        move = move_t::null;
                        return true;
                    }

                    if (_parallel)
                    {
                        auto ret = std::try_lock(_vmutex[u], _vmutex[v],
                                                 _vmutex[s], _vmutex[t]);
                        if (ret == -1)
                        {
                            if (_vlocked[u] || _vlocked[v] ||
                                _vlocked[s] || _vlocked[t])
                            {
                                _vmutex[u].unlock();
                                _vmutex[v].unlock();
                                _vmutex[s].unlock();
                                _vmutex[t].unlock();
                                return false;
                            }
                            else
                            {
                                _vlocked[u] = true;
                                _vmutex[u].unlock();
                                _vlocked[v] = true;
                                _vmutex[v].unlock();
                                _vlocked[s] = true;
                                _vmutex[s].unlock();
                                _vlocked[t] = true;
                                _vmutex[t].unlock();
                            }
                        }
                    }

                    // swap u, v with s, t

                    if (v == t)
                    {
                        dS += _state._dstate->get_edges_dS({u, s}, v, {x, x2}, {x2, x});
                    }
                    else
                    {
                        dS += _state._dstate->get_edge_dS(u, v, x, x2);
                        dS += _state._dstate->get_edge_dS(s, t, x2, x);
                    }

                    if (!graph_tool::is_directed(_state._u))
                    {
                        if (u == s)
                        {
                            dS += _state._dstate->get_edges_dS({v, t}, u, {x, x2}, {x2, x});
                        }
                        else
                        {
                            dS += _state._dstate->get_edge_dS(v, u, x, x2);
                            dS += _state._dstate->get_edge_dS(t, s, x2, x);
                        }
                    }
                }
                break;
            case move_t::tmove:
                {
                    s = numeric_limits<size_t>::max();
                    auto& c = _candidates[v];
                    if (c.empty())
                    {
                        std::uniform_int_distribution<size_t>
                            vsample(0, num_vertices(_state._u)-1);
                        t = v;
                        while (t == v)
                            t = vsample(rng);
                    }
                    else
                    {
                        t = uniform_sample(c, rng);
                    }

                    if (_verbose > 0)
                        cout << u << " " << v << " " << t << " " << get<0>(edge_state(v, t)) << endl;

                    if (u == t || v == t || u == v || get<0>(edge_state(v, t)) > 0)
                    {
                        u = v = s = t = numeric_limits<size_t>::max();
                        move = move_t::null;
                        return true;
                    }

                    if (_parallel)
                    {
                        auto ret = std::try_lock(_vmutex[u], _vmutex[v], _vmutex[t]);
                        if (ret == -1)
                        {
                            if (_vlocked[u] || _vlocked[v] || _vlocked[t])
                            {
                                _vmutex[u].unlock();
                                _vmutex[v].unlock();
                                _vmutex[t].unlock();
                                return false;
                            }
                            else
                            {
                                _vlocked[u] = true;
                                _vmutex[u].unlock();
                                _vlocked[v] = true;
                                _vmutex[v].unlock();
                                _vlocked[t] = true;
                                _vmutex[t].unlock();
                            }
                        }
                    }

                    // swap u with t

                    //t->v add and u->v remove
                    dS += _state._dstate->get_edges_dS({u, t}, v, {x, 0}, {0, x});

                    if (!graph_tool::is_directed(_state._u))
                    {
                        //t<-v add
                        dS += _state._dstate->get_edge_dS(v, t, 0, x);
                        //u<-v remove
                        dS += _state._dstate->get_edge_dS(v, u, x, 0);
                    }
                }
                break;
            case move_t::swap:
                {
                    auto iter = uniform_sample_iter(_edges.begin(), _edges.end(), rng);
                    std::tie(s, t) = *iter;
                    pos_st = iter - _edges.begin();
                    if (!graph_tool::is_directed(_state._u) && coin(rng))
                        std::swap(s, t);

                    if (pos_st == pos_uv || u == v || u == t || v == t || u == s || v == s || s == t ||
                        get<0>(edge_state(u, t)) > 0 ||
                        get<0>(edge_state(s, v)) > 0)
                    {
                        u = v = s = t = numeric_limits<size_t>::max();
                        move = move_t::null;
                        return true;
                    }

                    if (_parallel)
                    {
                        auto ret = std::try_lock(_vmutex[u], _vmutex[v], _vmutex[s], _vmutex[t]);
                        if (ret == -1)
                        {
                            if (_vlocked[u] || _vlocked[v] || _vlocked[s] || _vlocked[t])
                            {
                                _vmutex[u].unlock();
                                _vmutex[v].unlock();
                                _vmutex[s].unlock();
                                _vmutex[t].unlock();
                                return false;
                            }
                            else
                            {
                                _vlocked[u] = true;
                                _vmutex[u].unlock();
                                _vlocked[v] = true;
                                _vmutex[v].unlock();
                                _vlocked[s] = true;
                                _vmutex[s].unlock();
                                _vlocked[t] = true;
                                _vmutex[t].unlock();
                            }
                        }
                    }

                    auto [m2, x2] = edge_state(s, t);

                    //dS for v<-s add and v<-u remove
                    dS += _state._dstate->get_edges_dS({u, s}, v, {x, 0}, {0, x2});
                    //dS for t<-u add and t<-s remove
                    dS += _state._dstate->get_edges_dS({s, u}, t, {x2, 0}, {0, x});

                    if (!graph_tool::is_directed(_state._u))
                    {
                        //dS for u<-t add and u<-v remove
                        dS += _state._dstate->get_edges_dS({v, t}, u, {x, 0}, {0, x});
                        //dS for s<-v add and s<-t remove
                        dS += _state._dstate->get_edges_dS({t, v}, s, {x2, 0}, {0, x2});
                    }
                }
                break;
            default:
                break;
            }

            return true;
        }

        void proposal_unlock(size_t pos)
        {
            if (!_parallel)
                return;

            auto& [u, v, s, t, pos_uv, pos_st] = _swaps[pos];

            for (auto x : std::array<size_t, 4>({u, v, s, t}))
            {
                if (x == numeric_limits<size_t>::max())
                    continue;
                _vlocked[x] = false;
            }
        }

        template <class RNG>
        move_t move_proposal(size_t pos, RNG& rng)
        {
            if (!_parallel)
                stage_proposal(pos, rng);
            auto& [move, dS] = _dS[get_thread_num()];
            return move;
        }

        template <class Lock>
        void perform_move(size_t pos, move_t move, Lock&)
        {
            auto& [u, v, s, t, pos_uv, pos_st] = _swaps[pos];

            auto [m, x] = edge_state(u, v);

            size_t m2;
            double x2;

            switch (move)
            {
            case move_t::move:
                std::tie(m2, x2) = edge_state(s, t);
                if (m2 < m)
                    _state.remove_edge(u, v, m - m2);
                if (m < m2)
                    _state.add_edge(u, v, m2 - m, x2);
                if (m2 > 0 && m > 0)
                    _state.update_edge(u, v, x2);
                if (m < m2)
                    _state.remove_edge(s, t, m2 - m);
                if (m > m2)
                    _state.add_edge(s, t, m - m2, x);
                if (m2 > 0 && m > 0)
                    _state.update_edge(s, t, x);
                _edges[pos_uv] = {s, t};
                _edges[pos_st] = {u, v};
                break;
            case move_t::tmove:
                _state.remove_edge(u, v, m);
                _state.add_edge(t, v, m, x);
                _edges[pos_uv] = {t, v};
                break;
            case move_t::swap:
                std::tie(m2, x2) = edge_state(s, t);
                _state.remove_edge(u, v, m);
                _state.remove_edge(s, t, m2);
                _state.add_edge(u, t, m, x);
                _state.add_edge(s, v, m2, x2);
                _edges[pos_uv] = {u, t};
                _edges[pos_st] = {s, v};
                break;
            default:
                break;
            }
        }

        class DummyLock
        {
        public:
            void unlock() {}
        };

        void perform_move(size_t pos, move_t& move)
        {
            DummyLock lock;
            perform_move(pos, move, lock);
        }

        std::tuple<double, double>
        virtual_move_dS(size_t pos, move_t move)
        {
            auto& [u, v, s, t, pos_uv, pos_st] = _swaps[pos];
            auto& [move_, dS] = _dS[get_thread_num()];
            dS *= _entropy_args.alpha;

            assert(move == move_);

            auto ea = _entropy_args;
            if (!ea.xdist)
                ea.xl1 = 0;
            ea.normal = false;

            auto [m, x] = edge_state(u, v);
            size_t m2;
            double x2;

            switch (move)
            {
            case move_t::move:
                std::tie(m2, x2) = edge_state(s, t);
                #pragma omp critical (swap_dS)
                {
                    if (m2 < m)
                    {
                        dS += _state.remove_edge_dS(u, v, m - m2, ea, false);
                        _state.remove_edge(u, v, m - m2);
                    }
                    if (m < m2)
                    {
                        dS += _state.add_edge_dS(u, v, m2 - m, x2, ea, false);
                        _state.add_edge(u, v, m2 - m, x2);
                    }
                    if (m2 > 0 && m > 0)
                    {
                        dS += _state.update_edge_dS(u, v, x2, ea, false);
                         _state.update_edge(u, v, x2);
                    }
                    if (m < m2)
                    {
                        dS += _state.remove_edge_dS(s, t, m2 - m, ea, false);
                        _state.remove_edge(s, t, m2 - m);
                    }
                    if (m > m2)
                    {
                        dS += _state.add_edge_dS(s, t, m - m2, x, ea, false);
                        _state.add_edge(s, t, m - m2, x);
                    }

                    if (m2 > 0 && m > 0)
                        dS += _state.update_edge_dS(s, t, x, ea, false);

                    if (m > m2)
                        _state.remove_edge(s, t, m - m2);
                    if (m < m2)
                        _state.add_edge(s, t, m2 - m, x2);
                    if (m2 > 0 && m > 0)
                        _state.update_edge(u, v, x);
                    if (m < m2)
                        _state.remove_edge(u, v, m2 - m);
                    if (m2 < m)
                        _state.add_edge(u, v, m - m2, x);
                }
                break;
            case move_t::tmove:
                #pragma omp critical (swap_dS)
                {
                    dS += _state.remove_edge_dS(u, v, m, ea, false);
                    _state.remove_edge(u, v, m, [](){}, false);
                    dS += _state.add_edge_dS(t, v, m, x, ea, false);
                    _state.add_edge(u, v, m, x, [](){}, false);
                }
                break;
            case move_t::swap:
                std::tie(m2, x2) = edge_state(s, t);
                #pragma omp critical (swap_dS)
                {
                    dS += _state.remove_edge_dS(u, v, m, ea, false);
                    _state.remove_edge(u, v, m, [](){}, false);
                    dS += _state.remove_edge_dS(s, t, m2, ea, false);
                    _state.remove_edge(s, t, m2, [](){}, false);

                    dS += _state.add_edge_dS(u, t, m, x, ea, false);
                    _state.add_edge(u, t, m, x, [](){}, false);

                    dS += _state.add_edge_dS(s, v, m2, x2, ea, false);

                    _state.remove_edge(u, t, m, [](){}, false);
                    _state.add_edge(s, t, m2, x2, [](){}, false);
                    _state.add_edge(u, v, m, x, [](){}, false);
                }
                break;
            default:
                break;
            }

            return {dS, 0.};
        }

        double entropy()
        {
            double S = _state.entropy(_entropy_args);
            if (_entropy_args.sbm)
                S += _state._block_state.entropy(_entropy_args);
            return S;
        }

        bool is_deterministic()
        {
            return _deterministic;
        }

        bool is_sequential()
        {
            return _sequential;
        }

        bool is_parallel()
        {
            return _parallel;
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
        constexpr bool skip_node(T&)
        {
            return false;
        }

        template <class T>
        constexpr bool node_state(T&)
        {
            return false;
        }

        template <class T>
        void step(T&, move_t&)
        {
        }

        template <class RNG>
        void init_iter(RNG&)
        {
        }
    };
};

} // graph_tool namespace

#endif //DYNAMICS_MCMC_SWAP_HH
