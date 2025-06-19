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

#ifndef DYNAMICS_MCMC_HH
#define DYNAMICS_MCMC_HH

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
    ((pold,, double, 0))                                                       \
    ((pnew,, double, 0))                                                       \
    ((pxu,, double, 0))                                                        \
    ((pm,, double, 0))                                                         \
    ((premove,, double, 0))                                                    \
    ((binary,, bool, 0))                                                       \
    ((maxiter,, size_t, 0))                                                    \
    ((tol,, double, 0))                                                        \
    ((entropy_args,, dentropy_args_t, 0))                                      \
    ((verbose,, int, 0))                                                       \
    ((sequential,, bool, 0))                                                   \
    ((deterministic,, bool, 0))                                                \
    ((parallel,, bool, 0))                                                     \
    ((niter,, size_t, 0))

template <class T1, class T2>
ostream& operator<<(ostream& s, std::tuple<T1, T2>& v)
{
    s << "(" << get<0>(v) << ", " << get<1>(v) << ")";
    return s;
}

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

        enum class xmove_t { xold = 0, xnew, m, remove };

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCDynamicsState(ATs&&... as)
            : MCMCDynamicsStateBase<Ts...>(as...),
              _vlist(_elist.shape()[0]),
              _edges(_elist.shape()[0]),
              _vmutex(num_vertices(_state._u)),
              _vlocked(num_vertices(_state._u), false)
        {
            std::iota(_vlist.begin(), _vlist.end(), 0);
            _state._eweight.reserve(3 * _vlist.size());
            _state._x.reserve(3 * _vlist.size());
            _xcaches.resize(get_num_threads());
// #ifndef NDEBUG
//             for (auto v : vertices_range(_state._u))
//                 _state._dstate.check_m(_state, v);
// #endif
            if (_state._disable_xdist)
                std::tie(_pold, _pnew) = std::make_tuple(0., _pold + _pnew);
            if (_binary)
                _pm = 0;
            std::vector<xmove_t> moves
                = {xmove_t::xold, xmove_t::xnew, xmove_t::m, xmove_t::remove};
            std::vector<double> probs
                = {_pold, _pnew, _pm, _premove};
            _all_move_sampler = Sampler<xmove_t, mpl::false_>(moves, probs);
            moves = {xmove_t::xold, xmove_t::xnew};
            probs = {_pold, _pnew};
            _new_move_sampler = Sampler<xmove_t, mpl::false_>(moves, probs);

            for (size_t i = 0; i < _vlist.size(); ++i)
                _edges[i] = {_elist[i][0], _elist[i][1]};
        }

        Sampler<xmove_t, mpl::false_> _all_move_sampler;
        Sampler<xmove_t, mpl::false_> _new_move_sampler;

        typedef typename State::xval_t xval_t;

        std::vector<size_t> _vlist;
        std::vector<std::tuple<size_t, size_t>> _edges;

        typedef std::tuple<int, xval_t> move_t;

        constexpr static move_t _null_move = {0, 0};

        std::bernoulli_distribution _xmove;

        std::vector<std::mutex> _vmutex;
        std::vector<uint8_t> _vlocked;

        std::vector<std::tuple<xmove_t, xval_t, double,
                               std::map<xval_t, double>>> _xcaches;

        bool proposal_lock(size_t pos)
        {
            if (!_parallel)
                return true;

            auto [u, v] = get_edge(pos);

            if (graph_tool::is_directed(_state._u) || u == v)
            {
                bool ret = _vmutex[v].try_lock();
                if (ret)
                {
                    if (_vlocked[v])
                    {
                        _vmutex[v].unlock();
                        return false;
                    }
                    else
                    {
                        _vlocked[v] = true;
                        _vmutex[v].unlock();
                        return true;
                    }
                }
                return false;
            }
            else
            {
                auto ret = std::try_lock(_vmutex[u], _vmutex[v]);
                if (ret == -1)
                {
                    if (_vlocked[u] || _vlocked[v])
                    {
                        _vmutex[u].unlock();
                        _vmutex[v].unlock();
                        return false;
                    }
                    else
                    {
                        _vlocked[u] = true;
                        _vlocked[v] = true;
                        _vmutex[u].unlock();
                        _vmutex[v].unlock();
                        return true;
                    }
                }
                return false;
            }
        }

        void proposal_unlock(size_t pos)
        {
            if (!_parallel)
                return;
            auto [u, v] = get_edge(pos);
            if (graph_tool::is_directed(_state._u) || u == v)
                _vlocked[v] = false;
            else
                _vlocked[u] = _vlocked[v] = false;
        }

        std::tuple<size_t, size_t> get_edge(size_t pos)
        {
            return _edges[pos];
        }

        std::tuple<size_t, xval_t> edge_state(size_t u, size_t v)
        {
            return _state.edge_state(u, v);
        }

        move_t node_state(size_t pos)
        {
            auto [u, v] = get_edge(pos);
            auto [m, x] = edge_state(u, v);
            return {m, x};
        }

        template <class T>
        constexpr bool skip_node(T&)
        {
            return false;
        }

        template <class RNG>
        bool stage_proposal(size_t pos, RNG& rng)
        {
            if (!proposal_lock(pos))
                return false;

            auto [u, v] = get_edge(pos);
            auto [m, x] = edge_state(u, v);

            auto& [move, nx, dS, xcache] = _xcaches[get_thread_num()];
            nx = numeric_limits<double>::quiet_NaN();
            dS = numeric_limits<double>::quiet_NaN();
            xcache.clear();

            if (m == 0)
                move = _state._xvals.empty() ? xmove_t::xnew : _new_move_sampler(rng);
            else
                move = _all_move_sampler(rng);

            if (move == xmove_t::xnew || !std::isinf(_beta))
                std::tie(nx, dS, xcache) = sample_nx(u, v, move == xmove_t::xold, rng);

            std::bernoulli_distribution xu(_pxu);

            bool unif = false;
            switch (move)
            {
            case xmove_t::remove:
                nx = 0;
                dS = _state.dstate_edge_dS(u, v, x, 0, _entropy_args);
                dS += (_state.edge_x_S(0, _entropy_args) -
                       _state.edge_x_S(x, _entropy_args));
                break;
            case xmove_t::xold:
                unif = xu(rng);
                if (unif)
                {
                    nx = uniform_sample(_state._xvals, rng);
                }
                else
                {
                    if (std::isinf(_beta))
                    {
                        std::tie(nx, dS, xcache) = sample_nx(u, v, true, rng);
                    }
                    else
                    {
                        std::shared_lock lock(_state._x_mutex);
                        nx = _state.find_closest(_state._xvals, nx);
                    }
                }
                if (unif || !std::isinf(_beta))
                {
                    dS = _state.dstate_edge_dS(u, v, x, nx, _entropy_args);
                    dS += (_state.edge_x_S(nx, _entropy_args) -
                           _state.edge_x_S(x, _entropy_args));
                }
                break;
            case xmove_t::m:
                nx = x;
                dS = 0;
                break;
            default:
                break;
            }

            assert(!std::isinf(nx) && !std::isnan(nx));
            assert(!std::isinf(dS) && !std::isnan(dS));

            return true;
        }

        template <class Cache>
        std::tuple<double, double> get_min(Cache& c)
        {
            auto iter = std::min_element(c.begin(), c.end(),
                                         [&](auto& a, auto& b)
                                         {
                                             return get<1>(a) < get<1>(b);
                                         });
            return *iter;
        }

        template <class RNG>
        size_t sample_m(size_t m, RNG& rng)
        {
            if (std::isinf(_beta))
                return 1;
            double a = m + .5;
            double p = 1/(a+1);
            std::geometric_distribution<size_t> random(p);
            return random(rng) + 1;
        }

        double sample_m_lprob(size_t nm, size_t m)
        {
            if (_binary)
                return 0;
            double a = m + .5;
            double p = 1/(a+1);
            return (nm - 1) * log1p(-p) + log(p);
        }

        std::array<double, 3> _ws;


        template <class RNG>
        move_t move_proposal(size_t pos, RNG& rng)
        {
            if (!_parallel)
                stage_proposal(pos, rng);

            auto [u, v] = get_edge(pos);
            auto [m, x] = edge_state(u, v);

            auto& [move, nx, dS, xcache] = _xcaches[get_thread_num()];
            assert(!std::isinf(dS));

            int dm = 0;
            if (m == 0 || move == xmove_t::m)
            {
                if (_binary)
                    dm = (m == 0) ? 1 : 0;
                else
                    dm = int(sample_m(m, rng)) - int(m);
            }

            if (move == xmove_t::remove)
                dm = -m;

            return {dm, nx};
        }

        template <class Lock>
        void perform_move(size_t pos, move_t& move, Lock& lock)
        {
            auto [u, v] = get_edge(pos);
            auto [m, x] = _state.edge_state(u, v);
            auto& [dm, nx] = move;
            if (dm == 0 && nx == x)
                return;

            if (dm == 0)
            {
                _state.update_edge(u, v, nx, [&](){ lock.unlock(); });
            }
            else if (dm > 0)
            {
                if (m == 0)
                {
                    _state.add_edge(u, v, dm, nx, [&](){ lock.unlock(); });
                }
                else
                {
                    _state.add_edge(u, v, dm, nx);
                    _state.update_edge(u, v, nx, [&](){ lock.unlock(); });
                }
            }
            else
            {
                if (m + dm == 0)
                {
                    _state.remove_edge(u, v, -dm, [&](){ lock.unlock(); });
                }
                else
                {
                    _state.remove_edge(u, v, -dm);
                    _state.update_edge(u, v, nx, [&](){ lock.unlock(); });
                }
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
        virtual_move_dS(size_t pos, move_t& mv)
        {
            auto [u, v] = get_edge(pos);
            auto [m, x] = _state.edge_state(u, v);
            auto& [dm, nx] = mv;

            if (dm == 0 && nx == x)
                return {0., 0.};

            auto& [move, nx_, dS_, xcache] = _xcaches[get_thread_num()];
            double dS = dS_;

            auto ea = _entropy_args;
            if (!ea.xdist)
                ea.xl1 = 0;
            ea.normal = false;

            if (dm == 0)
            {
                dS += _state.update_edge_dS(u, v, nx, ea, false);
            }
            else if (dm > 0)
            {
                dS += _state.add_edge_dS(u, v, dm, nx, ea, false);
                if (m > 0)
                    dS += _state.update_edge_dS(u, v, nx, ea, false);
            }
            else
            {
                dS += _state.remove_edge_dS(u, v, -dm, ea, false);
                if (m + dm > 0)
                    dS += _state.update_edge_dS(u, v, nx, ea, false);
            }

            assert(!std::isinf(dS) && !std::isnan(dS));

            double a = 0;

            double lf = 0;
            double lb = 0;
            if (!std::isinf(_beta))
            {
                switch (move)
                {
                case xmove_t::xold:
                    lf += sample_old_x_lprob(nx, xcache);
                    if (m == 0)
                    {
                        lf += log(_pold) - log(_pold + _pnew);
                        lf += sample_m_lprob(m + dm, m);

                        lb += log(_premove) - log(_pold + _pnew + _pm + _premove);
                    }
                    else
                    {
                        lf += log(_pold) - log(_pold + _pnew + _pm + _premove);

                        if (_pnew > 0)
                        {
                            lb += log(_pnew) - log(_pold + _pnew + _pm + _premove);
                            lb += sample_new_x_lprob(x, xcache);
                        }
                        else
                        {
                            lb = -numeric_limits<double>::infinity();
                        }

                        if (_pold > 0 && !_state._disable_xdist && (_state.get_count(_state._xhist, x) > 1))
                        {
                            lb = log_sum_exp(lb,
                                             log(_pold) - log(_pold + _pnew + _pm + _premove) +
                                             sample_old_x_lprob(x, xcache));
                        }
                    }
                    break;
                case xmove_t::xnew:
                    lf += sample_new_x_lprob(nx, xcache);
                    if (m == 0)
                    {
                        lf += log(_pnew) - log(_pold + _pnew);
                        lf += sample_m_lprob(m + dm, m);

                        lb += log(_premove) - log(_pold + _pnew + _pm + _premove);
                    }
                    else
                    {
                        lf += log(_pnew) - log(_pold + _pnew + _pm + _premove);

                        if (_pnew > 0)
                        {
                            lb += log(_pnew) - log(_pold + _pnew + _pm + _premove);
                            lb += sample_new_x_lprob(x, xcache);
                        }
                        else
                        {
                            lb = -numeric_limits<double>::infinity();
                        }

                        if (_pold > 0 && !_state._disable_xdist && (_state.get_count(_state._xhist, x) > 1))
                        {
                            lb = log_sum_exp(lb,
                                             log(_pold) - log(_pold + _pnew + _pm + _premove) +
                                             sample_old_x_lprob(x, xcache));
                        }
                    }
                    break;
                case xmove_t::remove:
                    lf = log(_premove) - log(_pold + _pnew + _pm + _premove);

                    if (_pnew > 0)
                    {
                        lb += log(_pnew) - log(_pold + _pnew);
                        lb += sample_new_x_lprob(x, xcache);
                    }
                    else
                    {
                        lb = -numeric_limits<double>::infinity();
                    }

                    if (_pold > 0 && !_state._disable_xdist && (_state.get_count(_state._xhist, x) > 1))
                    {
                        lb = log_sum_exp(lb,
                                         log(_pold) - log(_pold + _pnew) +
                                         sample_old_x_lprob(x, xcache));
                    }

                    lb += sample_m_lprob(m, 0);
                    break;
                case xmove_t::m:
                    lf = sample_m_lprob(m + dm, m);
                    lb = sample_m_lprob(m, m + dm);
                    break;
                }
                a = lb - lf;
            }

            if (_verbose)
                cout << "u: " << u << ", v: " << v << ", m: " << m
                     << ", m + dm: " << m + dm << ", nx: "
                     << nx << ", dS: " << dS << ", lf: " << lf << ", lb: " << lb << ", a: "
                     << a << ", -dS + a: " << -dS + a << endl;

            return {dS, a};
        }

        double entropy()
        {
            double S = _state.entropy(_entropy_args);
            if (_entropy_args.sbm)
                S += _state._block_state.entropy(_entropy_args);
            return S;
        }

        template <class RNG>
        auto sample_nx(size_t u, size_t v, bool xold, RNG& rng)
        {
            auto [m, x] = edge_state(u, v);
            auto [nx, xcache] = _state.sample_x(u, v, _beta, _maxiter,
                                                _tol, _entropy_args, xold, rng);
            double dS;
            auto iter = xcache.find(nx);
            if (iter == xcache.end())
            {
                dS = (_state.dstate_edge_dS(u, v, x, nx, _entropy_args) +
                      (_state.edge_x_S(nx, _entropy_args) -
                       _state.edge_x_S(x, _entropy_args)));
            }
            else
            {
                dS = iter->second;
            }

            assert(!std::isinf(nx) && !std::isnan(nx));
            assert(!std::isinf(dS) && !std::isnan(dS));

            return std::make_tuple(nx, dS, xcache);
        }

        template <class Cache>
        double sample_new_x_lprob(xval_t nx, Cache& xcache)
        {
            return _state.sample_x_lprob(nx, xcache, _beta);
        }

        template <class Cache>
        double sample_old_x_lprob(xval_t nx, Cache& xcache)
        {
            auto [a, b] = _state.get_close_int(_state._xvals, nx);
            SegmentSampler seg = _state.get_seg_sampler(xcache, _beta);
            double l = seg.lprob_int(a, b);
            return log_sum_exp(log1p(-_pxu) + l,
                               log(_pxu) - log(_state._xvals.size()));
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

#endif //DYNAMICS_MCMC_HH
