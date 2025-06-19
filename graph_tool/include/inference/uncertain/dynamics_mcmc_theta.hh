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

#ifndef DYNAMICS_THETA_MCMC_HH
#define DYNAMICS_THETA_MCMC_HH

#include "config.h"

#include <vector>
#include <mutex>

#include "graph_tool.hh"
#include "../../support/graph_state.hh"
#include "../../support/fibonacci_search.hh"
#include "dynamics.hh"
#include "segment_sampler.hh"
#include "openmp.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_DYNAMICS_STATE_params(State)                                      \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((pold,, double, 0))                                                       \
    ((pnew,, double, 0))                                                       \
    ((maxiter,, size_t, 0))                                                    \
    ((tol,, double, 0))                                                        \
    ((entropy_args,, dentropy_args_t, 0))                                      \
    ((verbose,, int, 0))                                                       \
    ((deterministic,, bool, 0))                                                \
    ((sequential,, bool, 0))                                                   \
    ((parallel,, bool, 0))                                                     \
    ((niter,, size_t, 0))


template <class State>
struct MCMCTheta
{
    GEN_STATE_BASE(MCMCDynamicsStateBase, MCMC_DYNAMICS_STATE_params(State))

    enum class xmove_t { xold = 0, xnew};

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
              _vlist(num_vertices(_state._u)),
              _vmutex(_vlist.size())
        {
            std::iota(_vlist.begin(), _vlist.end(), 0);
            _xcaches.resize(get_num_threads());

            if (_state._disable_tdist)
                std::tie(_pold, _pnew) = std::make_tuple(0., 1.);

            std::vector<xmove_t> moves
                = {xmove_t::xold, xmove_t::xnew};
            std::vector<double> probs
                = {_pold, _pnew};
            _move_sampler = Sampler<xmove_t, mpl::false_>(moves, probs);
        }

        Sampler<xmove_t, mpl::false_> _move_sampler;

        typedef typename State::tval_t tval_t;
        typedef tval_t move_t;

        std::vector<size_t> _vlist;

        constexpr static tval_t _null_move = numeric_limits<tval_t>::max();

        std::vector<std::mutex> _vmutex;

        bool proposal_lock(size_t)
        {
            return true;
        }

        void proposal_unlock(size_t)
        {
        }

        move_t node_state(size_t v)
        {
            return _state._theta[v];
        }

        template <class T>
        constexpr bool skip_node(T&)
        {
            return false;
        }

        std::vector<std::tuple<xmove_t, tval_t, double,
                               std::map<tval_t, double>>> _xcaches;

        std::array<double, 2> _ws;

        template <class RNG>
        double stage_proposal(size_t v, RNG& rng)
        {
            if (!proposal_lock(v))
                return false;

            auto& [move, nx, dS, xcache] = _xcaches[get_thread_num()];
            nx = numeric_limits<double>::quiet_NaN();
            dS = numeric_limits<double>::quiet_NaN();
            xcache.clear();

            move = _move_sampler(rng);
            std::tie(nx, dS, xcache) = sample_nx(v, move == xmove_t::xold, rng);

            // auto x = _state._theta[v];
            // if (move == xmove_t::xold)
            // {
            //     std::shared_lock lock(_state._t_mutex);
            //     nx = _state.find_closest(_state._tvals, nx);
            //     dS = _state.dstate_node_dS(v, nx - x, _entropy_args) +
            //         _entropy_args.tl1 * (abs(nx) - abs(x));
            // }

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
        move_t move_proposal(size_t v, RNG& rng)
        {
            if (!_parallel)
                stage_proposal(v, rng);

            auto& [move, nx, dS, xcache] = _xcaches[get_thread_num()];

            auto ea = _entropy_args;
            if (!ea.tdist)
                ea.tl1 = 0;

            dS += _state.update_node_dS(v, nx, ea, false);

            return nx;
        }

        template <class Lock>
        void perform_move(size_t v, move_t move, Lock&)
        {
            _state.update_node(v, move);
        }

        void perform_move(size_t v, move_t move)
        {
            _state.update_node(v, move);
        }

        std::tuple<double, double>
        virtual_move_dS(size_t v, move_t nx)
        {
            auto x = _state._theta[v];
            if (x == nx)
                return {0., 0.};

            auto& [move, nx_, dS, xcache] = _xcaches[get_thread_num()];

            double lf = 0;
            double lb = 0;
            double a = 0;
            if (!std::isinf(_beta))
            {
                switch (move)
                {
                case xmove_t::xold:
                    lf = sample_old_x_lprob(nx, xcache);
                    lf += log(_pold) - log(_pold + _pnew);

                    if (_pnew > 0)
                    {
                        lb += log(_pnew) - log(_pold + _pnew);
                        lb += sample_new_x_lprob(x, xcache);
                    }
                    else
                    {
                        lb = -numeric_limits<double>::infinity();
                    }

                    if (_pold > 0 && !_state._disable_tdist && (_state.get_count(_state._thist, x) > 1))
                    {
                        lb = log_sum_exp(lb,
                                         log(_pold) - log(_pold + _pnew) +
                                         sample_old_x_lprob(x, xcache));
                    }
                    break;
                case xmove_t::xnew:
                    lf = log(_pnew) - log(_pold + _pnew);
                    lf += sample_new_x_lprob(nx, xcache);

                    if (_pnew > 0)
                    {
                        lb += log(_pnew) - log(_pold + _pnew);
                        lb += sample_new_x_lprob(x, xcache);
                    }
                    else
                    {
                        lb = -numeric_limits<double>::infinity();
                    }

                    if (_pold > 0 && !_state._disable_tdist && (_state.get_count(_state._thist, x) > 1))
                    {
                        lb = log_sum_exp(lb,
                                         log(_pold) - log(_pold + _pnew) +
                                         sample_old_x_lprob(x, xcache));
                    }
                    break;
                }

                a = lb - lf;
            }

            if (_verbose)
                cout << v << ", x: " << x << ", nx: "
                     << nx << ", dS: " << dS << ", lf: " << lf << ", lb: "
                     << lb << ", a: " << a << ", -dS + a: " << -dS + a << endl;

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
        auto sample_nx(size_t v, bool told, RNG& rng)
        {
            auto x = _state._theta[v];
            auto [nx, xcache] = _state.sample_t(v, _beta, _maxiter,
                                                _tol, _entropy_args, told, rng);
            double dS;
            auto iter = xcache.find(nx);
            if (iter == xcache.end())
            {
                dS = (_state.dstate_node_dS(v, nx - x, _entropy_args) +
                      (_state.node_x_S(nx, _entropy_args) -
                       _state.node_x_S(x, _entropy_args)));
            }
            else
            {
                dS = iter->second;
            }

            return std::make_tuple(nx, dS, xcache);
        }

        template <class Class>
        double sample_new_x_lprob(tval_t nx, Class& xcache)
        {
            return _state.sample_t_lprob(nx, xcache, _beta);
        }

        template <class Class>
        double sample_old_x_lprob(tval_t nx, Class& xcache)
        {
            auto [a, b] = _state.get_close_int(_state._tvals, nx);
            SegmentSampler seg = _state.get_seg_sampler(xcache, _beta);
            return seg.lprob_int(a, b);
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

#endif //DYNAMICS_THETA_MCMC_HH
