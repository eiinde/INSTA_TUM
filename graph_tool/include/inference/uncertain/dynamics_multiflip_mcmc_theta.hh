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

#ifndef DYNAMICS_MULTIFLIP_MCMC_THETA_HH
#define DYNAMICS_MULTIFLIP_MCMC_THETA_HH

#include "config.h"

#include <vector>
#include <algorithm>

#include "graph_tool.hh"
#include "../../support/graph_state.hh"
#include "dynamics.hh"
#include "segment_sampler.hh"
#include "openmp.hh"

#include "idx_map.hh"
#include "../../loops/merge_split.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_DYNAMICS_STATE_params(State)                                      \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((psplit,, double, 0))                                                     \
    ((pmerge,, double, 0))                                                     \
    ((pmergesplit,, double, 0))                                                \
    ((pmovelabel,, double, 0))                                                 \
    ((nproposal, &, vector<size_t>&, 0))                                       \
    ((nacceptance, &, vector<size_t>&, 0))                                     \
    ((gibbs_sweeps,, size_t, 0))                                               \
    ((maxiter,, size_t, 0))                                                    \
    ((tol,, double, 0))                                                        \
    ((entropy_args,, dentropy_args_t, 0))                                      \
    ((verbose,, int, 0))                                                       \
    ((force_move,, bool, 0))                                                   \
    ((niter,, double, 0))

template <class State>
struct MCMCTheta
{
    GEN_STATE_BASE(MCMCDynamicsStateBase, MCMC_DYNAMICS_STATE_params(State))

    template <class... Ts>
    class MCMCDynamicsStateImp
        : public MCMCDynamicsStateBase<Ts...>,
          public MergeSplitStateBase
    {
    public:
        GET_PARAMS_USING(MCMCDynamicsStateBase<Ts...>,
                         MCMC_DYNAMICS_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_DYNAMICS_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCDynamicsStateImp(ATs&&... as)
           : MCMCDynamicsStateBase<Ts...>(as...)
        {
            _dS.resize(get_num_threads());
        }

        typedef typename State::tval_t tval_t;

        constexpr static tval_t _null_group = std::numeric_limits<tval_t>::infinity();

        constexpr static double _psingle = 0;

        constexpr static double _psrandom = 1;
        constexpr static double _psscatter = 1;
        constexpr static double _pscoalesce = 1;

        constexpr static bool _parallel = true;
        constexpr static bool _relabel = true;

        template <class F>
        void iter_nodes(F&& f)
        {
            for (auto v : vertices_range(_state._u))
                f(v);
        }

        template <class F>
        void iter_groups(F&& f)
        {
            for (auto r : _state._tvals)
                f(r);
        }

        tval_t get_group(size_t v)
        {
            return _state._theta[v];
        }


        template <bool sample_branch=true, class RNG, class VS = std::array<size_t,0>>
        tval_t sample_new_group(size_t v, RNG& rng, VS&& except = VS())
        {

            double x;
            do
            {
                x = get<0>(_state.sample_t(v, 1, _maxiter, _tol, _entropy_args,
                                           false, rng));
            }
            while (std::find(except.begin(), except.end(), x) != except.end());
            return x;
        }

        std::vector<std::array<std::tuple<double, double>,2>> _dS;
        std::mutex _move_mutex;

        void virtual_move_lock(size_t idx, double r, double s)
        {
            virtual_move_lock(idx, r, std::array<double,1>{s});
        }

        bool _move_locked = false;
        template <size_t d>
        void virtual_move_lock(size_t v, double r, const std::array<double,d>& s)
        {
            auto& dS = _dS[get_thread_num()];
            for (size_t i = 0; i < 2; ++i)
                dS[i] = {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN()};
            for (size_t i = 0; i < d; ++i)
            {
                if (std::isinf(s[i]))
                    continue;
                dS[i] = {s[i],
                         _state.dstate_node_dS(v, s[i] - r, _entropy_args) +
                         (_state.node_x_S(s[i], _entropy_args) -
                          _state.node_x_S(r, _entropy_args))};
            }
            _move_mutex.lock();
            _move_locked = true;
        }

        constexpr void virtual_move_unlock(size_t)
        {
            _move_locked = false;
            _move_mutex.unlock();
        }

        void move_node(size_t v, double r, bool /*cache*/)
        {
            if (!_move_locked)
            {
                _state.update_node(v, r);
            }
            else
            {
                _state.update_node(v, r);
                _move_mutex.unlock();
            }
        }

        constexpr void reserve_empty_groups(size_t)
        {
        }

        constexpr bool allow_move(size_t, size_t)
        {
            return true;
        }

        double virtual_move(size_t v, double, double s)
        {
            auto& dSs = _dS[get_thread_num()];
            double dS = (get<0>(dSs[0]) == s) ? get<1>(dSs[0]) : get<1>(dSs[1]);
            auto ea = _entropy_args;
            if (!ea.tdist)
                ea.tl1 = 0;
            return _state.update_node_dS(v, s, ea, false) + dS;
        }

        template <class RNG>
        double sample_group(size_t v, bool, //allow_empty,
                            RNG& rng)
        {
            auto r = _state._theta[v];

            auto first = _state._tvals.begin();
            auto last = _state._tvals.end();
            last--;

            auto iter = std::lower_bound(_state._tvals.begin(),
                                         _state._tvals.end(), r);

            if (iter != first && iter != last)
            {
                std::bernoulli_distribution random(.5);
                if (random(rng))
                    --iter;
                else
                    ++iter;
            }
            else if (iter != first)
            {
                --iter;
            }
            else if (iter != last)
            {
                ++iter;
            }

            return *iter;
        };

        double get_move_prob(size_t, double r, double s, bool, //allow_empty,
                             bool)
        {
            auto iter = std::lower_bound(_state._tvals.begin(),
                                         _state._tvals.end(), r);

            auto first = _state._tvals.begin();
            auto last = _state._tvals.end();
            last--;

            if (iter != first && iter != last)
            {
                if (s == *(iter+1) || s == *(iter-1))
                    return -log(2);
            }
            else if (iter != first)
            {
                if (s == *(iter-1))
                    return 0;
            }
            else if (iter != last)
            {
                if (s == *(iter+1))
                    return 0;
            }

            return -numeric_limits<double>::infinity();
        }

        template <class VS>
        std::tuple<double, double> relabel_group(double r, VS& vs)
        {
            auto [nx, dS, xcache] =
                _state.val_sweep([&](auto nx)
                                 {
                                     return _state.update_nodes_dS(vs, r, nx, _entropy_args);
                                 }, r, _state._tmin_bound, _state._tmax_bound, _beta,
                                 _maxiter, _tol);
            return {nx, dS};
        }
    };

    class gmap_t :
        public gt_hash_map<double, gt_hash_set<size_t>> {};

    template <class T>
    using iset = idx_set<T>;

    template <class T, class V>
    using imap = idx_map<T, V>;

    template <class T>
    using gset = gt_hash_set<T>;

    template <class... Ts>
    class MCMCDynamicsState:
        public MergeSplit<MCMCDynamicsStateImp<Ts...>,
                          size_t,
                          double,
                          iset,
                          imap,
                          gset,
                          gmap_t, false, true>
    {
    public:
        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCDynamicsState(ATs&&... as)
           : MergeSplit<MCMCDynamicsStateImp<Ts...>,
                        size_t,
                        double,
                        iset,
                        imap,
                        gset,
                        gmap_t, false, true>(as...)
        {}
    };
};

} // graph_tool namespace

#endif //DYNAMICS_MULTIFLIP_MCMC_THETA_HH
