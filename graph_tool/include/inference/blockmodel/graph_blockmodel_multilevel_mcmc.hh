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

#ifndef GRAPH_BLOCKMODEL_MULTILEVEL_MCMC_HH
#define GRAPH_BLOCKMODEL_MULTILEVEL_MCMC_HH

#include "config.h"

#include <vector>
#include <algorithm>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include "graph_blockmodel_util.hh"
#include <boost/mpl/vector.hpp>

#include "idx_map.hh"
#include "../loops/multilevel.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef vprop_map_t<int32_t>::type vmap_t;

#define MCMC_BLOCK_STATE_params(State)                                         \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((c,, double, 0))                                                          \
    ((d,, double, 0))                                                          \
    ((r,, double, 0))                                                          \
    ((random_bisect,, bool, 0))                                                \
    ((merge_sweeps,, size_t, 0))                                               \
    ((mh_sweeps,, size_t, 0))                                                  \
    ((parallel,, bool, 0))                                                     \
    ((init_min_iter,, size_t, 0))                                              \
    ((init_r,, double, 0))                                                     \
    ((init_beta,, double, 0))                                                  \
    ((gibbs,, bool, 0))                                                        \
    ((M,, size_t, 0))                                                          \
    ((global_moves,, bool, 0))                                                 \
    ((cache_states,, bool, 0))                                                 \
    ((B_min,, size_t, 0))                                                      \
    ((B_max,, size_t, 0))                                                      \
    ((b_min,, vmap_t, 0))                                                      \
    ((b_max,, vmap_t, 0))                                                      \
    ((force_accept,, bool, 0))                                                 \
    ((oentropy_args,, python::object, 0))                                      \
    ((verbose,, int, 0))                                                       \
    ((niter,, size_t, 0))

template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCBlockStateBase, MCMC_BLOCK_STATE_params(State))

    template <class... Ts>
    class MCMCBlockStateImp
        : public MCMCBlockStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MCMCBlockStateBase<Ts...>,
                         MCMC_BLOCK_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_BLOCK_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCBlockStateImp(ATs&&... as)
           : MCMCBlockStateBase<Ts...>(as...),
             _entropy_args(python::extract<typename State::_entropy_args_t&>(_oentropy_args))
        {
            GILRelease gil_release;

            _state.init_mcmc(*this);

            for (int i = 0; i < get_num_threads(); ++i)
                _m_entries.emplace_back(num_vertices(_state._bg));
            _pstates.resize(get_num_threads(), nullptr);

            if (_global_moves)
            {
                idx_set<size_t> rs_min, rs_max;
                for (auto v : vertices_range(_state._g))
                {
                    rs_min.insert(_b_min[v]);
                    rs_max.insert(_b_max[v]);
                }
                _has_b_min = rs_min.size() == _B_min;
                _has_b_max = rs_max.size() == _B_max;
            }

            auto cstate = _state._coupled_state;
            if (cstate != nullptr)
            {
                _bh = cstate->get_b();
                _hpclabel = cstate->get_pclabel();
            }
        }

        ~MCMCBlockStateImp()
        {
            #pragma omp parallel for schedule(runtime)
            for (size_t i = 1; i < _pstates.size(); ++i)
            {
                if (_pstates[i] != nullptr)
                    delete _pstates[i];
            }
        }

        bool _has_b_max = false;
        bool _has_b_min = false;

        std::vector<typename State::m_entries_t> _m_entries;
        std::vector<State*> _pstates;

        typename State::_entropy_args_t& _entropy_args;

        vprop_map_t<int32_t>::type::unchecked_t _bh;
        vprop_map_t<int32_t>::type::unchecked_t _hpclabel;

        idx_set<size_t> _rs;

        constexpr static size_t _null_group = null_group;

        template <class F>
        void iter_nodes(F&& f)
        {
            for (auto v : vertices_range(_state._g))
            {
                if (_state.node_weight(v) == 0)
                    continue;
                f(v);
            }
        }

        template <class F>
        void iter_groups(F&& f)
        {
            for (auto r : vertices_range(_state._bg))
            {
                if (_state._wr[r] == 0)
                    continue;
                f(r);
            }
        }

        size_t get_group(size_t v)
        {
            auto& state = (_pstates[0] == nullptr) ?
                _state : *_pstates[get_thread_num()];
            return state._b[v];
        }

        void split_parallel()
        {
            _pstates[0] = &_state;
            #pragma omp parallel for schedule(runtime)
            for (size_t i = 1; i < _pstates.size(); ++i)
            {
                if (_pstates[i] == nullptr)
                    _pstates[i] = _state.deep_copy();
                else
                    _pstates[i]->deep_assign(_state);
            }
        }

        void unsplit_parallel()
        {
            _pstates[0] = nullptr;
        }

        template <class RNG>
        size_t get_new_group(size_t v, bool inherit, RNG& rng)
        {
            size_t t = 0;
            if (_state._empty_groups.empty())
                t =_state.get_empty_block(v);
            else
                t = uniform_sample(_state._empty_groups, rng);

            if (inherit)
            {
                auto r = _state._b[v];
                _state._bclabel[t] = _state._bclabel[r];
                if (_state._coupled_state != nullptr)
                {
                    _bh[t] = _bh[r];
                    _hpclabel[t] = _state._pclabel[v];
                }
            }

            return t;
        }

        void move_node(size_t v, size_t r, bool cache)
        {
            auto& state = (_pstates[0] == nullptr) ?
                _state : *_pstates[get_thread_num()];

            if (cache)
                state.move_vertex(v, r, _m_entries[get_thread_num()]);
            else
                state.move_vertex(v, r);
        }

        double virtual_move(size_t v, size_t r, size_t s)
        {
            auto& state = (_pstates[0] == nullptr) ?
                _state : *_pstates[get_thread_num()];

            if (std::isinf(_beta) && state._coupled_state != nullptr)
            {
                if (_bh[r] != _bh[s])
                    return numeric_limits<double>::infinity();
            }
            return state.virtual_move(v, r, s, _entropy_args,
                                      _m_entries[get_thread_num()]);
        }

        template <class VS>
        size_t get_Bmin(VS& vs)
        {
            if (std::isinf(_beta) && _state._coupled_state != nullptr)
            {
                _rs.clear();
                for (auto& v : vs)
                    _rs.insert(_bh[get_group(v)]);
                return _rs.size();
            }
            return 1;
        }

        size_t get_b_min(size_t v)
        {
            return _b_min[v];
        }

        size_t get_b_max(size_t v)
        {
            return _b_max[v];
        }

        template <class RNG>
        size_t sample_group(size_t v, bool allow_random, bool allow_empty,
                            bool init_heuristic, RNG& rng)
        {
            auto& state = (_pstates[0] == nullptr) ?
                _state : *_pstates[get_thread_num()];
            if (!init_heuristic)
                return state.sample_block(v, allow_random ? _c : 0,
                                          allow_empty ? _d : 0, rng);
            else
                return state.sample_block_local(v, rng);
        }

        double get_move_prob(size_t v, size_t r, size_t s, bool allow_random,
                             bool allow_empty, bool reverse)
        {
            return _state.get_move_prob(v, r, s,
                                        allow_random ? _c : 0,
                                        allow_empty ? _d : 0,
                                        reverse);
        }

        double entropy()
        {
            return _state.entropy(_entropy_args, true);
        }

        template <class RNG>
        double sample_group_label(size_t, RNG&)
        {
            return 0;
        }

        double relabel_group_dS(size_t, double)
        {
            return 0;
        }

        void relabel_group(size_t, double)
        {
        }

        void relax_update(bool relax)
        {
            auto& state = (_pstates[0] == nullptr) ?
                _state : *_pstates[get_thread_num()];
            state.relax_update(relax);
        }

        template <class V>
        void push_state(V&& vs)
        {
            _state.push_state(vs);
        }

        void pop_state()
        {
            _state.pop_state();
        }

    };

    class gmap_t :
        public idx_map<size_t, idx_set<size_t, true>>
    {
    public:

        idx_set<size_t, true>& operator[](const size_t& key)
        {
            auto iter = find(key);
            if (iter == end())
                iter = insert(std::make_pair(key, idx_set<size_t, true>(_pos))).first;
            return iter->second;
        }

    private:
        std::vector<size_t> _pos;
    };

    template <class T>
    using iset = idx_set<T>;

    template <class T, class V>
    using imap = idx_map<T, V>;

    template <class... Ts>
    class MCMCBlockState:
        public Multilevel<MCMCBlockStateImp<Ts...>,
                          size_t,
                          size_t,
                          iset,
                          imap,
                          iset,
                          imap,
                          gmap_t, false>
    {
    public:
        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCBlockState(ATs&&... as)
           : Multilevel<MCMCBlockStateImp<Ts...>,
                        size_t,
                        size_t,
                        iset,
                        imap,
                        iset,
                        imap,
                        gmap_t, false>(as...)
        {}
    };
};

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_MULTILEVEL_MCMC_HH
