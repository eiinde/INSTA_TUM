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

#ifndef GRAPH_RANKED_MULTIFLIP_MCMC_HH
#define GRAPH_RANKED_MULTIFLIP_MCMC_HH

#include "config.h"

#include <vector>
#include <algorithm>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include "../blockmodel/graph_blockmodel_util.hh"
#include <boost/mpl/vector.hpp>

#include "idx_map.hh"
#include "../loops/merge_split.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_BLOCK_STATE_params(State)                                         \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((c,, double, 0))                                                          \
    ((d,, double, 0))                                                          \
    ((psingle,, double, 0))                                                    \
    ((psplit,, double, 0))                                                     \
    ((pmerge,, double, 0))                                                     \
    ((pmergesplit,, double, 0))                                                \
    ((pmovelabel,, double, 0))                                                 \
    ((nproposal, &, vector<size_t>&, 0))                                       \
    ((nacceptance, &, vector<size_t>&, 0))                                     \
    ((gibbs_sweeps,, size_t, 0))                                               \
    ((oentropy_args,, python::object, 0))                                      \
    ((verbose,, int, 0))                                                       \
    ((force_move,, bool, 0))                                                   \
    ((niter,, double, 0))

template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCBlockStateBase, MCMC_BLOCK_STATE_params(State))

    template <class... Ts>
    class MCMCBlockStateImp
        : public MCMCBlockStateBase<Ts...>,
          public MergeSplitStateBase
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
            _m_entries(num_vertices(_state._ustate._bg)),
            _entropy_args(python::extract<typename State::_entropy_args_t&>(_oentropy_args))
        {
            GILRelease gil_release;
            _state.init_mcmc(*this);
        }

        typename State::m_entries_t _m_entries;
        typename State::_entropy_args_t& _entropy_args;

        constexpr static size_t _null_group = null_group;

        constexpr static double _psrandom = 1;
        constexpr static double _psscatter = 1;
        constexpr static double _pscoalesce = 1;

        template <class F>
        void iter_nodes(F&& f)
        {
            for (auto v : vertices_range(_state._ustate._g))
            {
                if (_state._ustate.node_weight(v) == 0)
                    continue;
                f(v);
            }
        }

        template <class F>
        void iter_groups(F&& f)
        {
            for (auto r : vertices_range(_state._ustate._bg))
            {
                if (_state._ustate._wr[r] == 0)
                    continue;
                f(r);
            }
        }

        size_t get_group(size_t v)
        {
            return _state._ustate._b[v];
        }

        template <bool sample_branch=true, class RNG, class VS = std::array<size_t,0>>
        size_t sample_new_group(size_t v, RNG& rng, VS&& except = VS())
        {
            _state.get_empty_block(v, except.size() >= _state._ustate._empty_groups.size());
            size_t t;
            do
            {
                t = uniform_sample(_state._ustate._empty_groups, rng);
            } while (!except.empty() &&
                     std::find(except.begin(), except.end(), t) != except.end());

            _state._u_c[t] = _state.sample_u(rng);

            auto r = _state._ustate._b[v];
            _state._ustate._bclabel[t] = _state._ustate._bclabel[r];

            return t;
        }

        void move_node(size_t v, size_t r, bool cache)
        {
            if (cache)
                _state.move_vertex(v, r, _m_entries);
            else
                _state.move_vertex(v, r);
        }

        void reserve_empty_groups(size_t nB)
        {
            if (_state._ustate._empty_groups.size() < nB)
                _state.add_block(nB - _state._ustate._empty_groups.size());
        }

        bool allow_move(size_t r, size_t s)
        {
            return _state.allow_move(r, s);
        }

        double virtual_move(size_t v, size_t r, size_t s)
        {
            return _state.virtual_move(v, r, s, _entropy_args, _m_entries);
        }

        template <class RNG>
        size_t sample_group(size_t v, bool allow_empty, RNG& rng)
        {
            return _state.sample_block(v, _c, allow_empty ? _d : 0, rng);
        }

        double get_move_prob(size_t v, size_t r, size_t s, bool allow_empty,
                             bool reverse)
        {
            return _state.get_move_prob(v, r, s, _c, allow_empty ? _d : 0, reverse);
        }

        bool can_swap(size_t r, size_t s)
        {
            auto cstate = _state._ustate._coupled_state;
            if (cstate == nullptr)
                return _state._ustate._bclabel[r] == _state._ustate._bclabel[s];
            auto& hb = cstate->get_b();
            auto rr = hb[r];
            auto ss = hb[s];
            return (rr == ss) && (_state._ustate._bclabel[r] == _state._ustate._bclabel[s]);
        }

        void relax_update(bool relax)
        {
            _state.relax_update(relax);
        }

        void store_next_state(size_t v)
        {
            _state.store_next_state(v);
        }

        void clear_next_state()
        {
            _state._egroups.check(_state._bg, _state._mrs);
            _state.clear_next_state();
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
        public MergeSplit<MCMCBlockStateImp<Ts...>,
                          size_t,
                          size_t,
                          iset,
                          imap,
                          iset,
                          gmap_t, false, true>
    {
    public:
        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCBlockState(ATs&&... as)
           : MergeSplit<MCMCBlockStateImp<Ts...>,
                        size_t,
                        size_t,
                        iset,
                        imap,
                        iset,
                        gmap_t, false, true>(as...)
        {}
    };
};

} // graph_tool namespace

#endif //GRAPH_RANKED_MULTIFLIP_MCMC_HH
