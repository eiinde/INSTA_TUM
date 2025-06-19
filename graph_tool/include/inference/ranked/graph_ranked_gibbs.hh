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

#ifndef GRAPH_RANKED_GIBBS_HH
#define GRAPH_RANKED_GIBBS_HH

#include "config.h"

#include <vector>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include <boost/mpl/vector.hpp>

namespace graph_tool
{
using namespace boost;
using namespace std;

#define GIBBS_BLOCK_STATE_params(State)                                        \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((vlist,&, std::vector<size_t>&, 0))                                       \
    ((beta,, double, 0))                                                       \
    ((oentropy_args,, python::object, 0))                                      \
    ((allow_new_group,, bool, 0))                                              \
    ((sequential,, bool, 0))                                                   \
    ((deterministic,, bool, 0))                                                \
    ((verbose,, bool, 0))                                                      \
    ((niter,, size_t, 0))


template <class State>
struct Gibbs
{
    GEN_STATE_BASE(GibbsBlockStateBase, GIBBS_BLOCK_STATE_params(State))

    template <class... Ts>
    class GibbsBlockState
        : public GibbsBlockStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(GibbsBlockStateBase<Ts...>,
                         GIBBS_BLOCK_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, GIBBS_BLOCK_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        GibbsBlockState(ATs&&... as)
           : GibbsBlockStateBase<Ts...>(as...),
            _entropy_args(python::extract<typename State::_entropy_args_t&>(_oentropy_args))
        {
            _state.init_mcmc(*this);
        }

        typename State::_entropy_args_t& _entropy_args;

        double _c = numeric_limits<double>::infinity();

        std::vector<size_t> _candidate_groups;

        auto& get_moves(size_t)
        {
            _candidate_groups.clear();
            _candidate_groups.insert(_candidate_groups.end(),
                                     _state._ustate._candidate_groups.begin(),
                                     _state._ustate._candidate_groups.end());
            _candidate_groups.push_back(null_group);
            return _candidate_groups;
        }

        size_t node_state(size_t v)
        {
            return _state._b[v];
        }

        size_t node_weight(size_t v)
        {
            return _state._ustate.node_weight(v);
        }

        size_t _nr;
        double virtual_move_dS(size_t v, size_t nr, rng_t& rng)
        {
            size_t r = _state._b[v];
            if (!_allow_new_group && nr != r && _state.virtual_remove_size(v) == 0)
                return numeric_limits<double>::infinity();
            if (nr == null_group)
            {
                if (!_allow_new_group ||
                    _state._ustate._candidate_groups.size() == num_vertices(_state._ustate._g) ||
                    _state.virtual_remove_size(v) == 0)
                    return numeric_limits<double>::infinity();
                _state.get_empty_block(v);
                _nr = nr = uniform_sample(_state._ustate._empty_groups, rng);

                uniform_real_distribution<double> usample;
                _state._u[nr] = usample(rng);

                auto cstate = _state._ustate._coupled_state;
                if (cstate != nullptr)
                {
                    cstate->sample_branch(nr, r, rng);
                    //_state._dstate.copy_branch(nr, _state._ustate);
                }
                _state._ustate._bclabel[nr] = _state._ustate._bclabel[r];
            }
            return _state.virtual_move(v, r, nr, _entropy_args);
        }

        void perform_move(size_t v, size_t nr)
        {
            if (nr == null_group)
                nr = _nr;
            _state.move_vertex(v, nr);
        }
    };
};


} // graph_tool namespace

#endif //GRAPH_RANKED_GIBBS_HH
