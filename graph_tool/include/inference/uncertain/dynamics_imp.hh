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

#include "graph_tool.hh"
#include "random.hh"

#include <boost/python.hpp>

#define GRAPH_VIEWS never_filtered_never_reversed
#include "../../blockmodel/graph_blockmodel.hh"

#include "dynamics.hh"
#include "dynamics_base.hh"
#include "../../support/graph_state.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, BlockState, BLOCK_STATE_params)

template <class BaseState>
struct Dyn : Dynamics<BaseState> {};

template <class BaseState>
GEN_DISPATCH(dynamics_state, Dyn<BaseState>::template DynamicsState,
             DYNAMICS_STATE_params)

#define MAKE_STATE_(X) mcmc_##X##_sweep
#define MAKE_STATE(X) MAKE_STATE_(X)

python::object MAKE_STATE(BASE_NAME)(python::object odynamics_state,
                                     python::object ot, python::object os,
                                     python::dict params)
{
    python::object state;
    block_state::dispatch
        ([&](auto* bs)
         {
             typedef typename std::remove_reference<decltype(*bs)>::type block_state_t;
             dynamics_state<block_state_t>::dispatch
                 (odynamics_state,
                  [&](auto& s)
                  {
                      state = python::object(std::make_shared<DState>(params, s,
                                                                      ot, os));
                  }, false);
         });
    return state;
}

#define STRINGIFY_(X) #X
#define STRINGIFY(X) STRINGIFY_(X)

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;
    def("make_" STRINGIFY(BASE_NAME) "_state", &MAKE_STATE(BASE_NAME));

    auto name = name_demangle(typeid(DState).name());

    class_<DState, bases<DStateBase>, std::shared_ptr<DState>,
           boost::noncopyable>(name.c_str(), no_init);

}, 6);
