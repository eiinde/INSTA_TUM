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

#ifndef GRAPH_BLOCKMODEL_LAYERS_UTIL_HH
#define GRAPH_BLOCKMODEL_LAYERS_UTIL_HH

#include "config.h"

namespace graph_tool
{
using namespace boost;
using namespace std;

template <class State, class MEntries>
double virtual_move_covariate(size_t v, size_t r, size_t s, State& state,
                              MEntries& m_entries, bool reset)
{
    if (reset)
        state.get_move_entries(v, r, s, m_entries);

    double dS = 0;
    entries_op(m_entries, state._emat,
               [&](auto, auto, auto& me, auto d)
               {
                   int ers = (me != state._emat.get_null_edge()) ?
                       state._mrs[me] : 0;
                   assert(ers + d >= 0);
                   dS -= -lgamma_fast(ers + 1);
                   dS += -lgamma_fast(ers + d + 1);
               });
    return dS;
}

template <class Graph, class EMap>
double covariate_entropy(Graph& bg, EMap& mrs)
{
    double S = 0;
    for (auto e : edges_range(bg))
        S -= lgamma_fast(mrs[e] + 1);
    return S;
}

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_LAYERS_UTIL_HH
