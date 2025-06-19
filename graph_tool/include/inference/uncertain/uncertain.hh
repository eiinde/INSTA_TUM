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

#ifndef GRAPH_BLOCKMODEL_UNCERTAIN_HH
#define GRAPH_BLOCKMODEL_UNCERTAIN_HH

#include "config.h"

#include <vector>

#include "../support/graph_state.hh"
#include "uncertain_util.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define UNCERTAIN_STATE_params                                                 \
    ((g, &, all_graph_views, 1))                                               \
    ((q,, eprop_map_t<double>::type, 0))                                       \
    ((q_default,, double, 0))                                                  \
    ((S_const,, double, 0))                                                    \
    ((self_loops,, bool, 0))                                                   \
    ((max_m,, int, 0))

template <class BlockState>
struct Uncertain
{
    GEN_STATE_BASE(UncertainStateBase, UNCERTAIN_STATE_params)

    template <class... Ts>
    class UncertainState
        : public UncertainStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(UncertainStateBase<Ts...>,
                         UNCERTAIN_STATE_params)
        GET_PARAMS_TYPEDEF(Ts, UNCERTAIN_STATE_params)

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
        UncertainState(BlockState& block_state, ATs&&... args)
            : UncertainStateBase<Ts...>(std::forward<ATs>(args)...),
              _block_state(block_state)
        {
            GILRelease gil_release;

            _u_edges.resize(num_vertices(_u));
            for (auto e : edges_range(_u))
            {
                get_u_edge<true>(source(e, _u), target(e, _u)) = e;
                _E += _eweight[e];
            }
            _edges.resize(num_vertices(_g));
            for (auto e : edges_range(_g))
                get_edge<true>(source(e, _g), target(e, _g)) = e;
        }

        typedef BlockState block_state_t;
        BlockState& _block_state;
        typename BlockState::g_t& _u = _block_state._g;
        typename BlockState::eweight_t& _eweight = _block_state._eweight;
        GraphInterface::edge_t _null_edge;
        std::vector<double> _recs;

        std::vector<gt_hash_map<size_t, GraphInterface::edge_t>> _u_edges;
        std::vector<gt_hash_map<size_t, GraphInterface::edge_t>> _edges;

        size_t _E = 0;

        void set_q_default(double q_default)
        {
            _q_default = q_default;
        }

        void set_S_const(double S_const)
        {
            _S_const = S_const;
        }

        template <bool insert, class Graph, class Elist>
        auto& _get_edge(size_t u, size_t v, Graph& g, Elist& edges)
        {
            if (!graph_tool::is_directed(g) && u > v)
                std::swap(u, v);
            auto& qe = edges[u];
            if (insert)
                return qe[v];
            auto iter = qe.find(v);
            if (iter != qe.end())
                return iter->second;
            return _null_edge;
        }

        template <bool insert=false>
        auto& get_u_edge(size_t u, size_t v)
        {
            return _get_edge<insert>(u, v, _u, _u_edges);
        }

        template <bool insert=false>
        auto& get_edge(size_t u, size_t v)
        {
            return _get_edge<insert>(u, v, _g, _edges);
        }

        double entropy(const uentropy_args_t& ea)
        {
            double S = 0;
            if (ea.latent_edges)
            {
                for (auto m : edges_range(_g))
                {
                    double q_e = _q[m];
                    if (q_e == std::numeric_limits<double>::infinity())
                        continue;
                    auto& e = get_u_edge<false>(source(m, _g), target(m, _g));
                    if (e != _null_edge && _eweight[e] > 0 &&
                        (_self_loops || (source(e, _u) != target(e, _u))))
                        S += q_e;
                }

                for (auto e : edges_range(_u))
                {
                    auto& m = get_edge<false>(source(e, _u), target(e, _u));
                    if (m != _null_edge || _eweight[e] == 0 ||
                        (!_self_loops && source(m, _g) == target(m, _g)))
                        continue;
                    if (_q_default == std::numeric_limits<double>::infinity())
                        continue;
                    S += _q_default;
                }
                S += _S_const;
            }

            if (ea.density)
                S += _E * log(ea.aE) - lgamma_fast(_E + 1) + ea.aE;

            return -S;
        }

        double remove_edge_dS(size_t u, size_t v, int dm, const uentropy_args_t& ea)
        {
            auto& e = get_u_edge(u, v);
            double dS = _block_state.modify_edge_dS(source(e, _u),
                                                    target(e, _u),
                                                    e, -dm, ea);
            if (ea.density)
            {
                dS += log(ea.aE) * dm;
                dS += lgamma_fast(_E + 1 - dm) - lgamma_fast(_E + 1);
            }

            if (ea.latent_edges)
            {
                if (_eweight[e] == dm && (_self_loops || u != v))
                {
                    auto& m = get_edge<false>(u, v);
                    double q_e = (m == _null_edge) ? _q_default : _q[m];
                    dS += q_e;
                }
            }
            return dS;
        }

        double add_edge_dS(size_t u, size_t v, int dm, const uentropy_args_t& ea)
        {
            auto& e = get_u_edge(u, v);

            auto m = (e == _null_edge) ? 0 : _eweight[e];

            if (m + dm > _max_m)
                return numeric_limits<double>::infinity();

            double dS = _block_state.modify_edge_dS(u, v, e, dm, ea);
            if (ea.density)
            {
                dS -= log(ea.aE) * dm;
                dS += lgamma_fast(_E + 1 + dm) - lgamma_fast(_E + 1);
            }

            if (ea.latent_edges)
            {
                if ((e == _null_edge || _eweight[e] == 0) && (_self_loops || u != v))
                {
                    auto& m = get_edge<false>(u, v);
                    double q_e = (m == _null_edge) ? _q_default : _q[m];
                    dS -= q_e;
                }
            }
            return dS;
        }

        void remove_edge(size_t u, size_t v, int dm)
        {
            auto& e = get_u_edge(u, v);
            _block_state.template modify_edge<false>(u, v, e, dm);
            _E -= dm;
        }

        void add_edge(size_t u, size_t v, int dm)
        {
            auto& e = get_u_edge<true>(u, v);
            _block_state.template modify_edge<true>(u, v, e, dm);
            _E += dm;
        }

    };
};

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_UNCERTAIN_HH
