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

#ifndef GRAPH_BLOCKMODEL_MEASURED_HH
#define GRAPH_BLOCKMODEL_MEASURED_HH

#include "config.h"

#include <vector>

#include "../support/graph_state.hh"
#include "../blockmodel/graph_blockmodel_util.hh"
#include "uncertain.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MEASURED_STATE_params                                                  \
    ((g, &, all_graph_views, 1))                                               \
    ((n,, eprop_map_t<int>::type, 0))                                          \
    ((x,, eprop_map_t<int>::type, 0))                                          \
    ((n_default,, int, 0))                                                     \
    ((x_default,, int, 0))                                                     \
    ((alpha,, double, 0))                                                      \
    ((beta,, double, 0))                                                       \
    ((mu,, double, 0))                                                         \
    ((nu,, double, 0))                                                         \
    ((lp,, double, 0))                                                         \
    ((lq,, double, 0))                                                         \
    ((max_m,, int, 0))                                                         \
    ((self_loops,, bool, 0))

template <class BlockState>
struct Measured
{
    GEN_STATE_BASE(MeasuredStateBase, MEASURED_STATE_params)

    template <class... Ts>
    class MeasuredState
        : public MeasuredStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MeasuredStateBase<Ts...>,
                         MEASURED_STATE_params)
        GET_PARAMS_TYPEDEF(Ts, MEASURED_STATE_params)

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
        MeasuredState(BlockState& block_state, ATs&&... args)
            : MeasuredStateBase<Ts...>(std::forward<ATs>(args)...),
              _block_state(block_state)
        {
            GILRelease gil_release;

            _u_edges.resize(num_vertices(_u));
            for (auto e : edges_range(_u))
            {
                get_u_edge<true>(source(e, _u), target(e, _u)) = e;
                _E += _eweight[e];
            }

            size_t gE = 0;
            _edges.resize(num_vertices(_g));
            for (auto e : edges_range(_g))
            {
                get_edge<true>(source(e, _g), target(e, _g)) = e;
                _N += _n[e];
                _X += _x[e];
                gE++;
            }

            for (auto e : edges_range(_u))
            {
                if (_eweight[e] == 0 || (!_self_loops &&
                                         (source(e, _u) == target(e, _u))))
                    continue;
                auto ge = get_edge<false>(source(e, _u), target(e, _u));
                _T += (ge != _null_edge) ? _x[ge] : _x_default;
                _M += (ge != _null_edge) ? _n[ge] : _n_default;
            }

            uint64_t N = num_vertices(_g);
            if (_self_loops)
                _NP = graph_tool::is_directed(_g) ? N * N : (N * (N + 1)) / 2;
            else
                _NP = graph_tool::is_directed(_g) ? N * (N - 1) : (N * (N - 1)) / 2;

            _N += (_NP - gE) * _n_default;
            _X += (_NP - gE) * _x_default;

            if (!std::isnan(_lp))
                _logit1mp = log1p(-exp(_lp)) - _lp;
            if (!std::isnan(_lq))
            {
                _l1mq = log1p(-exp(_lq));
                _logitq = _lq - _l1mq;
            }
        }

        typedef BlockState block_state_t;
        BlockState& _block_state;
        typename BlockState::g_t& _u = _block_state._g;
        typename BlockState::eweight_t& _eweight = _block_state._eweight;
        GraphInterface::edge_t _null_edge;

        std::vector<gt_hash_map<size_t, GraphInterface::edge_t>> _u_edges;
        std::vector<gt_hash_map<size_t, GraphInterface::edge_t>> _edges;

        uint64_t _NP = 0;
        uint64_t _E = 0;

        uint64_t _N = 0;
        uint64_t _X = 0;
        uint64_t _T = 0;
        uint64_t _M = 0;

        double _logit1mp = std::numeric_limits<double>::quiet_NaN();
        double _l1mq = std::numeric_limits<double>::quiet_NaN();
        double _logitq = std::numeric_limits<double>::quiet_NaN();

        bool _E_prior;

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

        double get_MP(size_t T, size_t M, bool complete = true)
        {
            double S = 0;
            if (std::isnan(_lp))
            {
                S += lbeta((M - T) + _alpha, T + _beta);
                if (complete)
                    S -= lbeta(_alpha, _beta);
            }
            else
            {
                if (_lp == 0)
                    S -= (T == 0) ? 0 : std::numeric_limits<double>::infinity();
                else if (std::isinf(_lp))
                    S -= (M == T) ? 0 : std::numeric_limits<double>::infinity();
                else
                    S += _logit1mp * T + _lp * M;
            }

            if (std::isnan(_lq))
            {
                S += lbeta((_X - T) + _mu, ((_N - _X) - (M - T)) + _nu);
                if (complete)
                    S -= lbeta(_mu, _nu);
            }
            else
            {
                if (std::isinf(_lq))
                    S -= (_X == T) ? 0 : std::numeric_limits<double>::infinity();
                else if (_lq == 0)
                    S -= (_X - T == _N - M) ? 0 : std::numeric_limits<double>::infinity();
                else
                    S += _logitq * (_X - T) + _l1mq * (_N - M);
            }

            return S;
        }

        double entropy(const uentropy_args_t& ea)
        {
            double S = 0;
            size_t gE = 0;
            if (ea.latent_edges)
            {
                for (auto e : edges_range(_g))
                {
                    S += lbinom(_n[e], _x[e]);
                    gE++;
                }
                S += (_NP - gE) * lbinom(_n_default, _x_default);

                S += get_MP(_T, _M);
            }

            if (ea.density)
                S += -(_E * log(ea.aE)) + lgamma_fast(_E + 1) - ea.aE;

            return -S;
        }

        double get_dS(int dT, int dM)
        {
            // FIXME: Can be faster!
            double Si = get_MP(_T, _M, false);
            double Sf = get_MP(_T + dT, _M + dM, false);
            return -(Sf - Si);
        }

        double remove_edge_dS(size_t u, size_t v, int dm, const uentropy_args_t& ea)
        {
            auto& e = get_u_edge(u, v);

            double dS = _block_state.modify_edge_dS(u, v, e, -dm, ea);

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
                    int dT = (m == _null_edge) ? _x_default : _x[m];
                    int dM = (m == _null_edge) ? _n_default : _n[m];
                    dS += get_dS(-dT, -dM);
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
                    int dT = (m == _null_edge) ? _x_default : _x[m];
                    int dM = (m == _null_edge) ? _n_default : _n[m];
                    dS += get_dS(dT, dM);
                }
            }
            return dS;
        }

        void remove_edge(size_t u, size_t v, int dm)
        {
            auto& e = get_u_edge(u, v);

            if (_eweight[e] == dm && (_self_loops || u != v))
            {
                auto& m = get_edge<false>(u, v);
                int dT = (m == _null_edge) ? _x_default : _x[m];
                int dM = (m == _null_edge) ? _n_default : _n[m];
                _T -= dT;
                _M -= dM;
            }

            _block_state.template modify_edge<false>(u, v, e, dm);
            _E -= dm;
        }

        void add_edge(size_t u, size_t v, int dm)
        {
            auto& e = get_u_edge<true>(u, v);

            if ((e == _null_edge || _eweight[e] == 0) && (_self_loops || u != v))
            {
                auto& m = get_edge<false>(u, v);
                int dT = (m == _null_edge) ? _x_default : _x[m];
                int dM = (m == _null_edge) ? _n_default : _n[m];
                _T += dT;
                _M += dM;
            }

            _block_state.template modify_edge<true>(u, v, e, dm);
            _E += dm;
        }

        void set_hparams(double alpha, double beta, double mu, double nu)
        {
            _alpha = alpha;
            _beta = beta;
            _mu = mu;
            _nu = nu;
        }

        uint64_t get_N() { return _N; }
        uint64_t get_X() { return _X; }
        uint64_t get_T() { return _T; }
        uint64_t get_M() { return _M; }
    };
};

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_MEASURED_HH
