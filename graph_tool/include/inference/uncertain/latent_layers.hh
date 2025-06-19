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

#ifndef GRAPH_BLOCKMODEL_LATENT_LAYERS_HH
#define GRAPH_BLOCKMODEL_LATENT_LAYERS_HH

#include "config.h"

#include <vector>

#include "../support/graph_state.hh"
#include "uncertain_util.hh"
#include "measured.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef eprop_map_t<int32_t>::type emap_t;

#define LATENT_LAYERS_STATE_params                                             \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((og,, python::object, 0))                                                 \
    ((eweight,, emap_t, 0))                                                    \
    ((self_loops,, bool, 0))                                                   \
    ((measured,, bool, 0))                                                     \
    ((ag_orig, &, boost::any, 0))                                              \
    ((n,, eprop_map_t<int>::type, 0))                                          \
    ((x,, eprop_map_t<int>::type, 0))                                          \
    ((n_default,, int, 0))                                                     \
    ((x_default,, int, 0))                                                     \
    ((alpha,, long double, 0))                                                 \
    ((beta,, long double, 0))                                                  \
    ((mu,, long double, 0))                                                    \
    ((nu,, long double, 0))                                                    \
    ((lp,, long double, 0))                                                    \
    ((lq,, long double, 0))                                                    \
    ((max_w,, int, 0))


template <class Graph, class EW>
struct DummyBlockState
{
    DummyBlockState(Graph& g, EW& eweight)
        :_g(g), _eweight(eweight){}

    template <class E, class R, class EA>
    constexpr double modify_edge_dS(size_t, size_t, E&&, R&&, EA&&)
    {
        return 0.;
    }

    template <bool Add, class E, class R>
    constexpr void modify_edge(size_t, size_t, E&, R&&)
    {
    }

    typedef Graph g_t;
    typedef EW eweight_t;

    Graph& _g;
    EW _eweight;
};


template <class BlockState>
struct LatentLayers
{
    GEN_STATE_BASE(LatentLayersStateBase, LATENT_LAYERS_STATE_params)

    template <class... Ts>
    class LatentLayersState
        : public LatentLayersStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(LatentLayersStateBase<Ts...>,
                         LATENT_LAYERS_STATE_params)
        GET_PARAMS_TYPEDEF(Ts, LATENT_LAYERS_STATE_params)

        typedef BlockState block_state_t;
        typedef typename BlockState::g_t g_t;

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
        LatentLayersState(std::vector<std::reference_wrapper<BlockState>> block_state, ATs&&... args)
            : LatentLayersStateBase<Ts...>(std::forward<ATs>(args)...),
              _g(boost::any_cast<std::reference_wrapper<g_t>>(python::extract<boost::any&>(_og)()).get()),
              _block_state(block_state),
              _eweight_c(_eweight.get_checked()),
              _g_orig(boost::any_cast<std::reference_wrapper<g_t>>(_ag_orig).get())
        {
            GILRelease gil_release;

            _edges.resize(num_vertices(_g));
            for (auto e : edges_range(_g))
                get_edge<true>(source(e, _g), target(e, _g)) = e;

            _u_edges.resize(_block_state.size());
            _Es.resize(_block_state.size());
            for (size_t l = 0; l < _block_state.size(); ++l)
            {
                auto& bstate = _block_state[l].get();
                auto& u = bstate._g;
                _u_edges[l].resize(num_vertices(u));
                for (auto e : edges_range(u))
                {
                    get_u_edge<true>(l, source(e, u), target(e, u)) = e;
                    auto x = bstate._eweight[e];
                    _eweight[get_edge<>(source(e, u), target(e, u))] += x;
                    _E += x;
                    _Es[l] += x;
                }
            }

            _uea.latent_edges = true;
            _uea.density = false;

            if (_measured)
            {
                gil_release.restore();
                _dummy = std::make_shared<DummyBlockState<g_t, eweight_t>>(_g, _eweight);
                _mstate.emplace_back(*_dummy, _g_orig, _n.get_checked(),
                                     _x.get_checked(), _n_default, _x_default,
                                     _alpha, _beta, _mu, _nu, _lp, _lq,
                                     _max_w, _self_loops);
            }
        }

        g_t& _g;
        std::vector<std::reference_wrapper<BlockState>> _block_state;
        GraphInterface::edge_t _null_edge;

        std::vector<std::vector<gt_hash_map<size_t, GraphInterface::edge_t>>> _u_edges;
        std::vector<gt_hash_map<size_t, GraphInterface::edge_t>> _edges;

        typename eweight_t::checked_t _eweight_c;

        size_t _E = 0;
        std::vector<size_t> _Es;
        bool _E_prior;

        g_t& _g_orig;
        std::shared_ptr<DummyBlockState<g_t, eweight_t>> _dummy;
        std::vector<typename Measured<DummyBlockState<g_t, eweight_t>>::template MeasuredState<g_t, GET_COMMA_LIST(GET_PARAMS_TYPES(MEASURED_STATE_params))>> _mstate;
        uentropy_args_t _uea = entropy_args_t();

        template <bool insert, class Graph, class Elist>
        auto& _get_edge(size_t u, size_t v, Graph& g, Elist& edges)
        {
            if (!graph_tool::is_directed(g) && u > v)
                std::swap(u, v);
            auto& qe = edges[u];
            if constexpr (insert)
                return qe[v];
            auto iter = qe.find(v);
            if (iter != qe.end())
                return iter->second;
            return _null_edge;
        }

        template <bool insert=false>
        auto& get_u_edge(size_t l, size_t u, size_t v)
        {
            auto& bstate = _block_state[l].get();
            return _get_edge<insert>(u, v, bstate._g, _u_edges[l]);
        }

        template <bool insert=false>
        auto& get_edge(size_t u, size_t v)
        {
            return _get_edge<insert>(u, v, _g, _edges);
        }

        double entropy(uentropy_args_t ea)
        {
            double L = 0;

            if (ea.density)
                L -= -(_Es[0] * log(ea.aE)) + lgamma_fast(_Es[0] + 1) - ea.aE;

            if (_measured)
            {
                ea.density = false;
                L -= _mstate[0].entropy(ea);
            }

            return -L;
        }

        double remove_edge_dS(size_t l, size_t u, size_t v, const uentropy_args_t& ea)
        {
            auto& bstate = _block_state[l].get();
            auto& e = get_u_edge(l, u, v);
            auto& g = bstate._g;
            double dS = bstate.template modify_edge_dS<false>(source(e, g),
                                                              target(e, g),
                                                              e, ea);
            if (ea.density && l == 0)
            {
                dS += log(ea.aE);
                dS += lgamma_fast(_Es[0]) - lgamma_fast(_Es[0] + 1);
            }

            if (_measured && !std::isinf(dS))
                dS += _mstate[0].remove_edge_dS(u, v, 1, _uea);

            return dS;
        }

        double add_edge_dS(size_t l, size_t u, size_t v, const uentropy_args_t& ea)
        {
            auto& bstate = _block_state[l].get();
            auto& e = get_u_edge(l, u, v);
            double dS = bstate.template modify_edge_dS<true>(u, v, e, ea);
            if (ea.density && l == 0)
            {
                dS -= log(ea.aE);
                dS += lgamma_fast(_Es[0] + 2) - lgamma_fast(_Es[0] + 1);
            }

            if (_measured && !std::isinf(dS))
                dS += _mstate[0].add_edge_dS(u, v, 1, _uea);

            return dS;
        }

        void remove_edge(size_t l, size_t u, size_t v)
        {
            if (_measured)
                _mstate[0].remove_edge(u, v, 1);

            auto& bstate = _block_state[l].get();
            auto& e = get_u_edge(l, u, v);
            bstate.template modify_edge<false>(u, v, e);

            auto& ge = get_edge(u, v);
            _eweight[ge]--;
            if (_eweight[ge] == 0)
            {
                boost::remove_edge(ge, _g);
                ge = _null_edge;
                if (_measured)
                    _mstate[0].get_u_edge(u, v) = _null_edge;
            }
            _E--;
            _Es[l]--;
        }

        void add_edge(size_t l, size_t u, size_t v)
        {
            if (_measured)
                _mstate[0].add_edge(u, v, 1);

            auto& bstate = _block_state[l].get();
            auto& e = get_u_edge<true>(l, u, v);
            bstate.template modify_edge<true>(u, v, e);

            auto& ge = get_edge<true>(u, v);
            if (ge == _null_edge)
            {
                ge = boost::add_edge(u, v, _g).first;
                _eweight_c[ge] = 0;
                if (_measured)
                    _mstate[0].template get_u_edge<true>(u, v) = ge;

            }
            _eweight[ge]++;
            _E++;
            _Es[l]++;
        }

        void set_hparams(double alpha, double beta, double mu, double nu)
        {
            if (_measured)
            {
                _mstate[0]._alpha = alpha;
                _mstate[0]._beta = beta;
                _mstate[0]._mu = mu;
                _mstate[0]._nu = nu;
            }
        }

        uint64_t get_N() { return _mstate[0]._N; }
        uint64_t get_X() { return _mstate[0]._X; }
        uint64_t get_T() { return _mstate[0]._T; }
        uint64_t get_M() { return _mstate[0]._M; }
    };
};

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_LATENT_LAYERS_HH
