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

#ifndef GRAPH_MODULARITY_HH
#define GRAPH_MODULARITY_HH

#include "config.h"

#include <vector>

#include "idx_map.hh"

#include "../blockmodel/graph_blockmodel_util.hh"
#include "../support/graph_state.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

struct modularity_entropy_args_t
{
    double gamma;
};

typedef vprop_map_t<int32_t>::type vmap_t;
typedef mpl::vector<eprop_map_t<int32_t>::type,
                    eprop_map_t<double>::type> emaps_t;

#define BLOCK_STATE_params                                                     \
    ((g, &, never_directed, 1))                                                \
    ((eweight,, emaps_t, 1))                                                   \
    ((_abg, &, boost::any&, 0))                                                \
    ((b,, vmap_t, 0))

GEN_STATE_BASE(ModularityStateBase, BLOCK_STATE_params)

template <class... Ts>
class ModularityState
    : public ModularityStateBase<Ts...>
{
public:
    GET_PARAMS_USING(ModularityStateBase<Ts...>, BLOCK_STATE_params)
    GET_PARAMS_TYPEDEF(Ts, BLOCK_STATE_params)
    using typename ModularityStateBase<Ts...>::args_t;
    using ModularityStateBase<Ts...>::dispatch_args;

    typedef partition_stats<false> partition_stats_t;

    template <class... ATs,
              typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
    ModularityState(ATs&&... args)
        : ModularityStateBase<Ts...>(std::forward<ATs>(args)...),
          _bg(boost::any_cast<std::reference_wrapper<bg_t>>(__abg)),
          _N(HardNumVertices()(_g)),
          _E(0),
          _bclabel(_N),
          _pclabel(_N),
          _wr(_N),
          _args(std::forward<ATs>(args)...)
    {
        GILRelease gil_release;

        _wr.resize(num_vertices(_g), 0);
        _er.resize(num_vertices(_g), 0);
        _err.resize(num_vertices(_g), 0);

        for (auto v : vertices_range(_g))
        {
            auto r = _b[v];
            _er[r] += out_degreeS()(v, _g, _eweight);
            _wr[r]++;
        }

        for (size_t r = 0; r < _N; ++r)
        {
            if (_wr[r] == 0)
                _empty_groups.insert(r);
            else
                _candidate_groups.insert(r);
        }

        for (auto e : edges_range(_g))
        {
            auto r = _b[source(e, _g)];
            auto s = _b[target(e, _g)];
            if (r == s)
                _err[r] += 2 * _eweight[e];
            _E += _eweight[e];
        }
    }

    typedef typename
        std::conditional<is_directed_::apply<g_t>::type::value,
                         GraphInterface::multigraph_t,
                         undirected_adaptor<GraphInterface::multigraph_t>>::type
        bg_t;
    bg_t& _bg;

    typedef typename property_traits<eweight_t>::value_type w_t;

    size_t _N;
    w_t _E;

    idx_set<size_t> _empty_groups;
    idx_set<size_t> _candidate_groups;

    std::vector<size_t> _bclabel;
    std::vector<size_t> _pclabel;
    std::vector<size_t> _wr;

    constexpr static BlockStateVirtualBase* _coupled_state = nullptr;

    typedef int m_entries_t;

    UnityPropertyMap<int,GraphInterface::vertex_t> _vweight;
    simple_degs_t _degs;

    args_t _args;

    typedef modularity_entropy_args_t _entropy_args_t;

    // =========================================================================
    // State modification
    // =========================================================================

    void move_vertex(size_t v, size_t nr)
    {
        size_t r = _b[v];
        if (nr == r)
            return;

        size_t k = 0;
        w_t m = 0;
        for (auto e : out_edges_range(v, _g))
        {
            auto ew = _eweight[e];
            k += ew;
            auto u = target(e, _g);
            if (u == v)
            {
                m += ew;
                continue;
            }
            size_t s = _b[u];
            if (s == r)
                _err[r] -= 2 * ew;
            else if (s == nr)
                _err[nr] += 2 * ew;
        }

        _err[r] -= m;
        _err[nr] += m;

        _er[r] -= k;
        _er[nr] += k;

        _wr[r]--;
        _wr[nr]++;

        if (_wr[r] == 0)
        {
            _empty_groups.insert(r);
            _candidate_groups.erase(r);
        }

        if (_wr[nr] == 1)
        {
            _empty_groups.erase(nr);
            _candidate_groups.insert(nr);
        }

        _b[v] = nr;
    }

    template <class ME>
    void move_vertex(size_t v, size_t nr, ME&)
    {
        move_vertex(v, nr);
    }

    size_t virtual_remove_size(size_t v)
    {
        return _wr[_b[v]] - 1;
    }

    constexpr void add_block(size_t)
    {
    }

    double virtual_move(size_t v, size_t r, size_t nr,
                        const modularity_entropy_args_t& ea)
    {
        if (r == nr)
            return 0;

        std::array<w_t, 2> derr({0,0});
        w_t k = 0;
        w_t m = 0;
        for (auto e : out_edges_range(v, _g))
        {
            auto ew = _eweight[e];
            k += ew;
            auto u = target(e, _g);
            if (u == v)
            {
                m += ew;
                continue;
            }
            size_t s = _b[u];
            if (s == r)
                derr[0] -= 2 * ew;
            else if (s == nr)
                derr[1] += 2 * ew;
        }
        derr[0] -= m;
        derr[1] += m;

        double Qb = 0;
        double Qa = 0;

        double M = 2 * _E;

        Qb += _err[r] - ea.gamma * _er[r] * (_er[r] / M);
        Qb += _err[nr] - ea.gamma * _er[nr] * (_er[nr] / M);

        Qa += (_err[r] + derr[0]) - ea.gamma * (_er[r] - k) * ((_er[r] - k) / M);
        Qa += (_err[nr] + derr[1]) - ea.gamma * (_er[nr] + k) * ((_er[nr] + k) / M);

        double dS = -(Qa - Qb);
        return dS;
    }

    size_t get_empty_block(size_t, bool=false)
    {
        return *(_empty_groups.end() - 1);
    }

    size_t sample_block(size_t v, double c, double d, rng_t& rng)
    {
        std::bernoulli_distribution new_r(d);
        if (d > 0 && !_empty_groups.empty() && new_r(rng))
            return uniform_sample(_empty_groups, rng);
        c = std::max(std::min(c, 1.), 0.);
        std::bernoulli_distribution adj(1.-c);
        auto iter = out_neighbors(v, _g);
        if (iter.first != iter.second && adj(rng))
        {
            auto w = uniform_sample(iter.first, iter.second, rng);
            return _b[w];
        }
        return uniform_sample(_candidate_groups, rng);
    }

    size_t sample_block_local(size_t v, rng_t& rng)
    {
        return sample_block(v, 0, 0, rng);
    }

    // Computes the move proposal probability
    double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                         bool reverse)
    {
        size_t B = _candidate_groups.size();
        if (reverse)
        {
            if (_wr[s] == 1)
                return log(d);
            if (_wr[r] == 0)
                B++;
        }
        else
        {
            if (_wr[s] == 0)
                return log(d);
        }

        size_t k_s = 0;
        size_t k = 0;
        for (auto w : out_neighbors_range(v, _g))
        {
            if (size_t(_b[w]) == s)
                k_s++;
            k++;
        }

        if (B == _N)
            d = 0;

        if (k > 0)
        {
            double p = k_s / double(k);
            c = 1 - std::max(std::min(c, 1.), 0.);
            return log1p(-d) + log(c * p + (1. - c)/B);
        }
        else
        {
            return log1p(-d) - log(B);
        }
    }

    template <class MEntries>
    double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                         bool reverse, MEntries&&)
    {
        return get_move_prob(v, r, s, c, d, reverse);
    }

    template <class EArgs, class MEntries>
    double virtual_move(size_t v, size_t r, size_t nr, EArgs&& ea, MEntries&&)
    {
        return virtual_move(v, r, nr, ea);
    }

    double entropy(const modularity_entropy_args_t& ea)
    {
        double Q = 0;
        double M = 2 * _E;
        for (auto r : _candidate_groups)
            Q += _err[r] - ea.gamma * _er[r] * (_er[r] / M);
        return -Q;
    }

    double entropy(const modularity_entropy_args_t& ea, bool)
    {
        return entropy(ea);
    }

    template <class MCMCState>
    void init_mcmc(MCMCState&)
    {
    }

    constexpr size_t node_weight(size_t)
    {
        return 1;
    }

    bool is_last(size_t v)
    {
        return _wr[_b[v]] == 1;
    }

    constexpr bool allow_move(size_t, size_t)
    {
        return true;
    }

    template <class V>
    void push_state(V&) {}
    void pop_state() {}
    void store_next_state(size_t) {}
    void clear_next_state() {}
    void relax_update(bool) {}

    std::vector<w_t> _er;
    std::vector<w_t> _err;

    template <size_t... Is>
    ModularityState* deep_copy(index_sequence<Is...>)
    {
        auto b = _b.copy();
        auto args =
            dispatch_args(_args,
                          [&](std::string name, auto& a) -> decltype(auto)
                          {
                              typedef std::remove_reference_t<decltype(a)> a_t;
                              if (name == "b")
                              {
                                  auto& b_ = b; // workaround clang bug
                                  if constexpr (std::is_same_v<a_t, b_t>)
                                      return b_;
                                  return a;
                              }
                              return a;
                          });

        auto state = new ModularityState(std::get<Is>(args)...);
        return state;
    };

    ModularityState* deep_copy()
    {
        return deep_copy(make_index_sequence<sizeof...(Ts)>{});
    }

    template <class State>
    void deep_assign(const State& state)
    {
        _b.get_storage() = state._b.get_storage();
        _er = state._er;
        _err = state._err;
        _wr = state._wr;
        _empty_groups = state._empty_groups;
        _candidate_groups = state._candidate_groups;
    }
};

} // graph_tool namespace

#endif //GRAPH_MODULARITY_HH
