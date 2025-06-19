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

#ifndef GRAPH_PLANTED_PARTITION_HH
#define GRAPH_PLANTED_PARTITION_HH

#include "config.h"

#include <vector>

#include "../blockmodel/graph_blockmodel_util.hh"
#include "../support/graph_state.hh"

#include "idx_map.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

struct pp_entropy_args_t
{
    bool uniform;
    deg_dl_kind  degree_dl_kind;
};


typedef vprop_map_t<int32_t>::type vmap_t;

#define BLOCK_STATE_params                                                     \
    ((g, &, never_directed, 1))                                                \
    ((_abg, &, boost::any&, 0))                                                \
    ((b,, vmap_t, 0))                                                          \
    ((wr, &, vector<size_t>&, 0))                                              \
    ((er, &, vector<size_t>&, 0))                                              \
    ((err, &, vector<size_t>&, 0))                                             \
    ((eio, &, vector<size_t>&, 0))

GEN_STATE_BASE(PPStateBase, BLOCK_STATE_params)

template <class... Ts>
class PPState
    : public PPStateBase<Ts...>
{
public:
    GET_PARAMS_USING(PPStateBase<Ts...>, BLOCK_STATE_params)
    GET_PARAMS_TYPEDEF(Ts, BLOCK_STATE_params)
    using typename PPStateBase<Ts...>::args_t;
    using PPStateBase<Ts...>::dispatch_args;

    typedef partition_stats<false> partition_stats_t;

    template <class... ATs,
              typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
    PPState(ATs&&... args)
        : PPStateBase<Ts...>(std::forward<ATs>(args)...),
          _bg(boost::any_cast<std::reference_wrapper<bg_t>>(__abg)),
          _N(HardNumVertices()(_g)),
          _E(HardNumEdges()(_g)),
          _bclabel(num_vertices(_g)),
          _pclabel(num_vertices(_g)),
          _partition_stats(_g, _b, vertices_range(_g), _E,
                           num_vertices(_g), _vweight, _eweight, _degs),
          _args(std::forward<ATs>(args)...)
    {
        GILRelease gil_release;

        _wr.resize(num_vertices(_g), 0);
        _er.resize(num_vertices(_g), 0);
        _err.resize(num_vertices(_g), 0);
        _eio.resize(2, 0);

        for (auto v : vertices_range(_g))
        {
            auto r = _b[v];
            _wr[r]++;
            _er[r] += out_degree(v, _g);
        }

        for (size_t r = 0; r < num_vertices(_g); ++r)
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
            {
                _err[r] += 2;
                _eio[0]++;
            }
            else
            {
                _eio[1]++;
            }
        }
    }

    typedef typename
        std::conditional<is_directed_::apply<g_t>::type::value,
                         GraphInterface::multigraph_t,
                         undirected_adaptor<GraphInterface::multigraph_t>>::type
        bg_t;
    bg_t& _bg;

    size_t _N;
    size_t _E;

    idx_set<size_t> _empty_groups;
    idx_set<size_t> _candidate_groups;

    std::vector<size_t> _bclabel;
    std::vector<size_t> _pclabel;

    constexpr static BlockStateVirtualBase* _coupled_state = nullptr;

    typedef int m_entries_t;

    UnityPropertyMap<int,GraphInterface::vertex_t> _vweight;
    UnityPropertyMap<int,GraphInterface::edge_t> _eweight;
    simple_degs_t _degs;

    partition_stats_t _partition_stats;

    args_t _args;

    typedef pp_entropy_args_t _entropy_args_t;

    // =========================================================================
    // State modification
    // =========================================================================

    void move_vertex(size_t v, size_t nr)
    {
        size_t r = _b[v];
        if (nr == r)
            return;

        size_t k = 0;
        size_t m = 0;
        for (auto e : out_edges_range(v, _g))
        {
            ++k;
            auto u = target(e, _g);
            if (u == v)
            {
                ++m;
                continue;
            }
            size_t s = _b[u];
            if (s == r)
            {
                _err[r] -= 2;
                _eio[0]--;
            }
            else
            {
                _eio[1]--;
            }
            if (s == nr)
            {
                _err[nr] += 2;
                _eio[0]++;
            }
            else
            {
                _eio[1]++;
            }
        }

        _err[r] -= m;
        _err[nr] += m;

        _wr[r]--;
        _wr[nr]++;
        _er[r] -= k;
        _er[nr] += k;

        _partition_stats.remove_vertex(v, r, true, _g,
                                       _vweight, _eweight,
                                       _degs);
        _partition_stats.add_vertex(v, nr, true, _g,
                                    _vweight, _eweight,
                                    _degs);
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

    double virtual_move(size_t v, size_t r, size_t nr, const pp_entropy_args_t& ea)
    {
        if (r == nr)
            return 0;

        double Sb = 0;
        double Sa = 0;

        std::array<int, 2> dio({0,0});
        std::array<int, 2> derr({0,0});
        size_t k = 0;
        size_t m = 0;
        for (auto e : out_edges_range(v, _g))
        {
            ++k;
            auto u = target(e, _g);
            if (u == v)
            {
                ++m;
                continue;
            }
            size_t s = _b[u];
            if (s == r)
            {
                derr[0] -= 2;
                dio[0]--;
            }
            else
            {
                dio[1]--;
            }
            if (s == nr)
            {
                derr[1] += 2;
                dio[0]++;
            }
            else
            {
                dio[1]++;
            }
        }
        derr[0] -= m;
        derr[1] += m;

        size_t B = _candidate_groups.size();
        int dB = 0;
        if (_wr[r] == 1)
            dB--;
        if (_wr[nr] == 0)
            dB++;

        if (ea.uniform)
        {
            Sb -= lgamma_fast(_eio[0] + 1);
            Sb -= lgamma_fast(_eio[1] + 1);
            Sb += lgamma_fast(_er[r] + 1);
            Sb += lgamma_fast(_er[nr] + 1);
            Sb += (safelog_fast(B) - log(2)) * _eio[0];
            Sb += lbinom_fast(B, size_t(2)) * _eio[1];
            if (B > 1)
                Sb += safelog_fast(_E + 1);

            Sa -= lgamma_fast(_eio[0] + dio[0] + 1);
            Sa -= lgamma_fast(_eio[1] + dio[1] + 1);
            Sa += lgamma_fast(_er[r] - k + 1);
            Sa += lgamma_fast(_er[nr] + k + 1);
            Sa += (safelog_fast(B + dB) - log(2)) * (_eio[0] + dio[0]);
            Sa += lbinom_fast(B + dB, size_t(2)) * (_eio[1] + dio[1]);
            if (B + dB > 1)
                Sa += safelog_fast(_E + 1);
        }
        else
        {
            Sb += lgamma_fast(_er[r] + 1);
            Sb += lgamma_fast(_er[nr] + 1);
            Sb -= (_err[r]/2) * log(2) + lgamma_fast(_err[r]/2 + 1);
            Sb -= (_err[nr]/2) * log(2) + lgamma_fast(_err[nr]/2 + 1);
            Sb -= lgamma_fast(_eio[1] + 1);

            Sa += lgamma_fast(_er[r] - k + 1);
            Sa += lgamma_fast(_er[nr] + k + 1);
            Sa -= ((_err[r] + derr[0])/2) * log(2) + lgamma_fast((_err[r] + derr[0])/2 + 1);
            Sa -= ((_err[nr] + derr[1])/2) * log(2) + lgamma_fast((_err[nr] + derr[1])/2 + 1);
            Sa -= lgamma_fast(_eio[1] + dio[1] + 1);


            Sb += _eio[1] * lbinom_fast(B, size_t(2));
            Sb += lbinom_fast(B + _eio[0] - 1, size_t(_eio[0]));
            if (B > 1)
                Sb += safelog_fast(_E + 1);

            Sa += (_eio[1] + dio[1]) * lbinom_fast(size_t(B + dB), size_t(2));
            Sa += lbinom_fast(B + dB + _eio[0] + dio[0] - 1, size_t(_eio[0] + dio[0]));
            if (B + dB > 1)
                Sa += safelog_fast(_E + 1);
        }

        double dS = (Sa - Sb);

        dS += _partition_stats.get_delta_partition_dl(v, r, nr, _vweight);
        dS += _partition_stats.get_delta_deg_dl(v, r, nr, _vweight, _eweight,
                                                _degs, _g, ea.degree_dl_kind);
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
        std::bernoulli_distribution adj(1-c);
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

    double entropy(const pp_entropy_args_t& ea)
    {
        double S = 0;

        size_t B = _candidate_groups.size();

        if (ea.uniform)
        {
            S -= lgamma_fast(_eio[0] + 1);
            S -= lgamma_fast(_eio[1] + 1);
            for (auto r : _candidate_groups)
                S += lgamma_fast(_er[r] + 1);
            S += (safelog_fast(B) - log(2)) * _eio[0];
            S += lbinom_fast(B, size_t(2)) * _eio[1];
            if (B > 1)
                S += safelog_fast(_E + 1);
        }
        else
        {
            S -= lgamma_fast(_eio[1] + 1);
            S += (_eio[1]) * lbinom_fast(B, size_t(2));
            for (auto r : _candidate_groups)
            {
                S += lgamma_fast(_er[r] + 1);
                S -= (_err[r]/2) * log(2) + lgamma_fast(_err[r]/2 + 1);
            }

            S += lbinom_fast(B + _eio[0] - 1, size_t(_eio[0]));
            if (B > 1)
                S += safelog_fast(_E + 1);
        }

        S += _partition_stats.get_partition_dl();
        S += _partition_stats.get_deg_dl(ea.degree_dl_kind);

        for (auto v : vertices_range(_g))
        {
            S -= lgamma_fast(out_degree(v, _g) + 1);

            gt_hash_map<decltype(v), size_t> us;
            for (auto e : out_edges_range(v, _g))
            {
                auto u = target(e, _g);
                if (u < v)
                    continue;
                us[u] += _eweight[e];
            }

            for (auto& uc : us)
            {
                auto& u = uc.first;
                auto& m = uc.second;
                if (m > 1)
                {
                    if (u == v)
                    {
                        assert(m % 2 == 0);
                        S += lgamma_fast(m/2 + 1) + m * log(2) / 2;
                    }
                    else
                    {
                        S += lgamma_fast(m + 1);
                    }
                }
            }
        }

        return S;
    }

    double entropy(const pp_entropy_args_t& ea, bool)
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

    //owned by deep copy
    std::shared_ptr<wr_t> _wr_p;
    std::shared_ptr<er_t> _er_p;
    std::shared_ptr<err_t> _err_p;
    std::shared_ptr<eio_t> _eio_p;

    template <size_t... Is>
    PPState* deep_copy(index_sequence<Is...>)
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
                              else if (name == "wr")
                              {
                                  if constexpr (std::is_same_v<a_t, wr_t>)
                                      return *(new er_t(this->_wr));
                                  return a;
                              }
                              else if (name == "er")
                              {
                                  if constexpr (std::is_same_v<a_t, er_t>)
                                      return *(new er_t(this->_er));
                                  return a;
                              }
                              else if (name == "err")
                              {
                                  if constexpr (std::is_same_v<a_t, err_t>)
                                      return *(new err_t(this->_err));
                                  return a;
                              }
                              else if (name == "eio")
                              {
                                  if constexpr (std::is_same_v<a_t, eio_t>)
                                      return *(new err_t(this->_err));
                                  return a;
                              }
                              return a;
                          });

        auto state = new PPState(std::get<Is>(args)...);
        state->_wr_p = std::shared_ptr<wr_t>(&state->_wr);
        state->_er_p = std::shared_ptr<er_t>(&state->_er);
        state->_err_p = std::shared_ptr<err_t>(&state->_err);
        state->_eio_p = std::shared_ptr<eio_t>(&state->_eio);
        return state;
    };

    PPState* deep_copy()
    {
        return deep_copy(make_index_sequence<sizeof...(Ts)>{});
    }

    template <class State>
    void deep_assign(const State& state)
    {
        _b.get_storage() = state._b.get_storage();
        _wr = state._wr;
        _er = state._er;
        _err = state._err;
        _eio = state._eio;
        _partition_stats = state._partition_stats;
        _empty_groups = state._empty_groups;
        _candidate_groups = state._candidate_groups;
    }
};

} // graph_tool namespace

#endif //GRAPH_PLANTED_PARTITION_HH
