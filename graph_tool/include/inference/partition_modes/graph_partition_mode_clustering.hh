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

#ifndef GRAPH_PARTITION_MODE_CLUSTERING_HH
#define GRAPH_PARTITION_MODE_CLUSTERING_HH

#include "config.h"

#include <vector>

#include "../blockmodel/graph_blockmodel_util.hh"
#include "../blockmodel/graph_blockmodel_partition.hh"
#include "../support/graph_state.hh"
#include "graph_partition_mode.hh"

#include "idx_map.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef std::vector<int32_t> b_t;
typedef PartitionModeState::bv_t bv_t;

#define BLOCK_STATE_params                                                     \
    ((g, &, always_directed_never_reversed, 1))                                \
    ((_abg, &, boost::any&, 0))                                                \
    ((obs,, boost::python::object, 0))                                         \
    ((relabel_init,, bool, 0))                                                 \
    ((b, &, b_t&, 0))

GEN_STATE_BASE(ModeClusterStateBase, BLOCK_STATE_params)

template <class T>
static std::vector<T> vrange(size_t N)
{
    std::vector<T> vs(N);
    std::iota(vs.begin(), vs.end(), 0);
    return vs;
}

template <class... Ts>
class ModeClusterState
    : public ModeClusterStateBase<Ts...>
{
public:
    GET_PARAMS_USING(ModeClusterStateBase<Ts...>, BLOCK_STATE_params)
    GET_PARAMS_TYPEDEF(Ts, BLOCK_STATE_params)
    using typename ModeClusterStateBase<Ts...>::args_t;
    using ModeClusterStateBase<Ts...>::dispatch_args;

    template <class... ATs,
              typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
    ModeClusterState(ATs&&... args)
        : ModeClusterStateBase<Ts...>(std::forward<ATs>(args)...),
          _bg(boost::any_cast<std::reference_wrapper<bg_t>>(__abg)),
          _M(python::len(_obs)),
          _pos(_M),
          _modes(_M),
          _wr(_M),
          _bclabel(_M),
          _pclabel(_M),
          _vs(vrange<size_t>(_M)),
          _partition_stats(_g, _b, _vs, 0, _M, _vweight, _eweight, _degs),
          _next_state(_M),
          _args(args...)
    {
        for (int i = 0; i < python::len(_obs); ++i)
        {
            PartitionModeState::bv_t bv;
            for (int l = 0; l < python::len(_obs[i]); ++l)
            {
                PartitionModeState::b_t& b =
                    python::extract<PartitionModeState::b_t&>(_obs[i][l]);
                bv.emplace_back(b);
            }
            _bs.push_back(bv);
        }

        GILRelease gil_release;

        for (size_t r : _b)
            _wr[r]++;

        for (size_t r = 0; r < _M; ++r)
        {
            if (_wr[r] == 0)
                _empty_groups.insert(r);
            else
                _candidate_groups.insert(r);
        }

        for (size_t i = 0; i < _M; ++i)
        {
            auto r = _b[i];
            auto& x = _bs[i];
            _pos[i] = _modes[r].add_partition(x, _relabel_init);
        }
    }

    typedef typename
        std::conditional<is_directed_::apply<g_t>::type::value,
                         GraphInterface::multigraph_t,
                         undirected_adaptor<GraphInterface::multigraph_t>>::type
        bg_t;
    bg_t& _bg;

    std::vector<bv_t> _bs;

    size_t _M;
    std::vector<size_t> _pos;
    std::vector<PartitionModeState> _modes;

    std::vector<size_t> _wr;

    idx_set<size_t> _empty_groups;
    idx_set<size_t> _candidate_groups;

    std::vector<size_t> _bclabel;
    std::vector<size_t> _pclabel;

    UnityPropertyMap<int,GraphInterface::vertex_t> _vweight;
    UnityPropertyMap<int,GraphInterface::edge_t> _eweight;
    simple_degs_t _degs;
    std::vector<size_t> _vs;

    partition_stats<false> _partition_stats;

    std::vector<std::vector<std::tuple<size_t,std::vector<std::vector<int32_t>>>>> _lstack;
    std::vector<std::vector<std::vector<int32_t>>> _next_state;
    std::vector<size_t> _next_list;

    constexpr static BlockStateVirtualBase* _coupled_state = nullptr;

    std::shared_ptr<b_t> _bp;

    args_t _args;

    typedef int m_entries_t;

    typedef char _entropy_args_t;

    PartitionModeState& get_mode(size_t r)
    {
        return _modes[r];
    }

    // =========================================================================
    // State modification
    // =========================================================================

    void move_vertex(size_t v, size_t nr)
    {
        size_t r = _b[v];
        if (nr == r && _next_state[v].empty())
            return;

        _modes[r].remove_partition(_pos[v]);

        auto& x = _bs[v];
        if (_next_state[v].empty())
        {
            _pos[v] = _modes[nr].add_partition(x, true);
        }
        else
        {
            for (size_t l = 0; l < x.size(); ++l)
                x[l].get() = _next_state[v][l];
            _pos[v] = _modes[nr].add_partition(x, false);
        }

        if (nr == r)
            return;

        _wr[r]--;
        _wr[nr]++;

        _partition_stats.remove_vertex(v, r, false, _g,
                                       _vweight, _eweight,
                                       _degs);
        _partition_stats.add_vertex(v, nr, false, _g,
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

    template <class VS>
    void push_state(VS& vs)
    {
        _lstack.emplace_back();
        auto& back = _lstack.back();
        for (auto v : vs)
        {
            auto& bv = _bs[v];
            back.emplace_back();
            get<0>(back.back()) = v;
            auto& x = get<1>(back.back());
            for (auto& b : bv)
                x.push_back(b.get());
        }
    }

    void pop_state()
    {
        auto& back = _lstack.back();
        for (auto& vx : back)
        {
            size_t v = get<0>(vx);
            auto& x = get<1>(vx);
            auto& bv = _bs[v];
            auto r = _b[v];
            _modes[r].remove_partition(_pos[v]);
            for (size_t l = 0; l < bv.size(); ++l)
                bv[l].get() = x[l];
            _pos[v] = _modes[r].add_partition(bv, false);
        }
        _lstack.pop_back();
    }

    void store_next_state(size_t i)
    {
        _next_state[i].resize(_bs[i].size());
        for (size_t l = 0; l < _bs[i].size(); ++l)
            _next_state[i][l] = _bs[i][l].get();
        _next_list.push_back(i);
    }

    void clear_next_state()
    {
        for (auto v : _next_list)
            _next_state[v].clear();
        _next_list.clear();
    }

    void relax_update(bool) {}

    size_t virtual_remove_size(size_t v)
    {
        return _wr[_b[v]] - 1;
    }

    constexpr void add_block(size_t)
    {
    }

    double virtual_move(size_t v, size_t r, size_t nr)
    {
        if (r == nr)
            return 0;

        double dS = 0;

        auto& x = _bs[v];

        dS += _modes[r].virtual_remove_partition(x);
        dS += _modes[nr].virtual_add_partition(x);

        dS += _partition_stats.get_delta_partition_dl(v, r, nr, _vweight);

        return dS;
    }

    size_t get_empty_block(size_t, bool=false)
    {
        return *(_empty_groups.end() - 1);
    }

    size_t sample_block(size_t, double, double d, rng_t& rng)
    {
        std::bernoulli_distribution new_r(d);
        if (d > 0 && !_empty_groups.empty() && new_r(rng))
            return uniform_sample(_empty_groups, rng);
        return uniform_sample(_candidate_groups, rng);
    }

    size_t sample_block_local(size_t v, rng_t& rng)
    {
        return sample_block(v, 0, 0, rng);
    }

    // Computes the move proposal probability
    double get_move_prob(size_t, size_t r, size_t s, double, double d, bool reverse)
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

        if (B == _M)
            d = 0;

        return log(1. - d) - safelog_fast(B);
    }

    template <class MEntries>
    double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                         bool reverse, MEntries&&)
    {
        return get_move_prob(v, r, s, c, d, reverse);
    }

    template <class EArgs>
    double virtual_move(size_t v, size_t r, size_t nr, EArgs&&)
    {
        return virtual_move(v, r, nr);
    }

    template <class EArgs, class MEntries>
    double virtual_move(size_t v, size_t r, size_t nr, EArgs&&, MEntries&&)
    {
        return virtual_move(v, r, nr);
    }

    double virtual_add_partition(bv_t& x, size_t r, bool relabel = true)
    {
        double dS = _modes[r].virtual_add_partition(x, relabel);
        dS += _partition_stats.get_delta_partition_dl(0, null_group, r, _vweight);
        return dS;
    }

    void add_partition(bv_t& x, size_t r, bool relabel = true)
    {
        auto pos = _modes[r].add_partition(x, relabel);
        _pos.push_back(pos);
        _b.push_back(r);
        _bs.push_back(x);
        _partition_stats.change_vertex(0, r, _vweight, 1);
        _wr[r]++;

        _modes.emplace_back();
        _wr.push_back(0);
        _bclabel.push_back(0);
        _pclabel.push_back(0);
        _vs.push_back(_M);
        _next_state.emplace_back();
        _M++;
    }

    template <class RNG>
    double replace_partitions(RNG& rng)
    {
        std::vector<size_t> pos(_M);
        std::iota(pos.begin(), pos.end(), 0);
        std::shuffle(pos.begin(), pos.end(), rng);

        double dS = 0;
        for (auto j : pos)
        {
            auto& mode = _modes[_b[j]];
            auto& b = _bs[j];
            double ddS = mode.virtual_remove_partition(b);
            mode.remove_partition(_pos[j]);
            ddS += mode.virtual_add_partition(b);
            _pos[j] = mode.add_partition(b, ddS < 0);
            if (ddS < 0)
                dS += ddS;
        }
        return dS;
    }

    double entropy()
    {
        double S = 0;
        for (auto r: _candidate_groups)
            S += _modes[r].entropy();
        S += _partition_stats.get_partition_dl();
        return S;
    }

    template <class Eargs>
    double entropy(const Eargs&, bool)
    {
        return entropy();
    }

    double posterior_entropy(bool MLE)
    {
        double S = 0;
        for (size_t r = 0; r < _wr.size(); ++r)
        {
            if (_wr[r] == 0)
                continue;
            S += (_modes[r].posterior_entropy(MLE) * _wr[r]) / _M;
            S += -xlogx(_wr[r] / double(_M));
        }
        return S;
    }

    double posterior_lprob(size_t r, bv_t& b, bool MLE)
    {
        double L = log(_wr[r]) - log(_M);
        L += _modes[r].posterior_lprob(b, MLE);
        return L;
    }

    void relabel_mode(PartitionModeState& x, PartitionModeState& base)
    {
        size_t n = std::max(x._nr.size(), base._nr.size());
        x._nr.resize(n);
        x._count.resize(n);
        base._nr.resize(n);
        base._count.resize(n);

        adj_list<> g;
        typename vprop_map_t<int32_t>::type label(get(vertex_index_t(), g));
        typename vprop_map_t<bool>::type partition(get(vertex_index_t(), g));
        typename eprop_map_t<double>::type mrs(get(edge_index_t(), g));

        get_contingency_graph<true>(g, partition, label, mrs, x._nr, base._nr);

        typedef typename graph_traits<adj_list<>>::vertex_descriptor vertex_t;
        typename vprop_map_t<vertex_t>::type match(get(vertex_index_t(), g));

        auto u = undirected_adaptor<adj_list<>>(g);
        //maximum_weighted_matching(u, mrs, match);
        maximum_bipartite_weighted_matching(u, partition, mrs, match);

        idx_map<int32_t, size_t> x_vertices;
        for (auto v : vertices_range(u))
        {
            if (v < x._B)
                x_vertices[label[v]] = v;
            else
                break;
        }

        std::vector<int32_t> unmatched;
        int32_t max_s = 0;
        for (size_t r = 0; r < x._count.size(); ++r)
        {
            if (x._count[r] == 0)
                continue;
            auto v = match[x_vertices[r]];
            if (v == graph_traits<adj_list<>>::null_vertex())
                unmatched.push_back(r);
            else
                max_s = std::max(max_s, label[v]);
        }

        std::sort(unmatched.begin(), unmatched.end(),
                  [&](auto r, auto s) { return x._count[r] > x._count[s]; });

        idx_map<int32_t, int32_t> umatch;
        for (auto r : unmatched)
            umatch[r] = ++max_s;

        for (auto& jb : x._bs)
        {
            auto bv = x.get_nested_partition(jb.first);
            auto& b = bv[0].get();
            PartitionModeState::b_t b_orig = b;
            for (auto& r : b)
            {
                if (r == -1)
                    continue;
                auto v = match[x_vertices[r]];
                if (v != graph_traits<adj_list<>>::null_vertex())
                    r = label[v];
                else
                    r = umatch[r];
            }

            if (x._coupled_state != nullptr)
            {
                auto& c = x._coupled_state->get_partition(x._coupled_pos[jb.first]);
                relabel_nested(b, b_orig, c);
            }
        }
        x.rebuild_nr();
        if (x._coupled_state != nullptr)
            relabel_mode(*x._coupled_state, *base._coupled_state);
    }

    void relabel_modes(double epsilon, size_t maxiter)
    {
        auto modes = vrange<size_t>(_M);
        std::sort(modes.begin(), modes.end(),
                  [&](auto r, auto s) { return _wr[r] > _wr[s]; });
        PartitionModeState base;

        std::vector<idx_map<size_t, size_t>> pos(_M);
        for (size_t r : modes)
        {
            if (_wr[r] == 0)
                continue;
            auto& mode = _modes[r];
            if (!base._bs.empty())
                relabel_mode(mode, base);
            else
                mode.relabel();
            for (auto& jb : mode._bs)
            {
                auto b = mode.get_nested_partition(jb.first);
                pos[r][jb.first] = base.add_partition(b, false);
            }
        }

        double dS = epsilon + 1;
        size_t iter = 0;
        while (abs(dS) > epsilon && (maxiter == 0 || iter < maxiter))
        {
            dS = 0;
            for (size_t r : modes)
            {
                if (_wr[r] == 0)
                    continue;
                auto& mode = _modes[r];

                for (auto& jb : mode._bs)
                {
                    auto b = mode.get_nested_partition(jb.first);
                    dS += base.virtual_remove_partition(b);
                    base.remove_partition(pos[r][jb.first]);
                }

                if (!base._bs.empty())
                    relabel_mode(mode, base);

                for (auto& jb : mode._bs)
                {
                    auto b = mode.get_nested_partition(jb.first);
                    dS += base.virtual_add_partition(b, false);
                    pos[r][jb.first] = base.add_partition(b, false);
                }
            }
            iter++;
        }
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

    template <class RNG>
    std::pair<size_t, PartitionModeState::b_t>
    sample_partition(bool MLE, RNG& rng)
    {
        auto r = uniform_sample(_b, rng);
        return {r, _modes[r].sample_partition(MLE, rng)};
    }

    template <class RNG>
    std::pair<size_t, std::vector<PartitionModeState::b_t>>
    sample_nested_partition(bool MLE, bool fix_empty, RNG& rng)
    {
        auto r = uniform_sample(_b, rng);
        return {r, _modes[r].sample_nested_partition(MLE, fix_empty, rng)};
    }

    template <size_t... Is>
    ModeClusterState* deep_copy(index_sequence<Is...>)
    {
        b_t* bp = new b_t(_b);
        auto args =
            dispatch_args(_args,
                          [&](std::string name, auto& a) -> decltype(auto)
                          {
                              typedef std::remove_reference_t<decltype(a)> a_t;
                              if (name == "b")
                              {
                                  auto& b_ = *bp; // workaround clang bug
                                  if constexpr (std::is_same_v<a_t, b_t>)
                                      return b_;
                                  return a;
                              }
                              return a;
                          });

        auto state = new ModeClusterState(std::get<Is>(args)...);
        state->_bp = std::shared_ptr<b_t>(bp);
        return state;
    };

    ModeClusterState* deep_copy()
    {
        return deep_copy(make_index_sequence<sizeof...(Ts)>{});
    }

    template <class State>
    void deep_assign(const State& state)
    {
        _b = state._b;
        _wr = state._wr;
        _empty_groups = state._empty_groups;
        _candidate_groups = state._candidate_groups;
        _pos = state._pos;
    }
};

} // graph_tool namespace

#endif //GRAPH_PARTITION_MODE_CLUSTERING_HH
