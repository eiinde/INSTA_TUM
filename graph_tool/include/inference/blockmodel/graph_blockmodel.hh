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

#ifndef GRAPH_BLOCKMODEL_HH
#define GRAPH_BLOCKMODEL_HH

#include "config.h"

#include <vector>

#ifdef __clang__
#include <boost/algorithm/minmax_element.hpp>
#endif

#include "idx_map.hh"

#include "../support/graph_state.hh"
#include "graph_blockmodel_util.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef vprop_map_t<int32_t>::type vmap_t;
typedef eprop_map_t<int32_t>::type emap_t;
typedef UnityPropertyMap<int,GraphInterface::vertex_t> vcmap_t;
typedef UnityPropertyMap<int,GraphInterface::edge_t> ecmap_t;

template <class PMap>
auto uncheck(boost::any& amap, PMap*)
{
    return any_cast<typename PMap::checked_t&>(amap).get_unchecked();
}

template <class T, class V>
auto& uncheck(boost::any& amap, UnityPropertyMap<T,V>*)
{
    return any_cast<UnityPropertyMap<T,V>&>(amap);
}

inline simple_degs_t uncheck(boost::any& amap, simple_degs_t*)
{
    return any_cast<simple_degs_t>(amap);
}

typedef mpl::vector2<std::true_type, std::false_type> bool_tr;
typedef mpl::vector2<vcmap_t, vmap_t> vweight_tr;
typedef mpl::vector2<ecmap_t, emap_t> eweight_tr;

#ifdef GRAPH_BLOCKMODEL_RMAP_ENABLE
#    ifdef GRAPH_BLOCKMODEL_RMAP_ALL_ENABLE
typedef mpl::vector2<std::true_type, std::false_type> rmap_tr;
#    else
typedef mpl::vector1<std::true_type> rmap_tr;
#    endif
#else
typedef mpl::vector1<std::false_type> rmap_tr;
#endif

#ifndef GRAPH_VIEWS
#define GRAPH_VIEWS all_graph_views
#endif

#define BLOCK_STATE_params_gen(GVIEWS)                                         \
    ((g, &, GVIEWS, 1))                                                        \
    ((is_weighted,, mpl::vector1<std::true_type>, 1))                          \
    ((use_hash,, bool_tr, 1))                                                  \
    ((use_rmap,, rmap_tr, 1))                                                  \
    ((_abg,, boost::any, 0))                                                   \
    ((_aeweight,, boost::any, 0))                                              \
    ((_avweight,, boost::any, 0))                                              \
    ((mrs,, emap_t, 0))                                                        \
    ((mrp,, vmap_t, 0))                                                        \
    ((mrm,, vmap_t, 0))                                                        \
    ((wr,, vmap_t, 0))                                                         \
    ((b,, vmap_t, 0))                                                          \
    ((bclabel,, vmap_t, 0))                                                    \
    ((pclabel,, vmap_t, 0))                                                    \
    ((bfield,, vprop_map_t<std::vector<double>>::type, 0))                     \
    ((Bfield, &, std::vector<double>&, 0))                                     \
    ((deg_corr,, bool, 0))                                                     \
    ((rec_types,, std::vector<int32_t>, 0))                                    \
    ((rec,, std::vector<eprop_map_t<double>::type>, 0))                        \
    ((drec,, std::vector<eprop_map_t<double>::type>, 0))                       \
    ((brec,, std::vector<eprop_map_t<double>::type>, 0))                       \
    ((bdrec,, std::vector<eprop_map_t<double>::type>, 0))                      \
    ((brecsum,, vprop_map_t<double>::type, 0))                                 \
    ((wparams, &, std::vector<std::vector<double>>&, 0))                       \
    ((recdx, &, std::vector<double>&, 0))                                      \
    ((Lrecdx, &, std::vector<double>&, 0))                                     \
    ((epsilon, &, std::vector<double>&, 0))

#define BLOCK_STATE_params BLOCK_STATE_params_gen(GRAPH_VIEWS)

GEN_STATE_BASE(BlockStateBase, BLOCK_STATE_params)

template <class... Ts>
class BlockState
    : public BlockStateBase<Ts...>,
      public virtual BlockStateVirtualBase
{
public:
    GET_PARAMS_USING(BlockStateBase<Ts...>, BLOCK_STATE_params)
    GET_PARAMS_TYPEDEF(Ts, BLOCK_STATE_params)
    using typename BlockStateBase<Ts...>::args_t;
    using BlockStateBase<Ts...>::dispatch_args;

    typedef partition_stats<use_rmap_t::value> partition_stats_t;

    typedef typename std::conditional<is_weighted_t::value,
                                      vmap_t::unchecked_t, vcmap_t>::type vweight_t;
    typedef typename std::conditional<is_weighted_t::value,
                                      emap_t::unchecked_t, ecmap_t>::type eweight_t;

    template <class... ATs,
              typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
    BlockState(ATs&&... args)
        : BlockStateBase<Ts...>(std::forward<ATs>(args)...),
          _bg(boost::any_cast<std::reference_wrapper<bg_t>>(__abg)),
          _c_mrs(_mrs.get_checked()),
          _vweight(uncheck(__avweight, typename std::add_pointer<vweight_t>::type())),
          _eweight(uncheck(__aeweight, typename std::add_pointer<eweight_t>::type())),
          _emat(_g, _bg),
          _m_entries(num_vertices(_bg)),
          _args(std::forward<ATs>(args)...)
    {
        GILRelease gil_release;

        for (auto r : vertices_range(_bg))
        {
            if (_wr[r] == 0)
                _empty_groups.insert(r);
            else
                _candidate_groups.insert(r);
        }

        for (auto& p : _rec)
            _c_rec.push_back(p.get_checked());
        for (auto& p : _drec)
            _c_drec.push_back(p.get_checked());
        for (auto& p : _brec)
        {
            _c_brec.push_back(p.get_checked());
            double x = 0;
            for (auto me : edges_range(_bg))
                x += p[me];
            _recsum.push_back(x);
        }
        for (auto& p : _bdrec)
            _c_bdrec.push_back(p.get_checked());

        if (!_rec_types.empty())
        {
            _recx2.resize(_rec_types.size());
            _recdx.resize(_rec_types.size());
            for (auto me : edges_range(_bg))
            {
                if (_brec[0][me] > 0)
                {
                    _B_E++;
                    for (size_t i = 0; i < _rec_types.size(); ++i)
                    {
                        if (_rec_types[i] == weight_type::REAL_NORMAL)
                        {
                            _recx2[i] += std::pow(_brec[i][me], 2);
                            if (_brec[0][me] > 1)
                                _recdx[i] += \
                                    (_bdrec[i][me] -
                                     std::pow(_brec[i][me], 2) / _brec[0][me]);
                        }
                    }
                }
                if (_brec[0][me] > 1)
                    _B_E_D++;
            }
        }

        _rt = weight_type::NONE;
        for (auto rt : _rec_types)
        {
            _rt = rt;
            if (rt == weight_type::REAL_NORMAL)
                break;
        }
        _dBdx.resize(_rec_types.size());

        _N = 0;

        if constexpr (std::is_same_v<degs_t, degs_map_t::unchecked_t>)
            _degs.resize(num_vertices(_g));

        for (auto v : vertices_range(_g))
        {
            _N += _vweight[v];
            if constexpr (std::is_same_v<degs_t, degs_map_t::unchecked_t>)
            {
                _degs[v] = {in_degreeS()(v, _g, _eweight),
                            out_degreeS()(v, _g, _eweight)};
            }
        }

        _E = 0;
        for (auto e : edges_range(_g))
            _E += _eweight[e];

        init_partition_stats();
    }

    // =========================================================================
    // State modification
    // =========================================================================

    template <class MEntries, class EFilt>
    void get_move_entries(size_t v, size_t r, size_t nr, MEntries& m_entries,
                          EFilt&& efilt)
    {
        auto mv_entries = [&](auto&&... args)
            {
                move_entries(v, r, nr, _b, _g, _eweight, num_vertices(_bg),
                             m_entries, std::forward<EFilt>(efilt),
                             is_loop_nop(),
                             std::forward<decltype(args)>(args)...);
            };

        if (_rt == weight_type::NONE)
        {
            mv_entries();
        }
        else
        {
            if (_rt == weight_type::REAL_NORMAL)
                mv_entries(_rec, _drec);
            else
                mv_entries(_rec);
        }
    }

    template <class MEntries>
    void get_move_entries(size_t v, size_t r, size_t nr, MEntries& m_entries)
    {
        get_move_entries(v, r, nr, m_entries, [](auto) {return false;});
    }


    template <bool Add, class EFilt>
    void modify_vertex(size_t v, size_t r, EFilt&& efilt)
    {
        if constexpr (Add)
            get_move_entries(v, null_group, r, _m_entries,
                             std::forward<EFilt>(efilt));
        else
            get_move_entries(v, r, null_group, _m_entries,
                             std::forward<EFilt>(efilt));

        apply_delta<Add,!Add>(*this, _m_entries);

        if constexpr (Add)
            BlockState::add_partition_node(v, r);
        else
            BlockState::remove_partition_node(v, r);
    }

    bool allow_move(size_t r, size_t nr)
    {
        if (_coupled_state != nullptr)
        {
            auto& hb = _coupled_state->get_b();
            auto rr = hb[r];
            auto ss = hb[nr];
            if (rr != ss && !_coupled_state->allow_move(rr, ss))
                return false;
        }
        return _bclabel[r] == _bclabel[nr];
    }

    template <class MEntries>
    void move_vertex(size_t v, size_t r, size_t nr, MEntries& m_entries)
    {
        if (r == nr)
            return;

        apply_delta<true, true>(*this, m_entries);

        BlockState::remove_partition_node(v, r);
        BlockState::add_partition_node(v, nr);
    }

    template <class MEntries>
    void move_vertex(size_t v, size_t nr, MEntries& m_entries)
    {
        size_t r = _b[v];
        move_vertex(v, r, nr, m_entries);
    }

    // move a vertex from its current block to block nr
    void move_vertex(size_t v, size_t r, size_t nr)
    {
        if (r == nr)
            return;

        if (!allow_move(r, nr))
            throw ValueException("cannot move vertex across clabel barriers");

        get_move_entries(v, r, nr, _m_entries,
                         [](auto&) constexpr {return false;});

        move_vertex(v, r, nr, _m_entries);
    }

    void move_vertex(size_t v, size_t nr)
    {
        size_t r = _b[v];
        move_vertex(v, r, nr);
    }

    void propagate_delta(size_t u, size_t v,
                         std::vector<std::tuple<size_t, size_t,
                                                GraphInterface::edge_t, int,
                                                std::vector<double>>>& entries)
    {
        size_t r = _b[u];
        size_t s = _b[v];
        _m_entries.set_move(r, s, num_vertices(_bg));

        if (_rt == weight_type::NONE)
        {
            for (auto& rsd : entries)
            {
                _m_entries.template insert_delta<true>(_b[get<0>(rsd)],
                                                       _b[get<1>(rsd)],
                                                       get<3>(rsd));
            }
        }
        else
        {
            for (auto& rsd : entries)
            {
                recs_propagate_insert(*this, _b[get<0>(rsd)], _b[get<1>(rsd)],
                                      get<2>(rsd), get<3>(rsd), get<4>(rsd),
                                      _m_entries);
            }
        }
        apply_delta<true, true>(*this, _m_entries);
    }

    void add_edge(const GraphInterface::edge_t& e)
    {
        size_t r = _b[source(e, _g)];
        size_t s = _b[target(e, _g)];
        auto me = _emat.get_me(r, s);
        if (me == _emat.get_null_edge())
        {
            me = boost::add_edge(r, s, _bg).first;
            _emat.put_me(r, s, me);
            _c_mrs[me] = 0;
            for (size_t i = 0; i < _rec_types.size(); ++i)
            {
                _c_brec[i][me] = 0;
                _c_bdrec[i][me] = 0;
            }
            if (_coupled_state != nullptr)
                _coupled_state->add_edge(me);
        }
    }

    void remove_edge(const GraphInterface::edge_t& e)
    {
        size_t r = _b[source(e, _g)];
        size_t s = _b[target(e, _g)];
        auto me = _emat.get_me(r, s);
        if (me != _emat.get_null_edge() && _mrs[me] == 0)
        {
            _emat.remove_me(me, _bg);
            if (_coupled_state != nullptr)
                _coupled_state->remove_edge(me);
            else
                boost::remove_edge(me, _bg);
        }
        assert(e != _emat.get_null_edge());
        boost::remove_edge(e, _g);
    }

    void add_edge_rec(const GraphInterface::edge_t& e)
    {
        if (_rec_types.empty())
            return;
        auto crec = _rec[0].get_checked();
        crec[e] = 1;
        for (size_t i = 1; i < _rec_types.size(); ++i)
        {
            auto drec = _drec[i].get_checked();
            drec[e] = 0;
        }
    }

    void remove_edge_rec(const GraphInterface::edge_t& e)
    {
        if (_rec_types.empty())
            return;
        _rec[0][e] = 0;
    }

    void update_edge_rec(const GraphInterface::edge_t& e,
                         const std::vector<double>& delta)
    {
        if (_rec_types.empty())
            return;

        for (size_t i = 0; i < _rec_types.size(); ++i)
        {
            if (_rec_types[i] != weight_type::REAL_NORMAL)
                continue;

            auto rec = _c_rec[i][e];
            auto d = (std::pow(rec, 2) -
                      std::pow(rec - delta[i], 2));
            _c_drec[i][e] += d;
        }
    }

    void remove_partition_node(size_t v, size_t r)
    {
        assert(size_t(_b[v]) == r);

        if (_vweight[v] > 0 && _wr[r] == _vweight[v])
        {
            _candidate_groups.erase(r);
            _empty_groups.insert(r);

            if (_coupled_state != nullptr)
            {
                auto& hb = _coupled_state->get_b();
                _coupled_state->remove_partition_node(r, hb[r]);
                _coupled_state->set_vertex_weight(r, 0);
            }
        }

        _wr[r] -= _vweight[v];

        get_partition_stats(v).remove_vertex(v, r, _deg_corr, _g,
                                             _vweight, _eweight,
                                             _degs);
    }

    void add_partition_node(size_t v, size_t r)
    {
        _b[v] = r;

        _wr[r] += _vweight[v];

        get_partition_stats(v).add_vertex(v, r, _deg_corr, _g, _vweight,
                                          _eweight, _degs);

        if (_vweight[v] > 0 && _wr[r] == _vweight[v])
        {
            _empty_groups.erase(r);
            _candidate_groups.insert(r);

            if (_coupled_state != nullptr)
            {
                auto& hb = _coupled_state->get_b();
                _coupled_state->set_vertex_weight(r, 1);
                _coupled_state->add_partition_node(r, hb[r]);
            }
        }
    }

    template <class EFilt>
    void remove_vertex(size_t v, size_t r, EFilt&& efilt)
    {
        modify_vertex<false>(v, r, std::forward<EFilt>(efilt));
    }

    void remove_vertex(size_t v, size_t r)
    {
        remove_vertex(v, r, [](auto&) { return false; });
    }

    void remove_vertex(size_t v)
    {
        size_t r = _b[v];
        remove_vertex(v, r);
    }

    template <class Vlist>
    void remove_vertices(Vlist& vs)
    {
        typedef typename graph_traits<g_t>::vertex_descriptor vertex_t;
        typedef typename graph_traits<g_t>::edge_descriptor edges_t;

        gt_hash_set<vertex_t> vset(vs.begin(), vs.end());
        gt_hash_set<edges_t> eset;

        for (auto v : vset)
        {
            for (auto e : all_edges_range(v, _g))
            {
                auto u = (source(e, _g) == v) ? target(e, _g) : source(e, _g);
                if (vset.find(u) != vset.end())
                    eset.insert(e);
            }
        }

        for (auto v : vset)
            remove_vertex(v, _b[v],
                          [&](auto& e) { return eset.find(e) != eset.end(); });

        for (auto& e : eset)
        {
            vertex_t v = source(e, _g);
            vertex_t u = target(e, _g);
            vertex_t r = _b[v];
            vertex_t s = _b[u];

            auto me = _emat.get_me(r, s);

            auto ew = _eweight[e];
            _mrs[me] -= ew;

            assert(_mrs[me] >= 0);

            _mrp[r] -= ew;
            _mrm[s] -= ew;

            for (size_t i = 0; i < _rec_types.size(); ++i)
            {
                switch (_rec_types[i])
                {
                case weight_type::REAL_NORMAL: // signed weights
                    _bdrec[i][me] -= _drec[i][e];
                    [[gnu::fallthrough]];
                default:
                    _brec[i][me] -= _rec[i][e];
                }
            }

            if (_mrs[me] == 0)
            {
                _emat.remove_me(me, _bg);
                if (_coupled_state != nullptr)
                    _coupled_state->remove_edge(me);
                else
                    boost::remove_edge(me, _bg);
            }
        }
    }

    void remove_vertices(python::object ovs)
    {
        multi_array_ref<uint64_t, 1> vs = get_array<uint64_t, 1>(ovs);
        remove_vertices(vs);
    }

    template <class EFilt>
    void add_vertex(size_t v, size_t r, EFilt&& efilt)
    {
        modify_vertex<true>(v, r, std::forward<EFilt>(efilt));
    }

    void add_vertex(size_t v, size_t r)
    {
        add_vertex(v, r, [](auto&){ return false; });
    }

    template <class Vlist, class Blist>
    void add_vertices(Vlist& vs, Blist& rs)
    {
        if (vs.size() != rs.size())
            throw ValueException("vertex and group lists do not have the same size");

        typedef typename graph_traits<g_t>::vertex_descriptor vertex_t;

        gt_hash_map<vertex_t, size_t> vset;
        for (size_t i = 0; i < vs.size(); ++i)
            vset[vs[i]] = rs[i];

        typedef typename graph_traits<g_t>::edge_descriptor edges_t;

        gt_hash_set<edges_t> eset;
        for (auto vr : vset)
        {
            auto v = vr.first;
            for (auto e : all_edges_range(v, _g))
            {
                auto u = (source(e, _g) == v) ? target(e, _g) : source(e, _g);
                if (vset.find(u) != vset.end())
                    eset.insert(e);
            }
        }

        for (auto vr : vset)
            add_vertex(vr.first, vr.second,
                       [&](auto& e){ return eset.find(e) != eset.end(); });

        for (auto e : eset)
        {
            vertex_t v = source(e, _g);
            vertex_t u = target(e, _g);
            vertex_t r = vset[v];
            vertex_t s = vset[u];

            auto me = _emat.get_me(r, s);

            if (me == _emat.get_null_edge())
            {
                me = boost::add_edge(r, s, _bg).first;
                _emat.put_me(r, s, me);
                _c_mrs[me] = 0;
                for (size_t i = 0; i < _rec_types.size(); ++i)
                {
                    _c_brec[i][me] = 0;
                    _c_bdrec[i][me] = 0;
                }

                if (_coupled_state != nullptr)
                    _coupled_state->add_edge(me);
            }

            assert(me == _emat.get_me(r, s));

            auto ew = _eweight[e];

            _mrs[me] += ew;
            _mrp[r] += ew;
            _mrm[s] += ew;

            for (size_t i = 0; i < _rec_types.size(); ++i)
            {
                switch (_rec_types[i])
                {
                case weight_type::REAL_NORMAL: // signed weights
                    _bdrec[i][me] += _drec[i][e];
                    [[gnu::fallthrough]];
                default:
                    _brec[i][me] += _rec[i][e];
                }
            }
        }
    }

    void add_vertices(python::object ovs, python::object ors)
    {
        multi_array_ref<uint64_t, 1> vs = get_array<uint64_t, 1>(ovs);
        multi_array_ref<uint64_t, 1> rs = get_array<uint64_t, 1>(ors);
        add_vertices(vs, rs);
    }

    template <bool Add, bool Deplete=true>
    void modify_edge(size_t u, size_t v, GraphInterface::edge_t& e, int dm)
    {
        if (dm == 0)
            return;

        size_t r = _b[u];
        size_t s = _b[v];

        get_partition_stats(u).remove_vertex(u, r, _deg_corr, _g,
                                             _vweight, _eweight,
                                             _degs);
        if (u != v)
            get_partition_stats(v).remove_vertex(v, s, _deg_corr, _g,
                                                 _vweight, _eweight,
                                                 _degs);

        auto me = _emat.get_me(r, s);
        if constexpr (Add)
        {
            if (me == _emat.get_null_edge())
            {
                me = boost::add_edge(r, s, _bg).first;
                _emat.put_me(r, s, me);
                _c_mrs[me] = 0;
                for (size_t i = 0; i < _rec_types.size(); ++i)
                {
                    _c_brec[i][me] = 0;
                    _c_bdrec[i][me] = 0;
                }
            }

            if (_coupled_state == nullptr)
                _mrs[me] += dm;

            _mrp[r] += dm;
            _mrm[s] += dm;
        }
        else
        {
            assert(me != _emat.get_null_edge());
            if (_coupled_state == nullptr)
            {
                _mrs[me] -= dm;
                if (_mrs[me] == 0)
                {
                    _emat.remove_me(me, _bg);
                    boost::remove_edge(me, _bg);
                }
            }
            else
            {
                if (_mrs[me] == dm)
                    _emat.remove_me(me, _bg);
            }
            _mrp[r] -= dm;
            _mrm[s] -= dm;
        }

        modify_edge<Add, Deplete>(u, v, e, dm, _is_weighted);

        get_partition_stats(u).add_vertex(u, r, _deg_corr, _g,
                                          _vweight, _eweight,
                                          _degs);
        if (u != v)
            get_partition_stats(v).add_vertex(v, s, _deg_corr, _g,
                                              _vweight, _eweight,
                                              _degs);

        for (auto& ps : _partition_stats)
            ps.change_E(Add ? dm : -dm);

        if (_coupled_state != nullptr)
        {
            if constexpr (Add)
                _coupled_state->add_edge(r, s, me, dm);
            else
                _coupled_state->remove_edge(r, s, me, dm);
        }

        clear_egroups();
    }

    template <bool Add, bool Deplete>
    void modify_edge(size_t u, size_t v, GraphInterface::edge_t& e, int,
                     std::false_type)
    {
        if constexpr (Add)
        {
            e = boost::add_edge(u, v, _g).first;
            _E++;
        }
        else
        {
            boost::remove_edge(e, _g);
            e = GraphInterface::edge_t();
            _E--;
        }
    }

    template <bool Add, bool Deplete>
    void modify_edge(size_t u, size_t v, GraphInterface::edge_t& e, int dm,
                     std::true_type)
    {
        if constexpr (Add)
        {
            if (e == GraphInterface::edge_t())
            {
                e = boost::add_edge(u, v, _g).first;
                auto c_eweight = _eweight.get_checked();
                c_eweight[e] = dm;
            }
            else
            {
                _eweight[e] += dm;
            }

            get<1>(_degs[u]) += dm;
            if constexpr (is_directed_::apply<g_t>::type::value)
                get<0>(_degs[v]) += dm;
            else
                get<1>(_degs[v]) += dm;

            _E += dm;
        }
        else
        {
            _eweight[e] -= dm;
            if (_eweight[e] == 0 && Deplete)
            {
                boost::remove_edge(e, _g);
                e = GraphInterface::edge_t();
            }

            get<1>(_degs[u]) -= dm;
            if constexpr (is_directed_::apply<g_t>::type::value)
                get<0>(_degs[v]) -= dm;
            else
                get<1>(_degs[v]) -= dm;

            _E -= dm;
        }
    }

    void add_edge(size_t u, size_t v, GraphInterface::edge_t& e, int dm)
    {
        modify_edge<true, true>(u, v, e, dm);
    }

    void remove_edge(size_t u, size_t v, GraphInterface::edge_t& e, int dm)
    {
        modify_edge<false, true>(u, v, e, dm);
    }

    void set_vertex_weight(size_t v, int w)
    {
        set_vertex_weight(v, w, _vweight);
    }

    void set_vertex_weight(size_t, int, vcmap_t&)
    {
        throw ValueException("Cannot set the weight of an unweighted state");
    }

    template <class VMap>
    void set_vertex_weight(size_t v, int w, VMap&& vweight)
    {
        _N -= vweight[v];
        vweight[v] = w;
        _N += w;
    }

    void init_vertex_weight(size_t v)
    {
        init_vertex_weight(v, _vweight);
    }

    void init_vertex_weight(size_t, vcmap_t&)
    {
    }

    template <class VMap>
    void init_vertex_weight(size_t v, VMap&& vweight)
    {
        vweight.resize(num_vertices(_g));
        vweight[v] = 0;
    }

    template <class Vec>
    void move_vertices(Vec& v, Vec& nr)
    {
        for (size_t i = 0; i < std::min(v.size(), nr.size()); ++i)
            move_vertex(v[i], nr[i]);
    }

    void move_vertices(python::object ovs, python::object ors)
    {
        multi_array_ref<uint64_t, 1> vs = get_array<uint64_t, 1>(ovs);
        multi_array_ref<uint64_t, 1> rs = get_array<uint64_t, 1>(ors);
        if (vs.size() != rs.size())
            throw ValueException("vertex and group lists do not have the same size");
        move_vertices(vs, rs);
    }

    template <class VMap>
    void set_partition(VMap&& b)
    {
        vmap_t::unchecked_t hb;
        if (_coupled_state != nullptr)
            hb = _coupled_state->get_b();

        for (auto v : vertices_range(_g))
        {
            size_t r = b[v];
            while (r >= num_vertices(_bg))
                add_block();
            if (_wr[r] == 0)
            {
                if (_coupled_state != nullptr)
                    hb[r] = hb[_b[v]];
                _bclabel[r] = _bclabel[_b[v]];
            }
            move_vertex(v, r);
        }
    }

    void set_partition(boost::any& ab)
    {
        vmap_t& b = boost::any_cast<vmap_t&>(ab);
        set_partition<typename vmap_t::unchecked_t>(b.get_unchecked());
    }

    size_t virtual_remove_size(size_t v)
    {
        return _wr[_b[v]] - _vweight[v];
    }


    template <class EMap, class Edge, class Val>
    void set_prop(EMap& ec, Edge& e, Val&& val)
    {
        ec[e] = val;
    }

    template <class Edge, class Val>
    void set_prop(UnityPropertyMap<typename std::remove_reference<Val>::type, Edge>&,
                  Edge&, Val&&)
    {
    }

    size_t add_block(size_t n = 1)
    {
        _wr.resize(num_vertices(_bg) + n);
        _mrm.resize(num_vertices(_bg) + n);
        _mrp.resize(num_vertices(_bg) + n);
        _bclabel.resize(num_vertices(_bg) + n);
        _brecsum.resize(num_vertices(_bg) + n);
        size_t r = null_group;
        for (size_t i = 0; i < n; ++i)
        {
            r = boost::add_vertex(_bg);
            _wr[r] = _mrm[r] = _mrp[r] = 0;
            _empty_groups.insert(r);
            for (auto& p : _partition_stats)
                p.add_block();
            if (_egroups)
                _egroups->add_block();
            if (_coupled_state != nullptr)
                _coupled_state->coupled_resize_vertex(r);
        }
        _emat.add_block(_bg);
        return r;
    }

    void coupled_resize_vertex(size_t v)
    {
        _b.resize(num_vertices(_g));
        _bfield.resize(num_vertices(_g));
        init_vertex_weight(v);
        _pclabel.resize(num_vertices(_g));
        resize_degs(_degs);
    }

    void resize_degs(const simple_degs_t&) {}

    void resize_degs(typename degs_map_t::unchecked_t& degs)
    {
        degs.resize(num_vertices(_g));
    }

    // =========================================================================
    // Virtual state modification
    // =========================================================================

    // compute the entropy difference of a virtual move of vertex from block r
    // to nr
    template <bool exact, class MEntries>
    double virtual_move_sparse(size_t v, size_t r, size_t nr,
                               MEntries& m_entries)
    {
        if (r == nr)
            return 0.;

        double dS = entries_dS<exact>(m_entries, _mrs, _emat, _bg);

        auto [kin, kout] = get_deg(v, _eweight, _degs, _g);

        int dwr = _vweight[v];
        int dwnr = dwr;

        if (r == null_group && dwnr == 0)
            dwnr = 1;

        auto vt = [&](auto mrp, auto mrm, auto nr)
            {
                assert(mrp >= 0 && mrm >=0 && nr >= 0);
                if constexpr (exact)
                    return vterm_exact(mrp, mrm, nr, _deg_corr, _bg);
                else
                    return vterm(mrp, mrm, nr, _deg_corr, _bg);
            };

        if (r != null_group)
        {
            auto mrp_r = _mrp[r];
            auto mrm_r = _mrm[r];
            auto wr_r = _wr[r];
            dS += vt(mrp_r - kout, mrm_r - kin, wr_r - dwr);
            dS -= vt(mrp_r       , mrm_r      , wr_r      );
        }

        if (nr != null_group)
        {
            auto mrp_nr = _mrp[nr];
            auto mrm_nr = _mrm[nr];
            auto wr_nr = _wr[nr];
            dS += vt(mrp_nr + kout, mrm_nr + kin, wr_nr + dwnr);
            dS -= vt(mrp_nr       , mrm_nr      , wr_nr       );
        }

        return dS;
    }

    template <bool exact>
    double virtual_move_sparse(size_t v, size_t r, size_t nr)
    {
        return virtual_move_sparse<exact>(v, r, nr);
    }

    double virtual_move_dense(size_t v, size_t r, size_t nr, bool multigraph)
    {
        if (_deg_corr)
            throw GraphException("Dense entropy for degree corrected model not implemented!");

        typedef typename graph_traits<g_t>::vertex_descriptor vertex_t;

        if (r == nr)
            return 0;

        vector<int> deltap(num_vertices(_bg), 0);
        int deltal = 0;
        for (auto e : out_edges_range(v, _g))
        {
            vertex_t u = target(e, _g);
            vertex_t s = _b[u];
            if (u == v)
                deltal += _eweight[e];
            else
                deltap[s] += _eweight[e];
        }
        if constexpr (!is_directed_::apply<g_t>::type::value)
            deltal /= 2;

        vector<int> deltam(num_vertices(_bg), 0);
        if (is_directed_::apply<g_t>::type::value)
        {
            for (auto e : in_edges_range(v, _g))
            {
                vertex_t u = source(e, _g);
                if (u == v)
                    continue;
                vertex_t s = _b[u];
                deltam[s] += _eweight[e];
            }
        }

        double dS = 0;
        int dwr = _vweight[v];
        int dwnr = dwr;

        if (r == null_group && dwnr == 0)
            dwnr = 1;

        if (nr == null_group)
        {
            std::fill(deltap.begin(), deltap.end(), 0);
            std::fill(deltam.begin(), deltam.end(), 0);
            if (dwr != _wr[r])
                deltal = 0;
        }

        double Si = 0, Sf = 0;
        for (vertex_t s = 0; s < num_vertices(_bg); ++s)
        {
            if (_wr[s] == 0 && s != r && s != nr)
                continue;

            int ers = (r != null_group) ? get_beprop(r, s, _mrs, _emat) : 0;
            int enrs = (nr != null_group) ? get_beprop(nr, s, _mrs, _emat) : 0;

            if (!is_directed_::apply<g_t>::type::value)
            {
                if (s != nr && s != r)
                {
                    if (r != null_group)
                    {
                        Si += eterm_dense(r,  s, ers,              _wr[r],         _wr[s], multigraph, _bg);
                        Sf += eterm_dense(r,  s, ers - deltap[s],  _wr[r] - dwr,   _wr[s], multigraph, _bg);
                    }

                    if (nr != null_group)
                    {
                        Si += eterm_dense(nr, s, enrs,             _wr[nr],        _wr[s], multigraph, _bg);
                        Sf += eterm_dense(nr, s, enrs + deltap[s], _wr[nr] + dwnr, _wr[s], multigraph, _bg);
                    }
                }

                if (s == r)
                {
                    Si += eterm_dense(r, r, ers,                      _wr[r],       _wr[r],       multigraph, _bg);
                    Sf += eterm_dense(r, r, ers - deltap[r] - deltal, _wr[r] - dwr, _wr[r] - dwr, multigraph, _bg);
                }

                if (s == nr)
                {
                    Si += eterm_dense(nr, nr, enrs,                       _wr[nr],        _wr[nr],        multigraph, _bg);
                    Sf += eterm_dense(nr, nr, enrs + deltap[nr] + deltal, _wr[nr] + dwnr, _wr[nr] + dwnr, multigraph, _bg);

                    if (r != null_group)
                    {
                        Si += eterm_dense(r, nr, ers,                          _wr[r],       _wr[nr],        multigraph, _bg);
                        Sf += eterm_dense(r, nr, ers - deltap[nr] + deltap[r], _wr[r] - dwr, _wr[nr] + dwnr, multigraph, _bg);
                    }
                }
            }
            else
            {
                int esr = (r != null_group) ? get_beprop(s, r, _mrs, _emat) : 0;
                int esnr  = (nr != null_group) ? get_beprop(s, nr, _mrs, _emat) : 0;

                if (s != nr && s != r)
                {
                    if (r != null_group)
                    {
                        Si += eterm_dense(r, s, ers            , _wr[r]      , _wr[s]      , multigraph, _bg);
                        Sf += eterm_dense(r, s, ers - deltap[s], _wr[r] - dwr, _wr[s]      , multigraph, _bg);
                        Si += eterm_dense(s, r, esr            , _wr[s]      , _wr[r]      , multigraph, _bg);
                        Sf += eterm_dense(s, r, esr - deltam[s], _wr[s]      , _wr[r] - dwr, multigraph, _bg);
                    }

                    if (nr != null_group)
                    {
                        Si += eterm_dense(nr, s, enrs            , _wr[nr]       , _wr[s]        , multigraph, _bg);
                        Sf += eterm_dense(nr, s, enrs + deltap[s], _wr[nr] + dwnr, _wr[s]        , multigraph, _bg);
                        Si += eterm_dense(s, nr, esnr            , _wr[s]        , _wr[nr]       , multigraph, _bg);
                        Sf += eterm_dense(s, nr, esnr + deltam[s], _wr[s]        , _wr[nr] + dwnr, multigraph, _bg);
                    }
                }

                if(s == r)
                {
                    Si += eterm_dense(r, r, ers                                  , _wr[r]      , _wr[r]      , multigraph, _bg);
                    Sf += eterm_dense(r, r, ers - deltap[r]  - deltam[r] - deltal, _wr[r] - dwr, _wr[r] - dwr, multigraph, _bg);

                    if (nr != null_group)
                    {
                        Si += eterm_dense(r, nr, esnr                         , _wr[r]      , _wr[nr]       , multigraph, _bg);
                        Sf += eterm_dense(r, nr, esnr - deltap[nr] + deltam[r], _wr[r] - dwr, _wr[nr] + dwnr, multigraph, _bg);
                    }
                }

                if(s == nr)
                {
                    Si += eterm_dense(nr, nr, esnr                                   , _wr[nr]       , _wr[nr]       , multigraph, _bg);
                    Sf += eterm_dense(nr, nr, esnr + deltap[nr] + deltam[nr] + deltal, _wr[nr] + dwnr, _wr[nr] + dwnr, multigraph, _bg);

                    if (r != null_group)
                    {
                        Si += eterm_dense(nr, r, esr                         , _wr[nr]       , _wr[r]      , multigraph, _bg);
                        Sf += eterm_dense(nr, r, esr + deltap[r] - deltam[nr], _wr[nr] + dwnr, _wr[r] - dwr, multigraph, _bg);
                    }
                }
            }
        }

        return Sf - Si + dS;
    }


    template <class MEntries>
    [[gnu::hot]]
    double virtual_move(size_t v, size_t r, size_t nr, const entropy_args_t& ea,
                        MEntries& m_entries)
    {
        assert(size_t(_b[v]) == r || r == null_group);

        if (r == nr || _vweight[v] == 0)
            return 0;

        if (r != null_group && nr != null_group && !allow_move(r, nr))
            return std::numeric_limits<double>::infinity();

        get_move_entries(v, r, nr, m_entries, [](auto) constexpr { return false; });

        double dS = 0;
        if (ea.adjacency)
        {
            if (ea.dense)
            {
                dS = virtual_move_dense(v, r, nr, ea.multigraph);
            }
            else
            {
                if (ea.exact)
                    dS = virtual_move_sparse<true>(v, r, nr, m_entries);
                else
                    dS = virtual_move_sparse<false>(v, r, nr, m_entries);
            }
        }

        double dS_dl = 0;

        dS_dl += get_delta_partition_dl(v, r, nr, ea);

        if (ea.degree_dl || ea.edges_dl)
        {
            auto& ps = get_partition_stats(v);
            if (_deg_corr && ea.degree_dl)
                dS_dl += ps.get_delta_deg_dl(v, r, nr, _vweight, _eweight,
                                             _degs, _g, ea.degree_dl_kind);
            if (ea.edges_dl)
            {
                size_t actual_B = 0;
                for (auto& ps : _partition_stats)
                    actual_B += ps.get_actual_B();
                dS_dl += ps.get_delta_edges_dl(v, r, nr, _vweight, actual_B,
                                               _g);
            }
        }

        if (!_Bfield.empty() && ea.Bfield)
        {
            int dB = 0;
            if (virtual_remove_size(v) == 0)
                dB--;
            if (_wr[nr] == 0)
                dB++;
            if (dB != 0)
            {
                size_t actual_B = 0;
                for (auto& ps : _partition_stats)
                    actual_B += ps.get_actual_B();
                dS_dl += (actual_B < _Bfield.size()) ?
                    _Bfield[actual_B] : _Bfield.back();
                actual_B += dB;
                dS_dl -= (actual_B < _Bfield.size()) ?
                    _Bfield[actual_B] : _Bfield.back();
            }
        }

        int dL = 0;
        std::vector<double> LdBdx;
        if (ea.recs && _rt != weight_type::NONE)
        {
            LdBdx.resize(_rec_types.size(), 0);
            auto rdS = rec_entries_dS(*this, m_entries, ea, LdBdx, dL);
            dS += get<0>(rdS);
            dS_dl += get<1>(rdS);
        }

        if (_coupled_state != nullptr && _vweight[v] > 0)
        {
            m_entries._p_entries.clear();

            if (_rt == weight_type::NONE)
            {
                std::vector<double> dummy;
                entries_op(m_entries, _emat,
                           [&](auto t, auto u, auto& me, auto delta)
                           {
                               if (delta == 0)
                                   return;
                               m_entries._p_entries.emplace_back(t, u, me,
                                                                 delta, dummy);
                           });
            }
            else
            {
                wentries_op(m_entries, _emat,
                            [&](auto t, auto u, auto& me, auto delta, auto& edelta)
                            {
                                m_entries._p_entries.emplace_back(t, u, me,
                                                                  delta,
                                                                  get<0>(edelta));
                            });
            }

            int dr = (_wr[r] == _vweight[v] && _vweight[v] > 0) ? -1 : 0;
            int dnr = (_wr[nr] == 0 && _vweight[v] > 0) ? 1 : 0;
            if (!m_entries._p_entries.empty() || dr != 0 || dnr != 0)
            {
                dS_dl += _coupled_state->propagate_entries_dS(r, nr, dr, dnr,
                                                              m_entries._p_entries,
                                                              _coupled_entropy_args,
                                                              LdBdx, dL);
            }
        }
        return dS + ea.beta_dl * dS_dl;
    }

    double propagate_entries_dS(size_t u, size_t v, int du, int dv,
                                std::vector<std::tuple<size_t, size_t, GraphInterface::edge_t, int,
                                                       std::vector<double>>>& entries,
                                const entropy_args_t& ea, std::vector<double>& dBdx,
                                int dL)
    {
        size_t r = _b[u];
        size_t s = _b[v];

        if (u == v)
        {
            if (ea.recs && _rt == weight_type::REAL_NORMAL)
            {
                double dS;
                #pragma omp critical (propagate_entries_dS)
                {
                    _m_entries.set_move(r, s, num_vertices(_bg));
                    auto rdS = rec_entries_dS(*this, _m_entries, ea, dBdx, dL);
                    dS = get<0>(rdS) + get<1>(rdS);
                }
                entries.clear();
                if (_coupled_state != nullptr)
                    dS += _coupled_state->propagate_entries_dS(r, s, 0, 0,
                                                               entries,
                                                               _coupled_entropy_args,
                                                               dBdx, dL);
                return dS;
            }
            return 0.;
        }

        double dS = 0;

        #pragma omp critical (propagate_entries_dS)
        {
            _m_entries.set_move(r, s, num_vertices(_bg));

            auto comp =
                [&](auto&... dummy)
                {
                    if (du != 0)
                    {
                        for (auto t : out_neighbors_range(r, _bg))
                            _m_entries.template insert_delta<true>(r, t, 0, dummy...);
                        for (auto t : in_neighbors_range(r, _bg))
                            _m_entries.template insert_delta<true>(t, r, 0, dummy...);
                    }

                    if (dv != 0)
                    {
                        for (auto t : out_neighbors_range(s, _bg))
                            _m_entries.template insert_delta<true>(s, t, 0, dummy...);
                        for (auto t : in_neighbors_range(s, _bg))
                            _m_entries.template insert_delta<true>(t, s, 0, dummy...);
                    }
                };

            std::vector<double> dummy;
            if (!ea.recs || _rt == weight_type::NONE)
            {
                for (auto& iter : entries)
                    _m_entries.template insert_delta<true>(_b[get<0>(iter)],
                                                           _b[get<1>(iter)],
                                                           get<3>(iter));
                comp();
            }
            else
            {
                for (auto& iter : entries)
                    recs_propagate_insert(*this, _b[get<0>(iter)], _b[get<1>(iter)],
                                          get<2>(iter), get<3>(iter), get<4>(iter),
                                          _m_entries);
                dummy.resize(_rec.size(), 0.);
                comp(dummy, dummy);
            }

            entries.clear();

            auto e_diff =
                [&](auto rr, auto ss, auto& me, auto d)
                {
                    int ers = 0;
                    if (me != _emat.get_null_edge())
                        ers = this->_mrs[me];
                    auto wr = this->_wr[rr];
                    auto ws = this->_wr[ss];

                    dS -= eterm_dense(rr, ss, ers, wr, ws, true, _bg);

                    if (rr == r)
                        wr += du;
                    if (rr == s)
                        wr += dv;

                    if (ss == r)
                        ws += du;
                    if (ss == s)
                        ws += dv;

                    dS += eterm_dense(rr, ss, ers + d, wr, ws, true, _bg);
                };

            if (!ea.recs || _rt == weight_type::NONE)
            {
                if (ea.adjacency)
                {
                    entries_op(_m_entries, _emat,
                               [&](auto rr, auto ss, auto& me, auto d)
                               {
                                   e_diff(rr, ss, me, d);
                                   if (d == 0)
                                       return;
                                   entries.emplace_back(rr, ss, me, d, dummy);
                               });
                }
                else
                {
                    entries_op(_m_entries, _emat,
                               [&](auto rr, auto ss, auto& me, auto d)
                               {
                                   if (d == 0)
                                       return;
                                   entries.emplace_back(rr, ss, me, d, dummy);
                               });
                }
            }
            else
            {
                if (ea.adjacency)
                {
                    wentries_op(_m_entries, _emat,
                                [&](auto rr, auto ss, auto& me, auto d, auto& ed)
                                {
                                    e_diff(rr, ss, me, d);
                                    entries.emplace_back(rr, ss, me, d, get<0>(ed));
                                });
                }
                else
                {
                    wentries_op(_m_entries, _emat,
                                [&](auto rr, auto ss, auto& me, auto d, auto& ed)
                                {
                                    entries.emplace_back(rr, ss, me, d, get<0>(ed));
                                });
                }

                auto rdS = rec_entries_dS(*this, _m_entries, ea, dBdx, dL);
                dS += get<0>(rdS) + get<1>(rdS);
            }
        }

        int dr = (_wr[r] + du == 0) ? -1 : 0;
        int ds = (_wr[s] == 0) ? 1 : 0;
        if (_coupled_state != nullptr)
        {
            dS += _coupled_state->propagate_entries_dS(r, s, dr, ds,
                                                       entries,
                                                       _coupled_entropy_args,
                                                       dBdx, dL);
        }
        else
        {
            if (r != s && dr + ds != 0 && ea.edges_dl)
            {
                size_t actual_B = 0;
                for (auto& ps : _partition_stats)
                    actual_B += ps.get_actual_B();
                dS -= get_edges_dl(actual_B, _E, _g);
                dS += get_edges_dl(actual_B + dr + ds, _E, _g);
            }
        }
        return dS;
    }

    [[gnu::hot]]
    double virtual_move(size_t v, size_t r, size_t nr, const entropy_args_t& ea)
    {
        return virtual_move(v, r, nr, ea, _m_entries);
    }

    double get_delta_partition_dl(size_t v, size_t r, size_t nr,
                                  const entropy_args_t& ea)
    {
        if (r == nr)
            return 0;

        double dS = 0;

        auto& f = _bfield[v];
        if (!f.empty())
        {
            if (nr != null_group)
                dS -= (nr < f.size()) ? f[nr] : f.back();
            if (r != null_group)
                dS += (r < f.size()) ? f[r] : f.back();
        }

        if (ea.partition_dl)
        {
            auto& ps = get_partition_stats(v);
            dS += ps.get_delta_partition_dl(v, r, nr, _vweight);
        }

        if (_coupled_state != nullptr)
        {
            bool r_vacate = (r != null_group && _wr[r] == _vweight[v]);
            bool nr_occupy = (nr != null_group && _wr[nr] == 0);

            auto& bh = _coupled_state->get_b();
            if (r_vacate && nr_occupy)
            {
                dS += _coupled_state->get_delta_partition_dl(r, bh[r], bh[nr],
                                                             _coupled_entropy_args);
            }
            else
            {
                if (r_vacate)
                    dS += _coupled_state->get_delta_partition_dl(r, bh[r], null_group,
                                                                 _coupled_entropy_args);
                if (nr_occupy)
                    dS += _coupled_state->get_delta_partition_dl(nr, null_group, bh[nr],
                                                                 _coupled_entropy_args);
            }
        }
        return dS;
    }

    // =========================================================================
    // Move proposals
    // =========================================================================

    size_t get_empty_block(size_t v, bool force_add = false)
    {
        if (_empty_groups.empty() || force_add)
        {
            size_t s = add_block();
            auto r = _b[v];
            _bclabel[s] = _bclabel[r];
            if (_coupled_state != nullptr)
            {
                auto& hb = _coupled_state->get_b();
                hb[s] = hb[r];
                auto& hpclabel = _coupled_state->get_pclabel();
                hpclabel[s] = _pclabel[v];
            }
            return s;
        }
        return *(_empty_groups.end()-1);
    }

    void sample_branch(size_t v, size_t u, rng_t& rng)
    {
        size_t s;
        auto r = _b[u];

        std::bernoulli_distribution new_r(1./(_candidate_groups.size()+1));
        if (_candidate_groups.size() < num_vertices(_g) && new_r(rng))
        {
            get_empty_block(v);
            s = uniform_sample(_empty_groups, rng);
            if (_coupled_state != nullptr)
            {
                _coupled_state->sample_branch(s, r, rng);
                auto& hpclabel = _coupled_state->get_pclabel();
                hpclabel[s] = _pclabel[u];
            }
            _bclabel[s] = _bclabel[r];
        }
        else
        {
            s = uniform_sample(_candidate_groups, rng);
        }
        _b[v] = s;
    }

    void copy_branch(size_t r, BlockStateVirtualBase& state)
    {
        if (r >= num_vertices(_bg))
            add_block(r - num_vertices(_bg) + 1);

        _bclabel[r] = state.get_bclabel()[r];

        if (_coupled_state != nullptr)
        {
            auto& cstate = *state.get_coupled_state();
            auto& sbh = cstate.get_b();

            auto s = sbh[r];

            _coupled_state->copy_branch(s, cstate);

            auto& bh = _coupled_state->get_b();
            bh[r] = s;
            auto& pclabel = cstate.get_pclabel();
            auto& hpclabel = _coupled_state->get_pclabel();
            hpclabel[r] = pclabel[r];
        }
    }

    // Sample node placement
    size_t sample_block(size_t v, double c, double d, rng_t& rng)
    {
        size_t B = _candidate_groups.size();

        // attempt new block
        std::bernoulli_distribution new_r(d);
        if (d > 0 && B < _N && new_r(rng))
        {
            get_empty_block(v);
            auto s = uniform_sample(_empty_groups, rng);
            auto r = _b[v];
            if (_coupled_state != nullptr)
            {
                _coupled_state->sample_branch(s, r, rng);
                auto& hpclabel = _coupled_state->get_pclabel();
                hpclabel[s] = _pclabel[v];
            }
            _bclabel[s] = _bclabel[r];
            return s;
        }

        size_t s;
        if (!std::isinf(c) && total_degreeS()(v, _g) > 0)
        {
            auto u = graph_tool::random_neighbor(v, _g, rng);
            size_t t = _b[u];
            double p_rand = 0;
            if (c > 0)
            {
                if (is_directed_::apply<g_t>::type::value)
                    p_rand = c * B / double(_mrp[t] + _mrm[t] + c * B);
                else
                    p_rand = c * B / double(_mrp[t] + c * B);
            }

            std::bernoulli_distribution rand(p_rand);
            if (c == 0 || !rand(rng))
            {
                if (!_egroups)
                    init_egroups();
                s = _egroups->sample_edge(t, rng);
            }
            else
            {
                s = uniform_sample(_candidate_groups, rng);
            }
        }
        else
        {
            s = uniform_sample(_candidate_groups, rng);
        }

        return s;
    }

    size_t random_neighbor(size_t v, rng_t& rng)
    {
        if (total_degreeS()(v, _g) == 0)
            return v;
        return graph_tool::random_neighbor(v, _g, rng);
    }

    size_t sample_block_local(size_t v, rng_t& rng)
    {
        if (total_degreeS()(v, _g) > 0)
        {
            auto u = graph_tool::random_neighbor(v, _g, rng);
            auto w = graph_tool::random_neighbor(u, _g, rng);
            return _b[w];
        }
        else
        {
            return uniform_sample(_candidate_groups, rng);
        }
    }

    // Computes the move proposal probability
    template <class MEntries>
    double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                         bool reverse, MEntries& m_entries)
    {
        size_t B = _candidate_groups.size();

        if (r == s)
            reverse = false;

        if (reverse)
        {
            if (_wr[s] == _vweight[v])
                return log(d);

            if (_wr[r] == 0)
                B++;
        }
        else
        {
            if (_wr[s] == 0)
                return log(d);
        }

        if (B == _N)
            d = 0;

        if (std::isinf(c))
            return log(1. - d) - safelog_fast(B);

        double p = 0;
        size_t w = 0;

        size_t kin, kout;
        std::tie(kin, kout) = get_deg(v, _eweight, _degs, _g);

        m_entries.get_mes(_emat);

        auto sum_prob = [&](auto&& range)
            {
                for (auto u : range)
                {
                    size_t t = _b[u];
                    if (u == v)
                        t = r;
                    w++;

                    int mts = 0;
                    const auto& me = m_entries.get_me(t, s, _emat);
                    if (me != _emat.get_null_edge())
                        mts = _mrs[me];
                    int mtp = _mrp[t];
                    int mst = mts;
                    int mtm = mtp;

                    if constexpr (is_directed_::apply<g_t>::type::value)
                    {
                        mst = 0;
                        const auto& me = m_entries.get_me(s, t, _emat);
                        if (me != _emat.get_null_edge())
                            mst = _mrs[me];
                        mtm = _mrm[t];
                    }

                    if (reverse)
                    {
                        int dts = m_entries.get_delta(t, s);
                        int dst = dts;
                        if constexpr (is_directed_::apply<g_t>::type::value)
                            dst = m_entries.get_delta(s, t);

                        mts += dts;
                        mst += dst;

                        if (t == s)
                        {
                            mtp -= kout;
                            mtm -= kin;
                        }

                        if (t == r)
                        {
                            mtp += kout;
                            mtm += kin;
                        }
                    }

                    if constexpr (is_directed_::apply<g_t>::type::value)
                    {
                        p += (mts + mst + c) / (mtp + mtm + c * B);
                    }
                    else
                    {
                        if (t == s)
                            mts *= 2;
                        p += (mts + c) / (mtp + c * B);
                    }
                }
            };

        sum_prob(out_neighbors_range(v, _g));

        if constexpr (is_directed_::apply<g_t>::type::value)
            sum_prob(in_neighbors_range(v, _g));

        if (w > 0)
            return log(1. - d) + log(p) - log(w);
        else
            return log(1. - d) - safelog_fast(B);
    }

    double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                         bool reverse,
                         std::vector<std::tuple<size_t, size_t, int>>& p_entries)
    {
        _m_entries.set_move(r, s, num_vertices(_bg));
        for (auto& rsd : p_entries)
            _m_entries.template insert_delta<true>(get<0>(rsd), get<1>(rsd),
                                                   get<2>(rsd));
        return get_move_prob(v, r, s, c, d, reverse);
    }

    double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                         bool reverse)
    {
        get_move_entries(v, _b[v], (reverse) ? r : s, _m_entries);
        return get_move_prob(v, r, s, c, d, reverse, _m_entries);
    }

    bool is_last(size_t v)
    {
        return (_vweight[v] > 0) && (_wr[_b[v]] == _vweight[v]);
    }

    size_t node_weight(size_t v)
    {
        return _vweight[v];
    }

    // =========================================================================
    // Entropy computation
    // =========================================================================

    double get_deg_entropy(size_t v, const simple_degs_t&,
                           const std::array<int,2>& delta = {0, 0})
    {
        auto kin = in_degreeS()(v, _g, _eweight) + delta[0];
        auto kout = out_degreeS()(v, _g, _eweight) + delta[1];
        double S = -lgamma_fast(kin + 1) - lgamma_fast(kout + 1);
        return S * _vweight[v];
    }

    double get_deg_entropy(size_t v,
                           const typename degs_map_t::unchecked_t& degs,
                           const std::array<int,2>& delta = {0, 0})
    {
        auto& ks = degs[v];
        auto kin = get<0>(ks) + delta[0];
        auto kout = get<1>(ks) + delta[1];
        double S = -lgamma_fast(kin + 1) - lgamma_fast(kout + 1);
        return S * _vweight[v];
    }

    double sparse_entropy(bool multigraph, bool deg_entropy, bool exact)
    {
        double S = 0;

        if (exact)
        {
            for (auto e : edges_range(_bg))
                S += eterm_exact(source(e, _bg), target(e, _bg), _mrs[e], _bg);
            for (auto v : vertices_range(_bg))
                S += vterm_exact(_mrp[v], _mrm[v], _wr[v], _deg_corr, _bg);
        }
        else
        {
            for (auto e : edges_range(_bg))
                S += eterm(source(e, _bg), target(e, _bg), _mrs[e], _bg);
            for (auto v : vertices_range(_bg))
                S += vterm(_mrp[v], _mrm[v], _wr[v], _deg_corr, _bg);
        }

        if (_deg_corr && deg_entropy)
        {
            for (auto v : vertices_range(_g))
                S += get_deg_entropy(v, _degs);
        }

        if (multigraph)
            S += get_parallel_entropy();

        return S;
    }

    double dense_entropy(bool multigraph)
    {
        if (_deg_corr)
            throw GraphException("Dense entropy for degree corrected model not implemented!");
        double S = 0;
        for (auto e : edges_range(_bg))
        {
            auto r = source(e, _bg);
            auto s = target(e, _bg);
            S += eterm_dense(r, s, _mrs[e], _wr[r], _wr[s], multigraph, _bg);
        }
        return S;
    }

    double entropy(const entropy_args_t& ea, bool propagate=false)
    {
        double S = 0, S_dl = 0;

        if (ea.adjacency)
        {
            if (!ea.dense)
                S = sparse_entropy(ea.multigraph, ea.deg_entropy, ea.exact);
            else
                S = dense_entropy(ea.multigraph);

            assert(!isnan(S) && !isnan(S_dl));

            if (!ea.dense && !ea.exact)
            {
                size_t E = 0;
                #pragma omp parallel reduction(+:E)
                parallel_edge_loop_no_spawn
                    (_g,
                     [&](const auto& e)
                     {
                         E += _eweight[e];
                     });
                if (ea.multigraph)
                    S -= E;
                else
                    S += E;
            }

            assert(!isnan(S) && !isnan(S_dl));
        }

        if (ea.partition_dl)
            S_dl += get_partition_dl();

        assert(!isnan(S) && !isnan(S_dl));

        if (_deg_corr && ea.degree_dl)
            S_dl += get_deg_dl(ea.degree_dl_kind);

        assert(!isnan(S) && !isnan(S_dl));

        if (ea.edges_dl)
        {
            size_t actual_B = 0;
            for (auto& ps : _partition_stats)
                actual_B += ps.get_actual_B();
            S_dl += get_edges_dl(actual_B, _E, _g);
        }

        assert(!isnan(S) && !isnan(S_dl));

        #pragma omp parallel reduction(+:S_dl)
        parallel_vertex_loop_no_spawn
            (_g,
             [&](auto v)
             {
                 auto& f = _bfield[v];
                 if (f.empty())
                     return;
                 size_t r = _b[v];
                 S_dl -= (r < f.size()) ? f[r] : f.back();
             });

        if (ea.recs)
        {
            auto rdS = rec_entropy(*this, ea);
            S += get<0>(rdS);
            S_dl += get<1>(rdS);
        }

        assert(!isnan(S) && !isnan(S_dl));

        if (!_Bfield.empty() && ea.Bfield)
        {
            size_t actual_B = 0;
            for (auto& ps : _partition_stats)
                actual_B += ps.get_actual_B();
            S_dl -= (actual_B < _Bfield.size()) ?
                _Bfield[actual_B] : _Bfield.back();
        }
        assert(!isnan(S) && !isnan(S_dl));

        if (_coupled_state != nullptr && propagate)
            S_dl += _coupled_state->entropy(_coupled_entropy_args, true);

        assert(!isnan(S) && !isnan(S_dl));

        return S + S_dl * ea.beta_dl;
    }

    double get_partition_dl()
    {
        double S = 0;
        for (auto& ps : _partition_stats)
            S += ps.get_partition_dl();
        return S;
    }

    double get_deg_dl(int kind)
    {
        double S = 0;
        for (auto& ps : _partition_stats)
            S += ps.get_deg_dl(kind);
        return S;
    }

    template <class Vs, class Skip>
    double get_parallel_entropy(Vs&& vs, Skip&& skip, int delta = 0)
    {
        double S = 0;
        for (auto v : vs)
        {
            gt_hash_map<decltype(v), size_t> us;
            for (auto e : out_edges_range(v, _g))
            {
                auto u = target(e, _g);
                if (skip(v, u))
                    continue;
                us[u] += _eweight[e];
            }

            for (auto& uc : us)
            {
                auto& u = uc.first;
                auto m = uc.second;
                if (u == v && !is_directed_::apply<g_t>::type::value)
                    m += 2 * delta;
                else
                    m += delta;
                if (m > 1)
                {
                    if (u == v && !is_directed_::apply<g_t>::type::value)
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

    double get_parallel_entropy()
    {
        return get_parallel_entropy(vertices_range(_g),
                                    [](auto u, auto v)
                                    { return (u < v &&
                                              !is_directed_::apply<g_t>::type::value);
                                    });
    }

    double modify_edge_dS(size_t u, size_t v, const GraphInterface::edge_t& e,
                          int dm, const entropy_args_t& ea)
    {
        if (dm == 0)
            return 0;

        double S = 0, S_dl = 0;
        size_t r = _b[u];
        size_t s = _b[v];

        if (ea.degree_dl && _deg_corr)
        {
            constexpr size_t null = numeric_limits<size_t>::max();

            if (r != s || u == v)
            {
                std::array<std::tuple<size_t, int>, 2> kins, kouts;

                auto [kin, kout] = get_deg(u, _eweight, _degs, _g);

                if constexpr (is_directed_::apply<g_t>::type::value)
                    kins = {make_tuple(kin, 0), make_tuple(null, 0)};
                kouts = {make_tuple(kout, 0), make_tuple(null, 0)};

                if (u != v)
                {
                    if constexpr (is_directed_::apply<g_t>::type::value)
                        kins[0] = {null, 0};
                    kouts[1] = {kout + dm, 0};
                }
                else
                {
                    if constexpr (!is_directed_::apply<g_t>::type::value)
                    {
                        kouts[1] = {kout + 2 * dm, 0};
                    }
                    else
                    {
                        kins[1] = {kin + dm, 0};
                        kouts[1] = {kout + dm, 0};
                    }
                }

                S_dl -= get_partition_stats(u).get_deg_dl(ea.degree_dl_kind,
                                                          std::array<size_t,1>({r}),
                                                          kins, kouts);
                if constexpr (is_directed_::apply<g_t>::type::value)
                    get<1>(kins[0]) -= 1;
                get<1>(kouts[0]) -= 1;
                if constexpr (is_directed_::apply<g_t>::type::value)
                    get<1>(kins[1]) += 1;
                get<1>(kouts[1]) += 1;

                S_dl += get_partition_stats(u).get_deg_dl(ea.degree_dl_kind,
                                                          std::array<size_t,1>({r}),
                                                          kins, kouts);

                if (u != v) // r != s
                {
                    auto [kin, kout] = get_deg(v, _eweight, _degs, _g);

                    if constexpr (is_directed_::apply<g_t>::type::value)
                        kins = {make_tuple(kin, 0), make_tuple(null, 0)};
                    kouts = {make_tuple(kout, 0), make_tuple(null, 0)};

                    if constexpr (!is_directed_::apply<g_t>::type::value)
                    {
                        kouts[1] = {kout + dm, 0};
                    }
                    else
                    {
                        kins[1] = {kin + dm, 0};
                        kouts[0] = {null, 0};
                    }

                    S_dl -= get_partition_stats(v).get_deg_dl(ea.degree_dl_kind,
                                                              std::array<size_t,1>({s}),
                                                              kins, kouts);
                    if constexpr (is_directed_::apply<g_t>::type::value)
                        get<1>(kins[0]) -= 1;
                    get<1>(kouts[0]) -= 1;
                    if constexpr (is_directed_::apply<g_t>::type::value)
                        get<1>(kins[1]) += 1;
                    get<1>(kouts[1]) += 1;

                    S_dl += get_partition_stats(v).get_deg_dl(ea.degree_dl_kind,
                                                              std::array<size_t,1>({s}),
                                                              kins, kouts);
                }
            }
            else // r == s && u != v
            {
                if constexpr (!is_directed_::apply<g_t>::type::value)
                {
                    std::array<std::tuple<size_t, int>, 0> kins;
                    std::array<std::tuple<size_t, int>, 4> kouts;

                    auto [kin, kout] = get_deg(u, _eweight, _degs, _g);

                    kouts = {make_tuple(kout, -1), make_tuple(null, 0),
                             make_tuple(null, 0), make_tuple(null, 0)};
                    kouts[1] = {kout + dm, 1};

                    std::tie(kin, kout) = get_deg(v, _eweight, _degs, _g);

                    kouts[2] = {kout, -1};

                    kouts[3] = {kout + dm, 1};

                    for (size_t i = 0; i < 2; ++i)
                    {
                        if (get<0>(kouts[2]) == get<0>(kouts[i]))
                        {
                            get<0>(kouts[2]) = null;
                            get<1>(kouts[i]) += get<1>(kouts[2]);
                        }

                        if (get<0>(kouts[3]) == get<0>(kouts[i]))
                        {
                            get<0>(kouts[3]) = null;
                            get<1>(kouts[i]) += get<1>(kouts[3]);
                        }
                    }

                    S_dl += get_partition_stats(u).get_deg_dl(ea.degree_dl_kind,
                                                              std::array<size_t,1>({r}),
                                                              kins, kouts);
                    for (size_t i = 0; i < 4; ++i)
                        get<1>(kouts[i]) = 0;

                    S_dl -= get_partition_stats(u).get_deg_dl(ea.degree_dl_kind,
                                                              std::array<size_t,1>({r}),
                                                              kins, kouts);
                }
                else
                {
                    std::array<std::tuple<size_t, int>, 2> kins;
                    std::array<std::tuple<size_t, int>, 2> kouts;

                    auto [kin, kout] = get_deg(u, _eweight, _degs, _g);

                    kouts = {make_tuple(kout, -1), make_tuple(kout + dm, 1)};

                    std::tie(kin, kout) = get_deg(v, _eweight, _degs, _g);

                    kins = {make_tuple(kin, -1), make_tuple(kin + dm, 1)};

                    S_dl += get_partition_stats(u).get_deg_dl(ea.degree_dl_kind,
                                                              std::array<size_t,1>({r}),
                                                              kins, kouts);
                    for (size_t i = 0; i < 2; ++i)
                        get<1>(kins[i]) = get<1>(kouts[i]) = 0;

                    S_dl -= get_partition_stats(u).get_deg_dl(ea.degree_dl_kind,
                                                              std::array<size_t,1>({r}),
                                                              kins, kouts);

                }
            }
        }

        auto& me = _emat.get_me(r, s);
        size_t mrs = 0;
        if (me != _emat.get_null_edge())
            mrs = _mrs[me];

        if (ea.adjacency)
        {
            if (ea.dense)
            {
                S -= eterm_dense(r, s, mrs, _wr[r], _wr[s], ea.multigraph, _bg);
                S += eterm_dense(r, s, mrs + dm, _wr[r], _wr[s], ea.multigraph, _bg);
            }
            else
            {
                if (ea.exact)
                {
                    S -= eterm_exact(r, s, mrs, _bg);
                    S += eterm_exact(r, s, mrs + dm, _bg);
                    if (s != r)
                    {
                        S -= vterm_exact(_mrp[r],      _mrm[r],      _wr[r], _deg_corr, _bg);
                        S += vterm_exact(_mrp[r] + dm, _mrm[r] + dm, _wr[r], _deg_corr, _bg);
                        S -= vterm_exact(_mrp[s],      _mrm[s],      _wr[s], _deg_corr, _bg);
                        S += vterm_exact(_mrp[s] + dm, _mrm[s] + dm, _wr[s], _deg_corr, _bg);
                    }
                    else
                    {
                        S -= vterm_exact(_mrp[r],          _mrm[r],          _wr[r], _deg_corr, _bg);
                        if constexpr (is_directed_::apply<g_t>::type::value)
                            S += vterm_exact(_mrp[r] + dm, _mrm[r] + dm, _wr[r], _deg_corr, _bg);
                        else
                            S += vterm_exact(_mrp[r] + 2 * dm, _mrm[r] + 2 * dm, _wr[r], _deg_corr, _bg);
                    }
                }
                else
                {
                    S -= eterm(r, s, mrs, _bg);
                    S += eterm(r, s, mrs + dm, _bg);
                    if (s != r)
                    {
                        S += vterm(_mrp[r],      _mrm[r],      _wr[r], _deg_corr, _bg);
                        S += vterm(_mrp[r] + dm, _mrm[r] + dm, _wr[r], _deg_corr, _bg);
                        S += vterm(_mrp[s],      _mrm[s],      _wr[s], _deg_corr, _bg);
                        S += vterm(_mrp[s] + dm, _mrm[s] + dm, _wr[s], _deg_corr, _bg);
                    }
                    else
                    {
                        S -= vterm(_mrp[r],          _mrm[r],          _wr[r], _deg_corr, _bg);
                        if constexpr (is_directed_::apply<g_t>::type::value)
                            S += vterm(_mrp[r] + dm, _mrm[r] + dm, _wr[r], _deg_corr, _bg);
                        else
                            S += vterm(_mrp[r] + 2 * dm, _mrm[r] + 2 * dm, _wr[r], _deg_corr, _bg);
                    }
                }

                if (ea.multigraph)
                {
                    if constexpr (is_weighted_t::value)
                    {
                        auto m = (e == GraphInterface::edge_t()) ? 0 : _eweight[e];
                        if (u == v && !is_directed_::apply<g_t>::type::value)
                        {
                            auto e_S = [&](auto m)
                            {
                                return lgamma_fast(m/2 + 1) + m * log(2) / 2;
                            };
                            S -= e_S(2 * m);
                            S += e_S(2 * (m + dm));
                        }
                        else
                        {
                            auto e_S = [&](auto m)
                            {
                                return lgamma_fast(m + 1);
                            };
                            S -= e_S(m);
                            S += e_S(m + dm);
                        };
                    }
                    else
                    {
                        S -= get_parallel_entropy(std::array<size_t, 1>({u}),
                                                  [&](auto, auto w){ return w != v; }, 0);
                        S += get_parallel_entropy(std::array<size_t, 1>({u}),
                                                  [&](auto, auto w){ return w != v; }, dm);
                    }
                }

                if (_deg_corr)
                {
                    if (u != v)
                    {
                        S -= get_deg_entropy(u, _degs);
                        S += get_deg_entropy(u, _degs, {0, dm});
                        S -= get_deg_entropy(v, _degs);
                        if constexpr (is_directed_::apply<g_t>::type::value)
                            S += get_deg_entropy(v, _degs, {dm, 0});
                        else
                            S += get_deg_entropy(v, _degs, {0, dm});
                    }
                    else
                    {
                        S -= get_deg_entropy(u, _degs);
                        if constexpr (is_directed_::apply<g_t>::type::value)
                            S += get_deg_entropy(u, _degs, {dm, dm});
                        else
                            S += get_deg_entropy(u, _degs, {0, 2 * dm});
                    }
                }
            }
        }

        if (_coupled_state != nullptr)
        {
            S_dl += _coupled_state->modify_edge_dS(r, s, me, dm,
                                                   _coupled_entropy_args);
        }
        else
        {
            if (ea.edges_dl)
            {
                size_t actual_B = 0;
                for (auto& psi : _partition_stats)
                    actual_B += psi.get_actual_B();
                auto& ps = get_partition_stats(u);
                S_dl -= ps.get_edges_dl(actual_B, _g);
                S_dl += ps.get_edges_dl(actual_B, _g, dm);
            }
        }

        return S + S_dl * ea.beta_dl;
    }

    void init_partition_stats()
    {
        reset_partition_stats();
        size_t B = num_vertices(_bg);

// Clang 8.0 fails to correctly recognize these as ForwardIterators,
// triggering a static_assert in std::max_element(). See #576.
#ifndef __clang__
        auto vi = std::max_element(
#else
        auto vi = boost::first_max_element(
#endif
            vertices(_g).first, vertices(_g).second,
            [&](auto u, auto v)
            { return (this->_pclabel[u] <
                      this->_pclabel[v]); });

        size_t C = _pclabel[*vi] + 1;

        vector<vector<size_t>> vcs(C);
        vector<size_t> rc(num_vertices(_bg));
        for (auto v : vertices_range(_g))
        {
            vcs[_pclabel[v]].push_back(v);
            rc[_b[v]] = _pclabel[v];
        }

        for (size_t c = 0; c < C; ++c)
            _partition_stats.emplace_back(_g, _b, vcs[c], _E, B,
                                          _vweight, _eweight, _degs);

        for (auto r : vertices_range(_bg))
            _partition_stats[rc[r]].get_r(r);
    }

    void reset_partition_stats()
    {
        _partition_stats.clear();
        _partition_stats.shrink_to_fit();
    }

    partition_stats_t& get_partition_stats(size_t v)
    {
        size_t r = _pclabel[v];
        if (r >= _partition_stats.size())
            init_partition_stats();
        return _partition_stats[r];
    }

    template <class MCMCState>
    void init_mcmc(MCMCState& state)
    {
        clear_egroups();
        auto c = state._c;
        if (!std::isinf(c))
            init_egroups();
    }

    void couple_state(BlockStateVirtualBase& s, const entropy_args_t& ea)
    {
        _coupled_state = &s;
        _coupled_entropy_args = ea;
    }

    void decouple_state()
    {
        _coupled_state = nullptr;
    }

    BlockStateVirtualBase* get_coupled_state()
    {
        return _coupled_state;
    }

    void init_egroups()
    {
        _egroups = std::make_shared<EGroups>(_bg, _mrs);
    }

    void clear_egroups()
    {
        _egroups.reset();
    }

    void sync_emat()
    {
        _emat.sync(_bg);
    }

    size_t get_B_E()
    {
        return _B_E;
    }

    size_t get_B_E_D()
    {
        return _B_E_D;
    }

    size_t get_N()
    {
        return _N;
    }

    size_t get_E()
    {
        return _E;
    }

    vprop_map_t<int32_t>::type::unchecked_t& get_b()
    {
        return _b;
    }

    vprop_map_t<int32_t>::type::unchecked_t& get_bclabel()
    {
        return _bclabel;
    }

    vprop_map_t<int32_t>::type::unchecked_t& get_pclabel()
    {
        return _pclabel;
    }

    bool check_edge_counts(bool emat=true)
    {
        gt_hash_map<std::pair<size_t, size_t>, size_t> mrs;
        for (auto e : edges_range(_g))
        {
            assert(std::max(source(e, _g),
                            target(e, _g)) < _b.get_storage().size());
            size_t r = _b[source(e, _g)];
            size_t s = _b[target(e, _g)];
            if (!is_directed_::apply<g_t>::type::value && s < r)
                std::swap(r, s);
            mrs[std::make_pair(r, s)] += _eweight[e];
        }

        for (auto& rs_m : mrs)
        {
            auto r = rs_m.first.first;
            auto s = rs_m.first.second;
            size_t m_rs = 0;
            typename graph_traits<bg_t>::edge_descriptor me;
            if (emat)
            {
                me = _emat.get_me(r, s);
                if (me != _emat.get_null_edge())
                    m_rs = _mrs[me];
            }
            else
            {
                auto ret = boost::edge(r, s, _bg);
                me = ret.first;
                if (ret.second)
                    m_rs = _mrs[me];
            }
            if (m_rs != rs_m.second)
            {
                assert(false);
                return false;
            }
        }

        for (auto me : edges_range(_bg))
        {
            auto r = source(me, _bg);
            auto s = target(me, _bg);
            if (!is_directed_::apply<g_t>::type::value && s < r)
                std::swap(r, s);
            auto m_rs = mrs[std::make_pair(r, s)];
            if (m_rs != size_t(_mrs[me]))
            {
                assert(false);
                return false;
            }
        }

        if (_coupled_state != nullptr)
            if (!_coupled_state->check_edge_counts(false))
            {
                assert(false);
                return false;
            }
        return true;
    }

    void check_node_counts()
    {
#ifndef NDEBUG
        vector<size_t> wr(num_vertices(_bg));
        for (auto v : vertices_range(_g))
            wr[_b[v]] += _vweight[v];

        for (auto r : vertices_range(_bg))
            assert(size_t(_wr[r]) == wr[r]);
#endif
    }

    template <class V>
    void push_state(V&) {}
    void pop_state() {}
    void store_next_state(size_t) {}
    void clear_next_state() {}

    void relax_update(bool relax)
    {
        if (_egroups)
            _egroups->check(_bg, _mrs);
        _egroups_update = !relax;
        if (_coupled_state != nullptr)
            _coupled_state->relax_update(relax);
    }

    double _global = false;
    template <class G, size_t... Is>
    BlockState* deep_copy(G& g, eweight_t& eweight, rec_t& rec, drec_t& drec,
                          Lrecdx_t& Lrecdx, bool global, index_sequence<Is...>)
    {
        bg_t* bg;
        if constexpr (std::is_same_v<bg_t, GraphInterface::multigraph_t>)
            bg = new bg_t(_bg);
        else
            bg = new bg_t(*(new GraphInterface::multigraph_t(_bg.original_graph())));
        auto abg = boost::any(std::ref(*bg));

        auto args =
            dispatch_args(_args,
                          [&](std::string name, auto& a) -> decltype(auto)
                          {
                              typedef std::remove_reference_t<decltype(a)> a_t;
                              if (name == "_abg")
                              {
                                  auto& abg_ = abg; // workaround clang bug
                                  if constexpr (std::is_same_v<a_t, boost::any>)
                                      return abg_;
                                  return a;
                              }
                              else if (name == "recdx")
                              {
                                  if constexpr (std::is_same_v<a_t, recdx_t>)
                                      return *(new recdx_t(this->_recdx));
                                  return a;
                              }
                              else if (name == "Lrecdx")
                              {
                                  if constexpr (std::is_same_v<a_t, Lrecdx_t>)
                                  {
                                      auto& Lrecdx_ = Lrecdx; // workaround clang bug
                                      if (global)
                                          return *(new Lrecdx_t(Lrecdx_));
                                      else
                                          return Lrecdx_;
                                  }
                                  return a;
                              }
                              else if (name == "epsilon")
                              {
                                  if constexpr (std::is_same_v<a_t, epsilon_t>)
                                      return *(new epsilon_t(this->_epsilon));
                                  return a;
                              }
                              return a;
                          });

        BlockState* state = new BlockState(g, std::get<Is+1>(args)...);
        state->_bgp = std::shared_ptr<bg_t>(bg);
        if constexpr (!std::is_same_v<bg_t, GraphInterface::multigraph_t>)
            state->__bgp =
                std::shared_ptr<GraphInterface::multigraph_t>(&bg->original_graph());
        state->_eweight = eweight;
        state->_rec = rec;
        state->_drec = drec;
        state->_global = global;
        state->_recdxp = std::shared_ptr<recdx_t>(&state->_recdx);
        if (global)
            state->_Lrecdxp = std::shared_ptr<Lrecdx_t>(&state->_Lrecdx);
        state->_epsilonp = std::shared_ptr<epsilon_t>(&state->_epsilon);
        return state;
    }

    BlockState* deep_copy(boost::any args, bool global)
    {
        auto& [ag, eweight, rec, drec, Lrecdx] =
            boost::any_cast<std::tuple<boost::any, eweight_t, rec_t,
                                       drec_t, Lrecdx_t>&>(args);

        BlockState* state = nullptr;
        try
        {
            g_t& g = boost::any_cast<std::reference_wrapper<g_t>>(ag);
            state = deep_copy(g, eweight, rec, drec, Lrecdx,
                              global, make_index_sequence<sizeof...(Ts)-1>{});
        }
        catch (boost::bad_any_cast& e)
        {
            typedef undirected_adaptor<GraphInterface::multigraph_t> ug_t;
            if constexpr (std::is_same_v<g_t, ug_t>)
            {
                GraphInterface::multigraph_t& g =
                    boost::any_cast<std::reference_wrapper<GraphInterface::multigraph_t>>(ag);
                ug_t* ug = new ug_t(g);
                state = deep_copy(*ug, eweight, rec, drec, Lrecdx, global,
                                  make_index_sequence<sizeof...(Ts)-1>{});
                state->_ugp = std::shared_ptr<ug_t>(ug);
            }
            else
            {
                throw e;
            }
        }
        state->_mrs = state->_mrs.copy();
        state->_c_mrs = state->_mrs.get_checked();
        state->_mrp = state->_mrp.copy();
        if (graph_tool::is_directed(_g))
            state->_mrm = state->_mrm.copy();
        else
            state->_mrm = state->_mrp;
        state->_wr = state->_wr.copy();
        state->_b = state->_b.copy();
        state->_egroups = _egroups;
        state->_c_rec.clear();
        for (auto& p : state->_rec)
            state->_c_rec.push_back(p.get_checked());
        state->_c_drec.clear();
        for (auto& p : state->_drec)
            state->_c_drec.push_back(p.get_checked());
        state->_c_brec.clear();
        for (auto& p : state->_brec)
        {
            p = p.copy();
            state->_c_brec.push_back(p.get_checked());
        }
        state->_c_bdrec.clear();
        for (auto& p : state->_bdrec)
        {
            p = p.copy();
            state->_c_bdrec.push_back(p.get_checked());
        }
        state->_recsum = _recsum;
        state->_brecsum = _brecsum.copy();
        if (_coupled_state != nullptr)
        {
            state->_coupled_state =
                _coupled_state->
                    deep_copy(boost::any(std::make_tuple(boost::any(std::ref(state->_bg)),
                                                         state->_mrs,
                                                         state->_brec,
                                                         state->_bdrec,
                                                         state->_Lrecdx)));
            state->_coupled_statep =
                std::shared_ptr<BlockStateVirtualBase>(state->_coupled_state);
            state->_coupled_entropy_args = _coupled_entropy_args;
        }

        return state;
    }

    virtual BlockState* deep_copy(boost::any args)
    {
        return deep_copy(args, false);
    }

    BlockState* deep_copy()
    {
        return deep_copy(boost::any(std::make_tuple(boost::any(std::ref(_g)),
                                                    _eweight,
                                                    _rec,
                                                    _drec,
                                                    _Lrecdx)), true);
    }

    virtual void deep_assign(const BlockStateVirtualBase& state_)
    {
        const BlockState& state = *dynamic_cast<const BlockState*>(&state_);
        if constexpr (is_directed_::apply<g_t>::type::value)
            _bg = state._bg;
        else
            _bg.original_graph() = state._bg.original_graph();
        _mrs.get_storage() = state._mrs.get_storage();
        _mrp.get_storage() = state._mrp.get_storage();
        if constexpr (is_directed_::apply<g_t>::type::value)
            _mrm.get_storage() = state._mrm.get_storage();
        _wr.get_storage() = state._wr.get_storage();
        _b.get_storage() = state._b.get_storage();

        for (size_t i = 0; i < _brec.size(); ++i)
        {
            _brec[i].get_storage() = state._brec[i].get_storage();
            _bdrec[i].get_storage() = state._bdrec[i].get_storage();
        }

        _recdx = state._recdx;
        _Lrecdx = state._Lrecdx;
        _epsilon = state._epsilon;
        _recsum = state._recsum;
        _brecsum = state._brecsum;
        _recdx = state._recdx;
        _recx2 = state._recx2;

        _candidate_groups = state._candidate_groups;
        _empty_groups = state._empty_groups;
        _B_E = state._B_E;
        _B_E_D = state._B_E_D;
        _emat = state._emat;
        _partition_stats.clear();
        for (size_t i = 0; i < _partition_stats.size(); ++i)
            _partition_stats[i] = state._partition_stats[i];

        if (_coupled_state != nullptr)
            _coupled_state->deep_assign(*state._coupled_state);
    }

//private:
    typedef typename
        std::conditional<is_directed_::apply<g_t>::type::value,
                         GraphInterface::multigraph_t,
                         undirected_adaptor<GraphInterface::multigraph_t>>::type
        bg_t;
    bg_t& _bg;

    idx_set<size_t> _candidate_groups;
    idx_set<size_t> _empty_groups;

    typename mrs_t::checked_t _c_mrs;
    std::vector<typename rec_t::value_type::checked_t> _c_rec;
    std::vector<typename drec_t::value_type::checked_t> _c_drec;
    std::vector<typename brec_t::value_type::checked_t> _c_brec;
    std::vector<typename bdrec_t::value_type::checked_t> _c_bdrec;
    std::vector<double> _recsum;
    std::vector<double> _recx2;
    std::vector<double> _dBdx;
    size_t _B_E = 0;
    size_t _B_E_D = 0;
    int _rt = weight_type::NONE;
    size_t _N;
    size_t _E;

    vweight_t _vweight;
    eweight_t _eweight;

    typedef typename std::conditional<is_weighted_t::value,
                                      degs_map_t::unchecked_t,
                                      simple_degs_t>::type degs_t;

    degs_t _degs;

    typedef typename std::conditional<use_hash_t::value,
                                      EHash<bg_t>,
                                      EMat<bg_t>>::type
        emat_t;
    emat_t _emat;

    std::shared_ptr<EGroups> _egroups;
    bool _egroups_update = true;

    std::vector<partition_stats_t> _partition_stats;

    typedef EntrySet<g_t, bg_t, std::vector<double>,
                     std::vector<double>> m_entries_t;
    m_entries_t _m_entries;

    std::vector<std::tuple<size_t, size_t, int>>
        _pp_entries;

    typedef entropy_args_t _entropy_args_t;
    BlockStateVirtualBase* _coupled_state = nullptr;
    entropy_args_t _coupled_entropy_args;
    args_t _args;

    // owned by deep copies
    std::shared_ptr<undirected_adaptor<GraphInterface::multigraph_t>> _ugp;
    std::shared_ptr<bg_t> _bgp;
    std::shared_ptr<GraphInterface::multigraph_t> __bgp;
    std::shared_ptr<BlockStateVirtualBase> _coupled_statep;
    std::shared_ptr<recdx_t> _recdxp;
    std::shared_ptr<Lrecdx_t> _Lrecdxp;
    std::shared_ptr<epsilon_t> _epsilonp;
};

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_HH
