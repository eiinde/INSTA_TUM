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

#ifndef DYNAMICS_BASE_HH
#define DYNAMICS_BASE_HH

#include <algorithm>
#include <iostream>
#include <shared_mutex>

#include "../../../hash_map_wrap.hh"
#include "../../../idx_map.hh"

#include "openmp.hh"

#include "dynamics.hh"

namespace graph_tool
{

template <class Type>
std::vector<typename Type::unchecked_t> from_list(python::object list)
{
    vector<typename Type::unchecked_t> v;
    for (int i = 0; i < python::len(list); ++i)
    {
        boost::any& a = python::extract<boost::any&>(list[i])();
        v.push_back(boost::any_cast<Type>(a).get_unchecked());
    }
    return v;
};

template <class Spec, bool discrete, bool keep_k, bool tshift>
class NSumStateBase
    : public DStateBase
{
public:
    typedef std::conditional_t<discrete, int32_t, double> value_t;
    typedef double m_t;

    typedef vprop_map_t<std::vector<int32_t>>::type tmap_t;
    typedef typename vprop_map_t<std::vector<value_t>>::type smap_t;

    template <class State>
    NSumStateBase(State& s, python::object ot, python::object os,
                  bool init_m=true)
        : _t(from_list<tmap_t>(ot)),
          _s(from_list<smap_t>(os)),
          _k(keep_k ? num_vertices(s._u) : 0),
          _spec(*static_cast<Spec*>(this)),
          _theta(s._theta),
          _m_mutex(num_vertices(s._u))
    {
        size_t nt = get_num_threads();
        _ms.resize(nt);
        _nms.resize(nt);
        _svs.resize(nt);
        _snvs.resize(nt);
        _ws.resize(nt);
        _dms.resize(nt);

        for (size_t i = 0; i < nt; ++i)
            _pos.emplace_back(num_vertices(s._u));

        if constexpr (keep_k)
        {
            for (auto v : vertices_range(s._u))
            {
                _k[v] = 0;
                for (auto e : in_or_out_edges_range(v, s._u))
                {
                    auto u = source(e, s._u);
                    if (u == v && !s._self_loops)
                        continue;
                    _k[v] += abs(s._x[e]);
                }
            }
        }

        _T = 0;
        if (_t.empty())
        {
            for (auto& sn : _s)
            {
                size_t T = 0;
                for (auto v : vertices_range(s._u))
                {
                    if (T == 0)
                        T = sn[v].size();
                    if (sn[v].size() != T)
                        throw ValueException("invalid uncompressed time "
                                             "series: all vertices must "
                                             "have the same number of "
                                             "states");
                }
                _T += T - int(tshift);
                _Tn.push_back(T - int(tshift));
                for (size_t i = 0; i < nt; ++i)
                    _dms[i].emplace_back(T - int(tshift));
            }
        }
        else
        {
            for (size_t n = 0; n < _t.size(); ++n)
            {
                auto& sn = _s[n];
                auto& tn = _t[n];
                for (auto v : vertices_range(s._u))
                {
                    if (sn[v].size() != tn[v].size())
                        throw ValueException("invalid compressed time "
                                             "series: all vertices must "
                                             "have the same number of "
                                             "states and times");
                    if (sn[v].empty())
                        throw ValueException("invalid compressed time "
                                             "series: all vertices must "
                                             "have nonempty states and times");
                }

                int T = 0;
                for (auto v : vertices_range(s._u))
                    T = std::max(tn[v].back(), T);
                for (auto v : vertices_range(s._u))
                {
                    auto& sv = sn[v];
                    auto& tv = tn[v];
                    if (tv.back() < T)
                    {
                        tv.push_back(T);
                        sv.push_back(sv.back());
                    }
                }
                _Tn.push_back(T - int(tshift));
                _T += T + 1 - int(tshift);
                for (size_t i = 0; i < nt; ++i)
                    _dms[i].emplace_back(T - int(tshift));
            }
        }

        for (auto sn : _s)
            _m.emplace_back(num_vertices(s._u));

        _m_temp.resize(nt);
        for (auto& mt : _m_temp)
            mt.resize(_s.size());

        if (init_m)
            reset_m(s);
    };

    template <class State>
    void reset_m(State& s)
    {
        #pragma omp parallel
        parallel_vertex_loop_no_spawn
            (s._u,
             [&](auto v)
             {
                 for (auto& m : _m)
                     m[v].clear();
             });

        auto xc = s._x.get_checked();

        #pragma omp parallel
        parallel_vertex_loop_no_spawn
            (s._u,
             [&](auto v)
             {
                 iter_time<false, false, false>
                     (in_or_out_neighbors_range(v, s._u), v,
                      [&](auto n, auto t, auto& su, size_t)
                      {
                          m_t m = 0;
                          for (auto e : in_or_out_edges_range(v, s._u))
                          {
                              auto u = source(e, s._u);
                              if (u == v && !s._self_loops)
                                  continue;
                              m += _spec.transform_input(v, u, su(u)) * xc[e];
                          }
                          if (_t.empty() || t == 0 || m != get<1>(_m[n][v].back()))
                              _m[n][v].emplace_back(t, m);
                      });

                 for (auto& m : _m)
                 {
                     if (m[v].empty())
                         m[v].emplace_back(0, 0);
                 }
             });
    }

    // template <class State>
    // bool check_m(State& s, size_t v)
    // {
    //     auto xc = s._x.get_checked();
    //     bool check = true;

    //     iter_time<true, false, false>
    //         (in_or_out_neighbors_range(v, s._u), v,
    //          [&](auto, auto t, auto, auto& su, auto om, size_t)
    //          {
    //              m_t m = 0;
    //              for (auto e : in_or_out_edges_range(v, s._u))
    //              {
    //                  auto u = source(e, s._u);
    //                  if (u == v && !s._self_loops)
    //                      continue;
    //                  m += xc[e] * _spec.transform_input(v, u, su(u));
    //              }
    //              if (abs(om - m) > 1e-8)
    //              {
    //                  std::cout << m << " " << om << std::endl;
    //                  assert(false);
    //                  check = false;
    //              }
    //          });
    //     return check;
    // }

    template <bool follow_m, bool follow_v, bool update_m, class F>
    void iter_time_uncompressed(size_t v, F&& f)
    {
        for (size_t n = 0; n < _s.size(); ++n)
        {
            auto& snv = _s[n][v];
            auto& mnv = _m[n][v];
            auto& sn = _s[n];

            m_t zero = 0;

            for (size_t t = 0; t < snv.size() - int(tshift); ++t)
            {
                [[maybe_unused]] auto& m = follow_m ? get<1>(mnv[t]) : zero;
                [[maybe_unused]] auto s_v = follow_v ? snv[t] : 0;
                [[maybe_unused]] auto s_nv = (follow_v && tshift) ? snv[t+1] : 0;

                auto st = [&](auto u) { return sn[u][t]; };

                if constexpr (follow_m)
                {
                    if constexpr (follow_v)
                    {
                        if constexpr (tshift)
                            f(n, t, st, m, 1, s_v, s_nv);
                        else
                            f(n, t, st, m, 1, s_v);
                    }
                    else
                    {
                        f(n, t, st, m, 1);
                    }
                }
                else
                {
                    if constexpr (follow_v)
                    {
                        if constexpr (tshift)
                            f(n, t, st, 1, s_v, s_nv);
                        else
                            f(n, t, st, 1, s_v);
                    }
                    else
                    {
                        f(n, t, st, 1);
                    }
                }
            }
        }
    }

    template <bool follow_m, bool follow_v, bool update_m, class US, class F>
    void iter_time_compressed(US&& us, size_t v, F&& f)
    {
        size_t tid = get_thread_num();

        auto& m_temp = _m_temp[tid];
        if constexpr (update_m)
        {
            for (auto& m : m_temp)
                m.clear();
        }

        auto& pos = _pos[tid];

        m_t zero = 0;

        for (size_t n = 0; n < _s.size(); ++n)
        {
            auto& snv = _s[n][v];
            auto& tn = _t[n];
            auto& sn = _s[n];

            if (tshift && snv.size() <= 1)
                continue;

            for (auto u : us)
                pos[u] = 0;

            auto st = [&](auto u) { return sn[u][pos[u]]; };

            [[maybe_unused]] size_t pos_m = 0;
            [[maybe_unused]] auto& mnv = _m[n][v];
            [[maybe_unused]] auto* m = (follow_m) ? &get<1>(mnv[0]) : &zero;

            [[maybe_unused]] auto& tnv = _t[n][v];
            [[maybe_unused]] size_t pos_v = 0;
            [[maybe_unused]] auto s_v = (follow_v) ? snv[0] : 0;
            [[maybe_unused]] size_t pos_nv = 0;
            [[maybe_unused]] auto s_nv = (follow_v) ? snv[0] : 0;

            if constexpr (tshift && follow_v)
            {
                if (pos_nv + 1 < tnv.size() && tnv[pos_nv + 1] == 1)
                    s_nv = snv[++pos_nv];
            }

            size_t t = 0;
            while (t <= _Tn[n])
            {
                // determine next time point
                auto nt = _Tn[n];
                for (auto u : us)
                {
                    auto upos = pos[u];
                    auto& tnu = tn[u];
                    if (upos + 1 < tnu.size())
                        nt = std::min(nt, size_t(tnu[upos + 1]));
                }

                if constexpr (follow_m)
                {
                    if (pos_m + 1 < mnv.size())
                        nt = std::min(nt, get<0>(mnv[pos_m + 1]));
                }

                if constexpr (follow_v)
                {
                    if (pos_v + 1 < tnv.size())
                        nt = std::min(nt, size_t(tnv[pos_v + 1]));

                    if constexpr (tshift)
                    {
                        // need to be t-1 w.r.t. nv
                        if (pos_nv + 1 < tnv.size())
                            nt = std::min(nt, size_t(tnv[pos_nv + 1] - 1));
                    }
                }

                int w = nt - t;
                if constexpr (follow_m)
                {
                    if constexpr (update_m)
                    {
                        auto& mn = m_temp[n];
                        mn.emplace_back(t, *m);
                        m = &get<1>(mn.back());
                    }

                    if constexpr (follow_v)
                    {
                        if constexpr (tshift)
                            f(n, t, st, *m, w, s_v, s_nv);
                        else
                            f(n, t, st, *m, w, s_v);
                    }
                    else
                    {
                        f(n, t, st, *m, w);
                    }

                    if constexpr (update_m)
                    {
                        auto& mn = m_temp[n];
                        if (mn.size() > 1 && *m == get<1>(mn[mn.size() - 2]))
                        {
                            mn.pop_back();
                            m = &get<1>(mn.back());
                        }
                    }
                }
                else
                {
                    if constexpr (follow_v)
                    {
                        if constexpr (tshift)
                            f(n, t, st, w, s_v, s_nv);
                        else
                            f(n, t, st, w, s_v);
                    }
                    else
                    {
                        f(n, t, st, w);
                    }
                }

                if (t == _Tn[n])
                    break;

                t = nt;

                // update current states at time t
                for (auto u : us)
                {
                    auto& upos = pos[u];
                    auto& tnu = tn[u];
                    auto npos = upos + 1;
                    if (npos < tnu.size() && t == size_t(tnu[npos]))
                        upos = npos;
                }

                if constexpr (follow_m)
                {
                    // update current m
                    auto npos_m = pos_m + 1;
                    if (npos_m < mnv.size() && t == get<0>(mnv[npos_m]))
                    {
                        m = &get<1>(mnv[npos_m]);
                        pos_m = npos_m;
                    }
                    else if constexpr (update_m)
                    {
                        m = &get<1>(mnv[pos_m]);
                    }
                }

                if constexpr (follow_v)
                {
                    // update v state
                    auto npos_v = pos_v + 1;
                    if (npos_v < tnv.size() && t == size_t(tnv[npos_v]))
                    {
                        s_v = snv[npos_v];
                        pos_v = npos_v;
                    }

                    if constexpr (tshift)
                    {
                        // update nv state
                        auto npos_nv = pos_nv + 1;
                        if (npos_nv < tnv.size() && t == size_t(tnv[npos_nv] - 1))
                        {
                            s_nv = snv[npos_nv];
                            pos_nv = npos_nv;
                        }
                    }
                }
            }
        }

        if constexpr (update_m)
        {
            for (size_t n = 0; n < m_temp.size(); ++n)
            {
                auto& m = _m[n][v];
                m.swap(m_temp[n]);
                if (m.empty())
                    m.emplace_back(0, 0);
            }
        }
    }


    template <bool follow_m, bool follow_v, bool update_m, class US, class F>
    void iter_time(US&& us, size_t v, F&& f)
    {
        if (_t.empty())
        {
            iter_time_uncompressed<follow_m, follow_v, update_m>
                (v, std::forward<F&&>(f));
        }
        else
        {
            auto dispatch =
                [&]()
                {
                    iter_time_compressed<follow_m, follow_v, update_m>
                        (std::forward<US&&>(us), v, std::forward<F&&>(f));
                };

            if constexpr (follow_m)
            {
                if constexpr (update_m)
                {
                    std::unique_lock lock(_m_mutex[v]);
                    dispatch();
                }
                else
                {
                    std::shared_lock lock(_m_mutex[v]);
                    dispatch();
                }
            }
            else
            {
                dispatch();
            }
        }
    }

    template <class VS, class DX>
    void update_edges_dispatch(VS&& us, size_t v, const DX& x, const DX& nx)
    {
        if constexpr (keep_k)
        {
            auto& kv = _k[v];
            for (size_t i = 0; i < x.size(); ++i)
                kv += abs(nx[i]) - abs(x[i]);
        }

        DX dx(nx);
        for (size_t i = 0; i < x.size(); ++i)
            dx[i] -= x[i];

        iter_time<true, false, true>
            (us, v,
             [&](auto, auto, auto& su, auto& m, int)
             {
                 m_t dm = 0;
                 for (size_t i = 0; i < us.size(); ++i)
                 {
                     auto& u = us[i];
                     dm += _spec.transform_input(v, u, su(u)) * dx[i];
                 }
                m += dm;
                assert(!std::isinf(m) && !std::isnan(m));
             });
    }

    void update_edges(std::vector<size_t>& us, size_t v, std::vector<double>& x,
                      std::vector<double>& nx)
    {
        update_edges_dispatch(us, v, x, nx);
    }

    void update_edge(size_t u, size_t v, double x, double nx)
    {
        update_edges_dispatch(std::array<size_t,1>({u}), v,
                              std::array<double,1>({x}),
                              std::array<double,1>({nx}));
    }

    double get_edge_dS(size_t u, size_t v, double x, double nx)
    {
        if (nx == x)
            return 0;
        if (_t.empty())
            return get_edge_dS_uncompressed(u, v, x, nx);
        else
            return get_edge_dS_compressed(u, v, x, nx);
    }

    double get_edge_dS_compressed(size_t u, size_t v, double x, double nx);

    double get_edge_dS_uncompressed(size_t u, size_t v, double x, double nx);

    double get_edges_dS(std::vector<size_t>& us, size_t v, std::vector<double>& x,
                        std::vector<double>& nx)
    {
        if (_t.empty())
            return get_edges_dS_uncompressed(us, v, x, nx);
        else
            return get_edges_dS_compressed(us, v, x, nx);
    }

    double get_edges_dS_compressed(std::vector<size_t>& us, size_t v,
                                   std::vector<double>& x,
                                   std::vector<double>& nx);

    double get_edges_dS_uncompressed(std::vector<size_t>& us, size_t v,
                                     std::vector<double>& x,
                                     std::vector<double>& nx);

    double get_edges_dS(const std::array<size_t,2>& us, size_t v,
                        const std::array<double,2>& x,
                        const std::array<double,2>& nx)
    {
        if (_t.empty())
            return get_edges_dS_uncompressed(us, v, x, nx);
        else
            return get_edges_dS_compressed(us, v, x, nx);
    }

    double get_edges_dS_compressed(const std::array<size_t,2>& us, size_t v,
                                   const std::array<double,2>& x,
                                   const std::array<double,2>& nx);

    double get_edges_dS_uncompressed(const std::array<size_t,2>& us, size_t v,
                                     const std::array<double,2>& x,
                                     const std::array<double,2>& nx);

    template <bool compressed, bool m_offload, class VS, class DX>
    double get_edge_dS_dispatch_direct(VS&& us, size_t v, const DX& x,
                                       const DX& nx)
    {
        auto k = keep_k ? _k[v] : 0;
        auto nk = k;
        if constexpr (keep_k)
        {
            for (size_t i = 0; i < x.size(); ++i)
                nk += abs(nx[i]) - abs(x[i]);
        }

        DX dx(nx);
        for (size_t i = 0; i < x.size(); ++i)
            dx[i] -= x[i];

        auto theta = _theta[v];
        auto ntheta = theta;
        theta = _spec.transform_theta(theta, k);
        ntheta = _spec.transform_theta(ntheta, nk);

        double La = 0;
        double Lb = 0;

        auto tid = get_thread_num();
        std::vector<std::vector<m_t>>& dms = _dms[tid];

        if constexpr (m_offload)
        {
            auto f =
                [&](auto n, auto t, auto&& su, auto, int, auto...)
                {
                    auto& dm = dms[n][t];
                    dm = 0;
                    for (size_t i = 0; i < us.size(); ++i)
                    {
                        auto& u = us[i];
                        dm += _spec.transform_input(v, u, su(u)) * dx[i];
                    }
                };

            if constexpr (!compressed)
                iter_time_uncompressed<true, true, false>(v, f);
            else
                iter_time_compressed<true, true, false>(us, v, f);
        }

        auto f =
            [&](auto n, auto t, auto&& su, auto m, int w, auto... s_v)
            {
                m_t dm = 0;
                if constexpr (m_offload)
                {
                    dm = dms[n][t];
                }
                else
                {
                    for (size_t i = 0; i < us.size(); ++i)
                    {
                        auto& u = us[i];
                        dm += _spec.transform_input(v, u, su(u)) * dx[i];
                    }
                }

                if constexpr (!keep_k)
                {
                    Lb += _spec.log_P(theta, m, s_v...) * w;
                    La += _spec.log_P(theta, m + dm, s_v...) * w;
                }
                else
                {
                    Lb += _spec.log_P(theta, m, k, s_v...) * w;
                    La += _spec.log_P(ntheta, m + dm, nk, s_v...) * w;
                }
            };

        if constexpr (!compressed)
            iter_time_uncompressed<true, true, false>(v, f);
        else
            iter_time_compressed<true, true, false>(us, v, f);

        auto dL = La - Lb;

        return -dL;
    }


    std::vector<std::vector<m_t>> _ms, _nms;
    std::vector<std::vector<value_t>> _svs, _snvs;
    std::vector<std::vector<int>> _ws;
    std::vector<std::vector<std::vector<m_t>>> _dms;

    template <bool compressed, class VS, class DX>
    double get_edge_dS_dispatch_indirect(VS&& us, size_t v, const DX& x,
                                         const DX& nx)
    {
        auto k = keep_k ? _k[v] : 0;
        auto nk = k;
        if constexpr (keep_k)
        {
            for (size_t i = 0; i < x.size(); ++i)
                nk += abs(nx[i]) - abs(x[i]);
        }

        DX dx(nx);
        for (size_t i = 0; i < x.size(); ++i)
            dx[i] -= x[i];

        double La = 0;
        double Lb = 0;

        auto tid = get_thread_num();
        std::vector<m_t>& ms = _ms[tid];
        std::vector<m_t>& nms = _nms[tid];
        std::vector<value_t>& svs = _svs[tid];
        std::vector<value_t>& snvs = _snvs[tid];
        std::vector<int>& ws = _ws[tid];

        ms.clear();
        nms.clear();
        svs.clear();
        snvs.clear();
        ws.clear();

        auto f =
            [&](auto, auto, auto&& su, auto m, int w, auto s_v, auto... s_nv)
            {
                m_t dm = 0;
                for (size_t i = 0; i < us.size(); ++i)
                {
                    auto& u = us[i];
                    dm += _spec.transform_input(v, u, su(u)) * dx[i];
                }
                ms.push_back(m);
                nms.push_back(m + dm);
                svs.push_back(s_v);
                if constexpr (sizeof...(s_nv) > 0)
                    snvs.emplace_back(s_nv...);
                ws.push_back(w);
            };

        if constexpr (!compressed)
            iter_time_uncompressed<true, true, false>(v, f);
        else
            iter_time_compressed<true, true, false>(us, v, f);

        auto theta = _theta[v];
        auto ntheta = theta;
        theta = _spec.transform_theta(theta, k);
        ntheta = _spec.transform_theta(ntheta, nk);

        for (size_t i = 0; i < ws.size(); ++i)
        {
            auto m = ms[i];
            auto nm = nms[i];
            auto s_v = svs[i];
            auto w = ws[i];
            if constexpr (!tshift)
            {
                if constexpr (!keep_k)
                {
                    Lb += _spec.log_P(theta, m, s_v) * w;
                    La += _spec.log_P(theta, nm, s_v) * w;
                }
                else
                {
                    Lb += _spec.log_P(theta, m, k, s_v) * w;
                    La += _spec.log_P(ntheta, nm, nk, s_v) * w;
                }
            }
            else
            {
                auto s_nv = snvs[i];
                if constexpr (!keep_k)
                {
                    Lb += _spec.log_P(theta, m, s_v, s_nv) * w;
                    La += _spec.log_P(theta, nm, s_v, s_nv) * w;
                }
                else
                {
                    Lb += _spec.log_P(theta, m, k, s_v, s_nv) * w;
                    La += _spec.log_P(ntheta, nm, nk, s_v, s_nv) * w;
                }
            }
        }

        auto dL = La - Lb;

        return -dL;
    }

    double get_node_prob(size_t v);

    double get_node_prob(size_t v, size_t n, size_t t, double s)
    {
        // TODO: fix compressed!
        double L;
        auto m = get<1>(_m[n][v][t]);
        auto theta = _theta[v];
        if constexpr (!keep_k)
        {
            if constexpr (tshift)
                L = _spec.log_P(theta, m, _s[n][v][t], s);
            else
                L = _spec.log_P(theta, m, s);
        }
        else
        {
            auto k = _k[v];
            theta = _spec.transform_theta(theta, k);
            if constexpr (tshift)
                L = _spec.log_P(theta, m, k, _s[n][v][t], s);
            else
                L = _spec.log_P(theta, m, k, s);
        }
        return L;
    }

    double get_node_dS(size_t v, double dt)
    {
        if (_t.empty())
            return get_node_dS_uncompressed(v, dt);
        else
            return get_node_dS_compressed(v, dt);
    }

    double get_node_dS_compressed(size_t v, double dt);

    double get_node_dS_uncompressed(size_t v, double dt);

    template <bool compressed>
    double get_node_dS_dispatch(size_t v, double dt)
    {
        double La = 0;
        double Lb = 0;
        auto k = keep_k ? _k[v] : 0;
        auto theta = _theta[v];
        auto ntheta = theta + dt;
        theta = _spec.transform_theta(theta, k);
        ntheta = _spec.transform_theta(ntheta, k);

        auto f =
            [&](auto, auto, auto&&, auto m, int w, auto... s_v)
            {
                if constexpr (!keep_k)
                {
                    Lb += _spec.log_P(theta, m, s_v...) * w;
                    La += _spec.log_P(ntheta, m, s_v...) * w;
                }
                else
                {
                    Lb += _spec.log_P(theta, m, k, s_v...) * w;
                    La += _spec.log_P(ntheta, m, k, s_v...) * w;
                }
             };

        if constexpr (!compressed)
            iter_time_uncompressed<true, true, false>(v, f);
        else
            iter_time_compressed<true, true, false>
                (std::array<size_t,0>({}), v, f);

        auto dL = La - Lb;

        return -dL;
    }

    double node_TE(size_t u, size_t v)
    {
        return node_TE(u, v,
                       [](auto x){return x;},
                       [](auto x){return x;});
    }

    template <class XB, class MB>
    double node_TE(size_t u, size_t v, XB&& get_x_bin, MB&& get_m_bin)
    {
        gt_hash_map<std::tuple<int,int>, size_t> pv;
        gt_hash_map<std::tuple<int,int,int>, size_t> pu, pvt;
        gt_hash_map<std::tuple<int,int,int,int>, size_t> put;

        size_t N = 0;

        auto f = [&](auto x_u_, auto x_v_, auto x_vt_, auto m, int w)
            {
                int x_u = get_x_bin(x_u_);
                int x_v = get_x_bin(x_v_);
                int x_vt = get_x_bin(x_vt_);
                int mi = get_m_bin(m);

                pv[{x_v, mi}] += w;
                pu[{x_v, mi, x_u}] += w;
                put[{x_v, mi, x_u, x_vt}] += w;
                pvt[{x_v, mi, x_vt}] += w;

                N += w;
            };

        if constexpr (tshift)
        {
            iter_time<true, true, false>
                (std::array<size_t, 1>({u}), v,
                 [&](auto, auto, auto& su, auto m, int w, auto& s, auto& s_nv)
                 {
                     f(su(u), s, s_nv, m, w);
                 });
        }
        else
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double I1 = 0;
        for (auto& [x, w] : pvt)
        {
            auto& [x_v, m, x_vt] = x;
            auto n = pv[{x_v, m}];
            I1 -= w * (log(w) - log(n));
        }

        double I2 = 0;
        for (auto& [x, w] : put)
        {
            auto& [x_v, m, x_u, x_vt] = x;
            auto n = pu[{x_v, m, x_u}];
            I2 -= w * (log(w) - log(n));
        }

        return (I1 - I2)/N;
    }

    double node_MI(size_t u, size_t v)
    {
        return node_MI(u, v,
                       [](auto x){return x;},
                       [](auto x){return x;});
    }

    template <class XB, class MB>
    double node_MI(size_t u, size_t v, XB&& get_x_bin,
                   MB&& get_m_bin)
    {
        gt_hash_map<std::tuple<int, int>, int> pv, pu;
        gt_hash_map<std::tuple<int, int, int>, int> puv;
        gt_hash_map<int, int> pm;

        size_t N = 0;

        auto f = [&](auto x_u_, auto x_v_, double m, int w)
            {
                int x_u = get_x_bin(x_u_);
                int x_v = get_x_bin(x_v_);
                int mi = get_m_bin(m);
                pu[{x_u, mi}] += w;
                pv[{x_v, mi}] += w;
                puv[{x_u, x_v, mi}] += w;
                pm[mi] += w;
                N += w;
            };

        if constexpr (tshift)
        {
            iter_time<true, true, false>
                (std::array<size_t, 1>({u}), v,
                 [&](auto, auto, auto& su, auto m, int w, auto& s,
                     auto&)
                 {
                     f(su(u), s, m, w);
                 });
        }
        else
        {
            iter_time<true, true, false>
                (std::array<size_t,1>({u}), v,
                 [&](auto, auto, auto& su, auto m, int w, auto& s)
                 {
                     f(su(u), s, m, w);
                 });
        }

        double MI = 0;
        for (auto& [xm, w] : puv)
        {
            auto& [xu, xv, m] = xm;
            auto nx = pu[{xu, m}];
            auto ny = pv[{xv, m}];
            MI += w * (log(w) - log(nx) - log(ny));
        }
        MI = MI/N + log(N);

        double H = 0;
        for (auto& [mi, w] : pm)
            H -= w * log(w);
        H = H/N + log(N);

        return MI - H;
    }

    template <class XB, class MB>
    double node_dist(size_t u, size_t v, bool directed,
                     XB&& get_x_bin,
                     MB&& get_m_bin)
    {
        if constexpr (tshift)
        {
            if (directed)
                return -node_TE(u, v, get_x_bin, get_m_bin);
            else
                return (-node_TE(u, v, get_x_bin, get_m_bin)
                        -node_TE(v, u, get_x_bin, get_m_bin));
        }
        else
        {
            if (directed)
                return -node_MI(u, v, get_x_bin, get_m_bin);
            else
                return (-node_MI(u, v, get_x_bin, get_m_bin)
                        -node_MI(v, u, get_x_bin, get_m_bin));
        }
    }

    double node_cov(size_t u, size_t v, bool toffset, bool pearson)
    {
        double xy = 0;
        double ax = 0, ay = 0;
        double ax2 = 0, ay2 = 0;
        size_t N = 0;

        auto f = [&](auto x, auto y, int w)
            {
                N += w;
                xy += x * y * w;
                ax += x * w;
                ay += y * w;
                ax2 += x * x * w;
                ay2 += y * y * w;
            };

        if constexpr (tshift)
        {
            iter_time<false, true, false>
                (std::array<size_t,1>({u}), v,
                 [&](auto, auto, auto& su, int w, auto& s, auto& s_nv)
                 {
                     if (toffset)
                         f(su(u), s_nv, w);
                     else
                         f(su(u), s, w);
                 });
        }
        else
        {
            iter_time<false, true, false>
                (std::array<size_t,1>({u}), v,
                 [&](auto, auto, auto& su, int w, auto& s)
                 {
                     f(su(u), s, w);
                 });
        }
        xy /= N;
        ax /= N;
        ay /= N;
        ax2 /= N;
        ay2 /= N;
        if (pearson)
            return (xy - ax * ay) / sqrt((ax2 - ax * ax) * (ay2 - ay * ay));
        else
            return (xy - ax * ay);
    }

    typedef typename vprop_map_t<std::vector<std::tuple<size_t, m_t>>>::type::unchecked_t mmap_t;

//protected:
    std::vector<tmap_t::unchecked_t> _t;
    std::vector<typename smap_t::unchecked_t> _s;
    std::vector<size_t> _Tn;
    size_t _T;

    std::vector<vprop_map_t<size_t>::type::unchecked_t> _pos;
    typename vprop_map_t<double>::type::unchecked_t _k;

    std::vector<mmap_t> _m;
    std::vector<std::vector<std::vector<std::tuple<size_t, m_t>>>> _m_temp;

    Spec& _spec;

    typename vprop_map_t<double>::type::unchecked_t _theta;

    std::vector<std::shared_mutex> _m_mutex;
};

}// graph_tool namespace

#endif //DYNAMICS_BASE_HH
