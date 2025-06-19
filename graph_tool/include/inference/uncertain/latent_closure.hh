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

#ifndef GRAPH_BLOCKMODEL_LATENT_CLOSURE_HH
#define GRAPH_BLOCKMODEL_LATENT_CLOSURE_HH

#include "config.h"

#include <vector>

#include "../support/graph_state.hh"
#include <boost/range/counting_range.hpp>

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef eprop_map_t<int32_t>::type emap_t;
typedef eprop_map_t<std::vector<int32_t>>::type evmap_t;
typedef vprop_map_t<int32_t>::type vmap_t;

#define LATENT_CLOSURE_STATE_params                                            \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((oa,, python::list, 0))                                                   \
    ((oaw,, python::list, 0))                                                  \
    ((om,, python::list, 0))                                                   \
    ((ox,, python::list, 0))                                                   \
    ((oM,, python::list, 0))                                                   \
    ((oE,, python::list, 0))                                                   \
    ((b,, vmap_t, 0))                                                          \
    ((L,, size_t, 0))

template <class Graphs, class F>
void iter_out_neighbors(size_t v, Graphs& gs, size_t l,
                        bool old, bool cur, F&& f)
{
    size_t lmax = (cur || l == 0) ? l : l - 1;
    for (size_t i = (old || l == 0) ? 0 : l - 1; i < lmax; ++i)
    {
        for (auto u : out_neighbors_range(v, *gs[i]))
        {
            if (u == v)
                continue;
            f(u);
        }
    }
}

template <class V>
static bool cmp_m(const V& m1, const V& m2)
{
    set<int> s1(m1.begin(), m1.end());
    set<int> s2(m2.begin(), m2.end());
    return s1 == s2;
}

template <class BlockState>
struct LatentClosure
{
    GEN_STATE_BASE(LatentClosureStateBase, LATENT_CLOSURE_STATE_params)

    template <class... Ts>
    class LatentClosureState
        : public LatentClosureStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(LatentClosureStateBase<Ts...>,
                         LATENT_CLOSURE_STATE_params)
        GET_PARAMS_TYPEDEF(Ts, LATENT_CLOSURE_STATE_params)

        typedef typename BlockState::g_t g_t;

        std::vector<g_t*> get_vs(python::list& oa)
        {
            std::vector<g_t*> vs;
            for (int i = 0; i < python::len(oa); ++i)
                vs.push_back(&boost::any_cast<std::reference_wrapper<g_t>>(python::extract<boost::any&>(oa[i])()).get());
            return vs;
        }

        std::vector<emap_t::unchecked_t> get_ws(python::list& oa)
        {
            std::vector<emap_t::unchecked_t> vs;
            for (int i = 0; i < python::len(oa); ++i)
                vs.push_back(boost::any_cast<emap_t>(python::extract<boost::any>(oa[i])()).get_unchecked());
            return vs;
        }

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
        LatentClosureState(BlockState& ebstate, BlockState& pbstate, size_t l, ATs&&... args)
            : LatentClosureStateBase<Ts...>(std::forward<ATs>(args)...),
              _l(l),
              _a(get_vs(_oa)),
              _aw(get_ws(_oaw)),
              _gc(*_a[_l]),
              _gw(_aw[_l]),
              _m(boost::any_cast<evmap_t>(python::extract<boost::any>(_om[_l])()).get_unchecked()),
              _x(boost::any_cast<evmap_t>(python::extract<boost::any>(_ox[_l])()).get_unchecked()),
              _ebstate(ebstate),
              _pbstate(pbstate),
              _g(_gc),
              _eweight(_gw),
              _mark(num_vertices(_g)),
              _M(boost::any_cast<vmap_t>(python::extract<boost::any>(_oM[_l])()).get_unchecked()),
              _E(boost::any_cast<vmap_t>(python::extract<boost::any>(_oE[_l])()).get_unchecked())
        {
            GILRelease gil_release;

            if (_l > 0)
            {
                for (auto v : vertices_range(_g))
                {
                    iter_out_neighbors
                        (v, _a, _l, true, true,
                         [&](auto u)
                         {
                             _mark[u] = 1;
                         });

                    iter_out_neighbors
                        (v, _a, _l, true, false,
                         [&](auto u)
                         {
                             iter_out_neighbors
                                 (u, _a, _l, false, true,
                                  [&](auto w)
                                  {
                                      if (_mark[w] == 0 && w != v)
                                          _M[u]++;
                                  });
                         });

                    iter_out_neighbors
                        (v, _a, _l, false, true,
                         [&](auto u)
                         {
                             iter_out_neighbors
                                 (u, _a, _l, true, true,
                                  [&](auto w)
                                  {
                                      if (_mark[w] == 0 && w != v)
                                          _M[u]++;
                                  });
                         });

                    iter_out_neighbors
                        (v, _a, _l, true, true,
                         [&](auto u)
                         {
                             _mark[u] = 0;
                         });
                }

                for (auto v : vertices_range(_g))
                    _M[v] /= 2;

                for (auto e : edges_range(_gc))
                {
                    _m[e] = get_m(source(e, _gc), target(e, _gc));
                    for (auto u : _x[e])
                    {
                        if (std::find(_m[e].begin(), _m[e].end(), u) == _m[e].end())
                            throw GraphException("Invalid initial state!");
                        _E[u]++;
                    }
                }

                for (auto v : vertices_range(_g))
                {
                    if (_M[v] > 0)
                        _Nm++;
                    if (_E[v] > 0)
                        _Ne++;
                }

                // for (auto e : edges_range(_g))
                // {
                //     auto u = source(e, _g);
                //     auto v = target(e, _g);
                //     if (u == v)
                //         continue;
                //     auto me = get_m(u, v);
                //     for (auto i : me)
                //         assert(_M[i] > 0);
                // }
            }
        }

        size_t _l;

        std::vector<g_t*> _a;
        std::vector<typename emap_t::unchecked_t> _aw;

        g_t& _gc;
        typename emap_t::unchecked_t _gw;

        typename evmap_t::unchecked_t _m;
        typename evmap_t::unchecked_t _x;

        BlockState& _ebstate;
        BlockState& _pbstate;

        g_t& _g;
        typename emap_t::unchecked_t _eweight;

        bool _blayer;

        typename vprop_map_t<int8_t>::type::unchecked_t _mark;

        GraphInterface::edge_t _null_edge;

        std::vector<LatentClosureState*> _cstates;

        typename vprop_map_t<int32_t>::type::unchecked_t _M;
        typename vprop_map_t<int32_t>::type::unchecked_t _E;
        size_t _Nm = 0;
        size_t _Ne = 0;
        size_t _i = std::numeric_limits<size_t>::max();

        void set_cstates(std::vector<LatentClosureState*>& cstates)
        {
            _cstates = cstates;
        }

        std::vector<int32_t> get_m(size_t u, size_t v, bool virt=false)
        {
            std::vector<int32_t> m;

            if (u == v)
                return m;

            if (!virt)
            {
                bool exists = false;
                iter_out_neighbors
                    (u, _a, _l, true, true,
                     [&](auto w)
                     {
                         if (w == v)
                             exists = true;
                     });
                if (exists)
                    return m;
            }

            iter_out_neighbors
                (u, _a, _l, true, false,
                 [&](auto w)
                 {
                     _mark[w] = 1;
                 });

            iter_out_neighbors
                (v, _a, _l, false, true,
                 [&](auto w)
                 {
                     if (_mark[w])
                         m.push_back(w);
                 });

            iter_out_neighbors
                (u, _a, _l, true, false,
                 [&](auto w)
                 {
                     _mark[w] = 0;
                 });

            iter_out_neighbors
                (u, _a, _l, false, true,
                 [&](auto w)
                 {
                     _mark[w] = 1;
                 });

            iter_out_neighbors
                (v, _a, _l, true, true,
                 [&](auto w)
                 {
                     if (_mark[w])
                         m.push_back(w);
                 });

            iter_out_neighbors
                (u, _a, _l, false, true,
                 [&](auto w)
                 {
                     _mark[w] = 0;
                 });

            assert(std::set<int>(m.begin(), m.end()).size() == m.size());
            return m;
        }

        double entropy()
        {
            double L = 0;

            for (auto v : vertices_range(_gc))
            {
                L -= lbinom_fast(_M[v], _E[v]);
                if (_E[v] > 0)
                    L -= safelog_fast(_M[v]);
            }

            L -= lbinom_fast(_Nm, _Ne);
            L -= safelog_fast(_Nm + 1);

            return -L;
        }

        template <class RNG>
        void internal_move_proposal(size_t u, size_t v, RNG& rng)
        {
            if (_l > 0)
            {
                auto me = get_m(u, v, true);
                if (!me.empty())
                    _i = uniform_sample(me, rng);
                else
                    _i = std::numeric_limits<size_t>::max();
            }

            if (_l + 1 < _L)
                _cstates[_l + 1]->internal_move_proposal(u, v, rng);
        }

        template <bool Add, class Edge, class Eargs>
        double modify_edge_dS(size_t u, size_t v, Edge& e, Eargs& ea)
        {
            double dS = 0;
            if (_l == 0)
                dS = _ebstate.modify_edge_dS(u, v, e, Add ? 1 : -1, ea);
            else
                dS = modify_edge_g_dS<Add>(u, v);

            if (std::isinf(dS) || u == v)
                return dS;

            if ((Add && (e == _null_edge || _gw[e] == 0)) || (!Add && _gw[e] == 1))
            {
                for (size_t l = _l + 1; l < _L; ++l)
                {
                    dS += _cstates[l]->template modify_edge_a_dS<Add>(u, v, _l < l - 1);
                    if (std::isinf(dS))
                        return dS;
                }
            }

            return dS;
        }

        template <bool Add, class Edge>
        void modify_edge(size_t u, size_t v, Edge& e)
        {
            if (u != v)
            {
                if ((Add && (e == _null_edge || _gw[e] == 0)) || (!Add && _gw[e] == 1))
                {
                    for (size_t l = _l + 1; l < _L; ++l)
                        _cstates[l]->template modify_edge_a<Add>(u, v, _l < l - 1);
                }
            }

            if (_l == 0)
                _ebstate.template modify_edge<Add>(u, v, e, 1);
            else
                modify_edge_g<Add>(u, v, e);
        }

        template <bool Add>
        double modify_edge_g_dS(size_t u, size_t v)
        {
            if (u == v)
            {
                if constexpr (Add)
                    return std::numeric_limits<double>::infinity();
                else
                    return -std::numeric_limits<double>::infinity();
            }

            auto e = edge(u, v, _gc);

            bool m = false;
            bool x = false;
            if (e.second)
            {
                auto& me = _m[e.first];
                m = std::find(me.begin(), me.end(), _i) != me.end();

                auto& xe = _x[e.first];
                x = std::find(xe.begin(), xe.end(), _i) != xe.end();

                assert(cmp_m(me, get_m(u, v)));
                assert(std::set<int>(me.begin(), me.end()).size() == me.size());
            }
            else
            {
                auto me = get_m(u, v);
                m = std::find(me.begin(), me.end(), _i) != me.end();
            }

            if ((x == Add) || (!m && Add))
                return std::numeric_limits<double>::infinity();

            double Lb = 0, La = 0;

            size_t E = _E[_i];
            size_t M = _M[_i];
            size_t Ne = _Ne;

            Lb -= lbinom_fast(M, E);
            Lb -= lbinom_fast(_Nm, Ne);

            if (E > 0)
                Lb -= safelog_fast(M);

            if constexpr (Add)
            {
                E++;
                if (E == 1)
                    Ne++;
            }
            else
            {
                E--;
                if (E == 0)
                    Ne--;
            }

            La -= lbinom_fast(M, E);
            La -= lbinom_fast(_Nm, Ne);
            if (E > 0)
                La -= safelog_fast(M);

            return -(La - Lb);
        }

        template <bool Add>
        double modify_edge_a_dS(size_t u, size_t v, bool old)
        {
            if (u == v)
                return 0;

            auto e_g = edge(u, v, _gc);
            if (e_g.second && _gw[e_g.first] > 0)
            {
                if constexpr (Add)
                    return std::numeric_limits<double>::infinity();
                else
                    return -std::numeric_limits<double>::infinity();
            }

            double La = 0, Lb = 0;

            gt_hash_map<size_t, int> dM;

            auto get_dS =
                [&](size_t u_, size_t v_)
                {
                    iter_out_neighbors
                        (v_, _a, _l, true, true,
                         [&](auto w)
                         {
                             _mark[w] = 1;
                         });

                    iter_out_neighbors
                        (u_, _a, _l, not old, true,
                         [&](auto w)
                         {
                             if (_mark[w] > 0 || w == v_)
                                 return;

                             if constexpr (Add)
                                 dM[u_]++;
                             else
                                 dM[u_]--;

                             if (!Add)
                             {
                                 auto e_g = edge(v_, w, _gc);
                                 if (!e_g.second)
                                     return;

                                 auto& xe = _x[e_g.first];
                                 if (std::find(xe.begin(), xe.end(), u_) != xe.end())
                                     La = -std::numeric_limits<double>::infinity();
                             }
                         });

                    iter_out_neighbors
                        (v_, _a, _l, true, true,
                         [&](auto w)
                         {
                             _mark[w] = 0;
                         });
                };

            get_dS(u, v);
            if (std::isinf(La))
                return std::numeric_limits<double>::infinity();

            get_dS(v, u);
            if (std::isinf(La))
                return std::numeric_limits<double>::infinity();

            auto m = get_m(u, v, true);
            for (auto i : m)
            {
                if constexpr (Add)
                    dM[i]--;
                else
                    dM[i]++;
            }

            size_t Nm = _Nm;

            Lb -= lbinom_fast(Nm, _Ne);
            Lb -= safelog_fast(Nm + 1);

            for (auto& im : dM)
            {
                if (im.second == 0)
                    continue;

                auto i = im.first;
                auto dM = im.second;

                auto E = _E[i];
                auto M = _M[i];

                Lb -= lbinom_fast(M, E);
                La -= lbinom_fast(M + dM, E);
                if (E > 0)
                {
                    Lb -= safelog_fast(M);
                    La -= safelog_fast(M + dM);
                }

                if (M == 0 && dM > 0)
                    Nm++;
                if (M > 0 && M + dM == 0)
                    Nm--;
            }

            La -= lbinom_fast(Nm, _Ne);
            La -= safelog_fast(Nm + 1);

            return -(La - Lb);
        }

        template <bool Add, class Edge>
        void modify_edge_g(size_t u, size_t v, Edge& new_e)
        {
            assert (_i != std::numeric_limits<size_t>::max());

            auto e = edge(u, v, _gc);
            if (!e.second)
            {
                auto gw = _gw.get_checked();
                auto m = _m.get_checked();
                auto x = _x.get_checked();
                e = add_edge(u, v, _gc);
                gw[e.first] = 0;
                m[e.first] = get_m(u, v);
                x[e.first].clear();
                new_e = e.first;
            }

            auto& xe = _x[e.first];

            if constexpr (Add)
            {
                _gw[e.first]++;
                xe.push_back(_i);
                _E[_i]++;
                if (_E[_i] == 1)
                    _Ne++;
            }
            else
            {
                _gw[e.first]--;
                xe.erase(std::remove(xe.begin(), xe.end(), _i), xe.end());
                if (xe.empty())
                {
                    remove_edge(e.first, _gc);
                    new_e = _null_edge;
                }
                _E[_i]--;
                if (_E[_i] == 0)
                    _Ne--;
            }
        }

        template <bool Add>
        void modify_edge_a(size_t u, size_t v, bool old)
        {
            if (u == v)
                return;

            assert(!edge(u, v, _gc).second || _gw[edge(u, v, _gc).first] == 0);

            auto change_m =
                [&](size_t u_, size_t v_)
                {
                    iter_out_neighbors
                        (v_, _a, _l, true, true,
                         [&](auto w)
                         {
                             _mark[w] = 1;
                         });

                    iter_out_neighbors
                        (u_, _a, _l, not old, true,
                         [&](auto w)
                         {
                             if (_mark[w] > 0 || w == v_)
                                 return;

                             if constexpr (Add)
                             {
                                 _M[u_]++;
                                 if (_M[u_] == 1)
                                     _Nm++;
                             }
                             else
                             {
                                 _M[u_]--;
                                 if (_M[u_] == 0)
                                     _Nm--;
                             }

                             assert(_M[u_] >= 0);


                             auto e_g = edge(v_, w, _gc);
                             if (!e_g.second)
                                 return;

                             auto& m = _m[e_g.first];

                             if constexpr (Add)
                                 m.push_back(u_);
                             else
                                 m.erase(std::remove(m.begin(), m.end(), u_), m.end());
                         });

                    iter_out_neighbors
                        (v_, _a, _l, true, true,
                         [&](auto w)
                         {
                             _mark[w] = 0;
                         });
                };

            change_m(u, v);
            change_m(v, u);

            auto me = get_m(u, v, true);
            for (auto i : me)
            {
                if constexpr (Add)
                {
                    _M[i]--;
                    if (_M[i] == 0)
                        _Nm--;
                }
                else
                {
                    _M[i]++;
                    if (_M[i] == 1)
                        _Nm++;
                }
                assert(_M[i] >= 0);
            }

            auto e_g = edge(u, v, _gc);
            if (e_g.second)
            {
                if constexpr (Add)
                    _m[e_g.first].clear();
                else
                    _m[e_g.first] = me;
            }
        }

        bool check()
        {
            typename vprop_map_t<int32_t>::type::unchecked_t M(num_vertices(_gc));
            typename vprop_map_t<int32_t>::type::unchecked_t E(num_vertices(_gc));

            for (auto e : edges_range(_gc))
            {
                if (!cmp_m(_m[e], get_m(source(e, _gc), target(e, _gc))))
                    assert(false);
                for (auto u : _x[e])
                {
                    if (std::find(_m[e].begin(), _m[e].end(), u) == _m[e].end())
                        assert(false);
                    E[u]++;
                }
            }

            for (auto v : vertices_range(_g))
            {
                if (E[v] != _E[v])
                    assert(false);

                iter_out_neighbors
                    (v, _a, _l, true, true,
                     [&](auto u)
                     {
                         _mark[u] = 1;
                     });

                iter_out_neighbors
                    (v, _a, _l, true, false,
                     [&](auto u)
                     {
                         iter_out_neighbors
                             (u, _a, _l, false, true,
                              [&](auto w)
                              {
                                  if (_mark[w] == 0 && w != v)
                                      M[u]++;
                              });
                     });

                iter_out_neighbors
                    (v, _a, _l, false, true,
                     [&](auto u)
                     {
                         iter_out_neighbors
                             (u, _a, _l, true, true,
                              [&](auto w)
                              {
                                  if (_mark[w] == 0 && w != v)
                                      M[u]++;
                              });
                     });

                iter_out_neighbors
                    (v, _a, _l, true, true,
                     [&](auto u)
                     {
                         _mark[u] = 0;
                     });
            }

            for (auto v : vertices_range(_g))
            {
                M[v] /= 2;
                if (M[v] != _M[v])
                    assert(false);
            }

            return true;
        }

    };

};

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_LATENT_CLOSURE_HH
