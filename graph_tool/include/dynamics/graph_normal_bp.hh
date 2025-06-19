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

#ifndef GRAPH_BP_HH
#define GRAPH_BP_HH

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "random.hh"
#include "parallel_rng.hh"

#include "../inference/support/util.hh"
#include "../generation/sampler.hh"

namespace graph_tool
{

class NormalBPState
{
public:

    typedef vprop_map_t<double>::type::unchecked_t vmap_t;
    typedef eprop_map_t<double>::type::unchecked_t emap_t;
    typedef eprop_map_t<std::vector<double>>::type::unchecked_t emmap_t;
    typedef vprop_map_t<uint8_t>::type::unchecked_t vfmap_t;

    template <class Graph, class RNG>
    NormalBPState(Graph& g, emap_t x, vmap_t mu, vmap_t theta, emmap_t em_m,
                  emmap_t em_s, vmap_t vm_m, vmap_t vm_s, bool marginal_init,
                  vfmap_t frozen, RNG&)
        : _x(x), _mu(mu), _theta(theta), _em_m(em_m), _em_s(em_s), _vm_m(vm_m),
          _vm_s(vm_s), _frozen(frozen)
    {
        std::normal_distribution<> mrand(0, 1);
        std::exponential_distribution<> srand(1);

        for (auto e : edges_range(g))
        {
            _em_m[e].resize(2);
            _em_s[e].resize(2);

            auto u = source(e, g);
            auto v = target(e, g);
            auto& m_uv = get_message(g, e, _em_m, u);
            auto& m_vu = get_message(g, e, _em_m, v);
            auto& s_uv = get_message(g, e, _em_s, u);
            auto& s_vu = get_message(g, e, _em_s, v);

            if (marginal_init)
            {
                m_uv = _vm_m[v];
                m_vu = _vm_m[u];
                s_uv = _vm_s[v];
                s_vu = _vm_s[u];
            }
            else
            {
                m_uv = 0; //mrand(rng);
                m_vu = 0; //mrand(rng);
                s_uv = 0; //srand(rng);
                s_vu = 0; //srand(rng);
            }
        }

        _temp_m = _em_m.copy();
        _temp_s = _em_s.copy();
    };

    template <class Graph, class Edge, class ME>
    double& get_message(Graph& g, const Edge& e, ME& me, size_t s)
    {
        auto u = source(e, g);
        auto v = target(e, g);
        if (u > v)
            std::swap(u, v);
        auto& m = me[e];
        if (s == u)
            return m[0];
        else
            return m[1];
    }

    template <class Graph>
    std::tuple<double, double>
    get_sums(Graph& g, size_t s, size_t t)
    {
        double mt = 0;
        double st = 0;
        for (auto ue : out_edges_range(s, g))
        {
            auto u = target(ue, g);
            if (u == t)
                continue;
            auto m_u_m = get_message(g, ue, _em_m, u);
            auto m_u_s = get_message(g, ue, _em_s, u);
            auto w = _x[ue];

            mt += w * m_u_m;
            st += w * w * m_u_s;
        }
        return {mt, st};
    }

    template <class Graph>
    double update_message(Graph& g, double& m_m, double& m_s, size_t s, size_t t)
    {
        auto [mt, st] = get_sums(g, s, t);
        double a = _theta[s] - st;
        double nm_m = (mt - _mu[s]) / a;
        double nm_s = 1. / a;
        double delta = abs(m_m - nm_m) + abs(m_s - nm_s);
        m_m = nm_m;
        m_s = nm_s;
        return delta;
    }

    template <class Graph, class Edge, class ME>
    double update_edge(Graph& g, const Edge& e, ME& me_m, ME& me_s)
    {
        auto u = source(e, g);
        auto v = target(e, g);
        auto& muv_m = get_message(g, e, me_m, u);
        auto& mvu_m = get_message(g, e, me_m, v);
        auto& muv_s = get_message(g, e, me_s, u);
        auto& mvu_s = get_message(g, e, me_s, v);
        double delta = 0;
        if (!_frozen[v])
            delta += update_message(g, muv_m, muv_s, u, v);
        if (!_frozen[u])
            delta += update_message(g, mvu_m, mvu_s, v, u);
        return delta;
    }

    template <class Graph>
    void update_marginals(Graph& g)
    {
        parallel_vertex_loop
            (g,
             [&] (auto v)
             {
                 auto& m_m = _vm_m[v];
                 auto& m_s = _vm_s[v];
                 update_message(g, m_m, m_s, v, _null);
             });
    }

    template <class Graph>
    double iterate(Graph& g, size_t niter)
    {
        double delta = 0;
        for (size_t i = 0; i < niter; ++i)
        {
            delta = 0;
            for (auto e : edges_range(g))
                delta += update_edge(g, e, _em_m, _em_s);
        }
        return delta;
    }

    template <class Graph>
    double iterate_parallel(Graph& g, size_t niter)
    {
        double delta = 0;
        for (size_t i = 0; i < niter; ++i)
        {
            delta = 0;
            #pragma omp parallel reduction(+:delta)
            parallel_edge_loop_no_spawn
                (g,
                 [&] (const auto& e)
                 {
                     _temp_m[e] = _em_m[e];
                     _temp_s[e] = _em_s[e];
                     delta += update_edge(g, e, _temp_m, _temp_s);
                 });

            #pragma omp parallel reduction(+:delta)
            parallel_edge_loop_no_spawn
                (g,
                 [&] (const auto& e)
                 {
                     _em_m[e] = _temp_m[e];
                     _em_s[e] = _temp_s[e];
                 });
        }
        return delta;
    }

    template <class Graph>
    double log_Z(Graph& g)
    {
        double lZ = 0;

        #pragma omp parallel reduction(+:lZ)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 if (_frozen[v])
                     return;
                 auto [mt, st] = get_sums(g, v, _null);
                 double a = (_theta[v] - st)/2;
                 double b = mt - _mu[v];
                 lZ += (b * b)/(4 * a) - log(a)/2 + log(M_PI)/2;
             });

        #pragma omp parallel reduction(+:lZ)
        parallel_edge_loop_no_spawn
            (g,
             [&] (const auto& e)
             {
                 auto u = source(e, g);
                 auto v = target(e, g);
#ifdef __clang__
                 auto ret = get_sums(g, u, v);
                 auto mt = get<0>(ret);
                 auto st = get<1>(ret);
#else
                 auto [mt, st] = get_sums(g, u, v);
#endif
                 auto get_lZ =
                     [&](auto w)
                     {
                         double a = (_theta[w] - st)/2;
                         double b = mt - _mu[w];
                         auto lZ_e = (b * b)/(4 * a) - log(a)/2; //+ log(M_PI)/2;

                         std::tie(mt, st) = get_sums(g, w, _null);
                         a = (_theta[w] - st)/2;
                         b = mt - _mu[w];
                         auto lZ_w = (b * b)/(4 * a) - log(a)/2; //+ log(M_PI)/2;
                         return lZ_w - lZ_e;
                     };

                 if (!_frozen[u])
                     lZ -= get_lZ(u);
                 else if (!_frozen[v])
                     lZ -= get_lZ(v);
             });

        return lZ;
    }

    template <class Graph, class VMap>
    double energy(Graph& g, VMap s)
    {
        double H = 0;
        #pragma omp parallel reduction(+:H)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 if (_frozen[v])
                     return;
                 auto sv = s[v];
                 H += (_theta[v] * sv * sv) / 2 - _mu[v] * sv;

             });

        #pragma omp parallel reduction(+:H)
        parallel_edge_loop_no_spawn
            (g,
             [&] (const auto& e)
             {
                 auto u = source(e, g);
                 auto v = target(e, g);
                 if (_frozen[u] && _frozen[v])
                     return;
                 H += _x[e] * s[u] * s[v];
             });

        return H;
    }

    template <class Graph, class VMap>
    double energies(Graph& g, VMap ss)
    {
        double H = 0;
        #pragma omp parallel reduction(+:H)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 if (_frozen[v])
                     return;
                 for (auto s : ss[v])
                     H += (_theta[v] * s * s) / 2 - _mu[v] * s;
             });

        #pragma omp parallel reduction(+:H)
        parallel_edge_loop_no_spawn
            (g,
             [&] (const auto& e)
             {
                 auto u = source(e, g);
                 auto v = target(e, g);
                 if (_frozen[u] && _frozen[v])
                     return;
                 auto& s_u = ss[u];
                 auto& s_v = ss[v];
                 auto xe = _x[e];
                 for (size_t m = 0; m < s_u.size(); ++m)
                     H += xe * s_u[m] * s_v[m];
             });

        return H;
    }

    template <class Graph, class VMap>
    double marginal_lprob(Graph& g, VMap s)
    {
        double L = 0;
        #pragma omp parallel reduction(+:L)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 if (_frozen[v])
                     return;
                 auto m_m = _vm_m[v];
                 auto m_s = _vm_s[v];
                 double a = (s[v] - m_m);
                 L += - a * a / (2 * m_s) - (log(m_s) + log(M_PI))/2;
             });
        return L;
    }

    template <class Graph, class VMap>
    double marginal_lprobs(Graph& g, VMap ss)
    {
        double L = 0;
        #pragma omp parallel reduction(+:L)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 if (_frozen[v])
                     return;
                 auto m_m = _vm_m[v];
                 auto m_s = _vm_s[v];
                 for (auto s : ss[v])
                 {
                     double a = (s - m_m);
                     L += - a * a / (2 * m_s) - (log(m_s) + log(M_PI))/2;
                 }
             });
        return L;
    }

    template <class Graph, class VMap, class RNG>
    void sample(Graph& g, VMap s, RNG& rng_)
    {
        parallel_rng<rng_t> prng(rng_);
        parallel_vertex_loop
            (g,
             [&] (auto v)
             {
                 auto& rng = prng.get(rng_);
                 std::normal_distribution<> mrand(_vm_m[v], sqrt(_vm_s[v]));
                 s[v] = mrand(rng);
             });
    }

private:
    emap_t _x;
    vmap_t _mu;
    vmap_t _theta;
    emmap_t _em_m;
    emmap_t _em_s;
    emmap_t _temp_m;
    emmap_t _temp_s;
    vmap_t _vm_m;
    vmap_t _vm_s;
    vfmap_t _frozen;
    constexpr static size_t _null = std::numeric_limits<size_t>::max();
};

} // namespace graph_tool

#endif // GRAPH_BP_HH
