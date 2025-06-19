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

class PottsBPState
{
public:

    typedef eprop_map_t<std::vector<double>>::type::unchecked_t emmap_t;
    typedef vprop_map_t<std::vector<double>>::type::unchecked_t vmmap_t;
    typedef eprop_map_t<double>::type::unchecked_t emap_t;
    typedef vprop_map_t<std::vector<double>>::type::unchecked_t vmap_t;
    typedef vprop_map_t<uint8_t>::type::unchecked_t vfmap_t;

    template <class Graph, class RNG>
    PottsBPState(Graph& g, boost::multi_array_ref<double, 2> f, emap_t x,
                 vmap_t theta, emmap_t em, vmmap_t vm, bool marginal_init,
                 vfmap_t frozen, RNG& rng)
        : _f(f), _x(x), _theta(theta), _em(em), _vm(vm), _q(f.shape()[0]),
          _frozen(frozen)
    {
        std::uniform_real_distribution<> rand;

        for (auto v : vertices_range(g))
        {
            if (_vm[v].empty())
            {
                for (size_t r = 0; r < _q; ++r)
                    _vm[v].push_back(log(rand(rng)));
            }
            _vm[v].resize(_q + 1);
            double lZ = _vm[v][_q] = log_Zm(_vm[v].begin());
            for (size_t r = 0; r < _q; ++r)
                _vm[v][r] -= lZ;
        }

        for (auto e : edges_range(g))
        {
            _em[e].resize(2 * (_q + 1));
            if (marginal_init)
            {
                auto u = source(e, g);
                auto v = target(e, g);
                auto m_uv = get_message(g, e, _em, u);
                auto m_vu = get_message(g, e, _em, v);
                for (size_t r = 0; r < _q + 1; ++r)
                {
                    *(m_uv + r) = _vm[v][r];
                    *(m_vu + r) = _vm[u][r];
                }
            }
            else
            {
                for (size_t r = 0; r < _q; ++r)
                {
                    _em[e][r] = log(rand(rng));
                    _em[e][r + _q + 1] = log(rand(rng));
                }
                double lZ;
                lZ = _em[e][_q] = log_Zm(_em[e].begin());
                for (size_t r = 0; r < _q; ++r)
                    _em[e][r] -= lZ;
                lZ = _em[e][2 * _q + 1] = log_Zm(_em[e].begin() + _q);
                for (size_t r = 0; r < _q; ++r)
                    _em[e][r + _q] -= lZ;
            }
        }

        _temp = _em.copy();
    };

    template <class Graph, class Edge, class ME>
    std::vector<double>::iterator get_message(Graph& g, const Edge& e, ME& me,
                                              size_t s)
    {
        auto u = source(e, g);
        auto v = target(e, g);
        if (u > v)
            std::swap(u, v);
        auto& m = me[e];
        if (s == u)
            return m.begin();
        else
            return m.begin() + _q + 1;
    }

    template <class Iter>
    double log_Zm(Iter iter)
    {
        double lZ = -_inf;
        for (size_t r = 0; r < _q; ++r)
            lZ = log_sum_exp(*(iter + r), lZ);
        return lZ;
    }

    template <class Graph, class Iter>
    double update_message(Graph& g, Iter m, size_t s, size_t t)
    {
        std::vector<double> nm(_q);
        for (size_t r = 0; r < _q; ++r)
        {
            nm[r] = -_theta[s][r];
            for (auto ue : out_edges_range(s, g))
            {
                auto u = target(ue, g);
                if (u == t)
                    continue;
                auto m_u = get_message(g, ue, _em, u);
                auto w = _x[ue];
                double temp = -_inf;
                auto fr = _f[r];
                for (size_t x = 0; x < _q; ++x)
                    temp = log_sum_exp(*(m_u + x) - w * fr[x], temp);
                nm[r] += temp;
            }
        }
        double lZ = log_Zm(nm.begin());
        double delta = 0;
        for (size_t r = 0; r < _q; ++r)
        {
            auto nm_r = nm[r] - lZ;
            auto& m_r =  *(m + r);
            delta += abs(nm_r - m_r);
            m_r = nm_r;
        }
        *(m + _q) = lZ;
        return delta;
    }

    template <class Graph, class Edge, class ME>
    double update_edge(Graph& g, const Edge& e, ME& me)
    {
        auto u = source(e, g);
        auto v = target(e, g);
        auto muv = get_message(g, e, me, u);
        auto mvu = get_message(g, e, me, v);
        double delta = 0;
        if (!_frozen[v])
            delta += update_message(g, muv, u, v);
        if (!_frozen[u])
            delta += update_message(g, mvu, v, u);
        return delta;
    }

    template <class Graph>
    void update_marginals(Graph& g)
    {
        parallel_vertex_loop
            (g,
             [&] (auto v)
             {
                 if (_frozen[v])
                     return;
                 auto m = _vm[v].begin();
                 update_message(g, m, v, _null);
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
                delta += update_edge(g, e, _em);
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
                     _temp[e] = _em[e];
                     delta += update_edge(g, e, _temp);
                 });

            #pragma omp parallel reduction(+:delta)
            parallel_edge_loop_no_spawn
                (g,
                 [&] (const auto& e)
                 {
                     _em[e] = _temp[e];
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
                 update_message(g, _vm[v].begin(), v, _null);
                 lZ += _vm[v][_q];
             });

        #pragma omp parallel reduction(+:lZ)
        parallel_edge_loop_no_spawn
            (g,
             [&] (const auto& e)
             {
                 auto u = source(e, g);
                 auto v = target(e, g);
                 if (!_frozen[u])
                 {
                     auto muv = get_message(g, e, _em, u);
                     lZ -= _vm[u][_q] - *(muv + _q);
                 }
                 else if (!_frozen[v])
                 {
                     auto mvu = get_message(g, e, _em, v);
                     lZ -= _vm[v][_q] - *(mvu + _q);
                 }
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
                 H += _theta[v][s[v]];
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
                 H += _x[e] * _f[s[u]][s[v]];
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
                     H += _theta[v][s];
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
                     H += xe * _f[s_u[m]][s_v[m]];
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
                 L += _vm[v][s[v]];
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
                 for (auto s : ss[v])
                     L += _vm[v][s];
             });
        return L;
    }

    template <class Graph, class VMap, class RNG>
    void sample(Graph& g, VMap s, RNG& rng_)
    {
        parallel_rng<rng_t> prng(rng_);

        std::vector<int> vals(_q);
        std::vector<double> probs(_q);
        for (size_t r = 0; r < _q; ++r)
            vals[r] = r;

        #pragma omp parallel firstprivate(probs)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 auto& rng = prng.get(rng_);
                 for (size_t r = 0; r < _q; ++r)
                     probs[r] = exp(_vm[v][r]);
                 Sampler<int> sampler(vals, probs);
                 s[v] = sampler(rng);
             });
    }

private:
    boost::multi_array_ref<double, 2> _f;
    emap_t _x;
    vmap_t _theta;
    emmap_t _em;
    emmap_t _temp;
    vmmap_t _vm;
    size_t _q;
    vfmap_t _frozen;
    constexpr static size_t _null = std::numeric_limits<size_t>::max();
    constexpr static double _inf = std::numeric_limits<double>::infinity();
};

} // namespace graph_tool

#endif // GRAPH_BP_HH
