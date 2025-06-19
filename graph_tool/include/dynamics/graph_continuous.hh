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

#ifndef GRAPH_CONTINUOUS_HH
#define GRAPH_CONTINUOUS_HH

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "random.hh"
#include "parallel_rng.hh"
#include "idx_map.hh"

namespace graph_tool
{
using namespace boost;

template <class Value = double>
class continuous_state_base
{
public:
    typedef Value s_t;
    typedef typename vprop_map_t<s_t>::type::unchecked_t smap_t;

    continuous_state_base(smap_t s, smap_t s_diff)
        : _s(s), _s_diff(s_diff) {}

    template <class Graph, class RNG>
    double get_node_diff(Graph&, size_t, RNG&) { return 0.; }

    smap_t _s;
    smap_t _s_diff;
};

class linear_state: public continuous_state_base<>
{
public:

    typedef vprop_map_t<double>::type::unchecked_t vmap_t;
    typedef eprop_map_t<double>::type::unchecked_t wmap_t;

    template <class Graph, class RNG>
    linear_state(Graph&, smap_t s, smap_t s_diff, python::dict params, RNG&)
        : continuous_state_base(s, s_diff),
          _sigma(any_cast<vmap_t::checked_t>(python::extract<any>(params["sigma"].attr("_get_any")())()).get_unchecked()),
          _w(any_cast<wmap_t::checked_t>(python::extract<any>(params["w"].attr("_get_any")())()).get_unchecked())
    {}


    template <class Graph, class RNG>
    double get_node_diff(Graph& g, size_t v, double, double dt, RNG& rng)
    {
        double diff = 0;
        for (auto e : in_or_out_edges_range(v, g))
        {
            auto u = source(e, g);
            diff += _w[e] * _s[u];
        }

        if (_sigma[v] > 0)
        {
            std::normal_distribution<> noise(0, sqrt(dt));
            diff += _sigma[v] * noise(rng);
        }

        return diff;
    }

private:

    vmap_t _sigma;
    wmap_t _w;
};

class LV_state: public continuous_state_base<>
{
public:

    typedef vprop_map_t<double>::type::unchecked_t vmap_t;
    typedef eprop_map_t<double>::type::unchecked_t wmap_t;

    template <class Graph, class RNG>
    LV_state(Graph&, smap_t s, smap_t s_diff, python::dict params, RNG&)
        : continuous_state_base(s, s_diff),
          _sigma(any_cast<vmap_t::checked_t>(python::extract<any>(params["sigma"].attr("_get_any")())()).get_unchecked()),
          _mig(any_cast<vmap_t::checked_t>(python::extract<any>(params["mig"].attr("_get_any")())()).get_unchecked()),
          _r(any_cast<vmap_t::checked_t>(python::extract<any>(params["r"].attr("_get_any")())()).get_unchecked()),
          _w(any_cast<wmap_t::checked_t>(python::extract<any>(params["w"].attr("_get_any")())()).get_unchecked())
    {}

    template <class Graph, class RNG>
    double get_node_diff(Graph& g, size_t v, double, double dt, RNG& rng)
    {
        double diff = _r[v];
        for (auto e : in_or_out_edges_range(v, g))
        {
            auto u = source(e, g);
            diff += _w[e] * _s[u];
        }
        diff *= _s[v];

        if (_sigma[v] > 0)
        {
            std::normal_distribution<> noise(0, sqrt(dt));
            diff += sqrt(_s[v]) * _sigma[v] * noise(rng);
        }

        diff += _mig[v];

        return diff;
    }

private:

    vmap_t _sigma;
    vmap_t _mig;
    vmap_t _r;
    wmap_t _w;
};

class kuramoto_state: public continuous_state_base<>
{
public:

    typedef vprop_map_t<double>::type::unchecked_t omap_t;
    typedef eprop_map_t<double>::type::unchecked_t wmap_t;

    template <class Graph, class RNG>
    kuramoto_state(Graph&, smap_t s, smap_t s_diff, python::dict params, RNG&)
        : continuous_state_base(s, s_diff),
          _omega(any_cast<omap_t::checked_t>(python::extract<any>(params["omega"].attr("_get_any")())()).get_unchecked()),
          _w(any_cast<wmap_t::checked_t>(python::extract<any>(params["w"].attr("_get_any")())()).get_unchecked()),
          _sigma(any_cast<omap_t::checked_t>(python::extract<any>(params["sigma"].attr("_get_any")())()).get_unchecked())
    {}

    template <class Graph, class RNG>
    double get_node_diff(Graph& g, size_t v, double, double dt, RNG& rng)
    {
        double diff = _omega[v];
        auto sv = _s[v];
        for (auto e : in_or_out_edges_range(v, g))
        {
            auto u = source(e, g);
            diff += _w[e] * std::sin(_s[u] - sv);
        }

        if (_sigma[v] > 0)
        {
            std::normal_distribution<> noise(0, sqrt(dt));
            diff += _sigma[v] * noise(rng);
        }

        return diff;
    }

private:

    omap_t _omega;
    wmap_t _w;
    omap_t _sigma;
};

template <class Graph, class State, class RNG>
void get_diff_sync(Graph& g, State state, double t, double dt, RNG& rng_)
{
    parallel_rng<rng_t> prng(rng_);
    parallel_vertex_loop
        (g,
         [&] (auto v)
         {
             auto& rng = prng.get(rng_);
             state._s_diff[v] = state.get_node_diff(g, v, t, dt, rng);
         });

}


} // namespace graph_tool

#endif // GRAPH_CONTINUOUS_HH
