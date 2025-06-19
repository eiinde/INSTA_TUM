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

#ifndef GRAPH_ASSORTATIVITY_HH
#define GRAPH_ASSORTATIVITY_HH

#include "shared_map.hh"
#include "graph_util.hh"
#include "hash_map_wrap.hh"

#if (BOOST_VERSION >= 106000)
# include <boost/math/special_functions/relative_difference.hpp>
#endif

namespace graph_tool
{
using namespace std;
using namespace boost;


// this will calculate the assortativity coefficient, based on the property
// pointed by 'deg'

struct get_assortativity_coefficient
{
    template <class Graph, class DegreeSelector, class Eweight>
    void operator()(const Graph& g, DegreeSelector deg, Eweight eweight,
                    double& r, double& r_err) const
    {
        typedef typename property_traits<Eweight>::value_type wval_t;
        wval_t n_edges = 0;
        wval_t e_kk = 0;

        typedef typename DegreeSelector::value_type val_t;
        typedef gt_hash_map<val_t, wval_t> map_t;
        map_t a, b;

        SharedMap<map_t> sa(a), sb(b);
        #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh()) \
            firstprivate(sa, sb) reduction(+:e_kk, n_edges)
        parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 val_t k1 = deg(v, g);
                 for (auto e : out_edges_range(v, g))
                 {
                     auto u = target(e, g);
                     auto w = eweight[e];
                     val_t k2 = deg(u, g);
                     if (k1 == k2)
                         e_kk += w;
                     sa[k1] += w;
                     sb[k2] += w;
                     n_edges += w;
                 }
             });

        sa.Gather();
        sb.Gather();

        double t1 = double(e_kk) / n_edges, t2 = 0.0;

        for (auto& ai : a)
        {
            auto bi = b.find(ai.first);
            if (bi != b.end())
                t2 += ai.second * bi->second;
        }
        t2 /= n_edges * n_edges;


#if (BOOST_VERSION >= 106000)
        if (boost::math::relative_difference(1., t2) > 1e-8)
#else
        if (abs(1.-t2) > 1e-8)
#endif
            r = (t1 - t2)/(1.0 - t2);
        else
            r = std::numeric_limits<double>::quiet_NaN();

        // "jackknife" variance
        double err = 0;
        size_t one = (graph_tool::is_directed(g)) ? 1 : 2;
        #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh()) \
            reduction(+:err)
        parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 val_t k1 = deg(v, g);
                 for (auto e : out_edges_range(v, g))
                 {
                     auto u = target(e, g);
                     auto w = eweight[e];
                     val_t k2 = deg(u, g);
                     double tl2 = (t2 * (n_edges * n_edges)
                                   - w * one * b[k1] - w * one * a[k2]) /
                         ((n_edges - w * one) * (n_edges - w * one));
                     double tl1 = t1 * n_edges;
                     if (k1 == k2)
                         tl1 -= one * w;
                     tl1 /= n_edges - one * w;
                     double rl = (tl1 - tl2) / (1.0 - tl2);
                     err += (r - rl) * (r - rl);
                }
             });
        if (!graph_tool::is_directed(g))
            err /= 2;
#if (BOOST_VERSION >= 106000)
        if (boost::math::relative_difference(1., t2) > 1e-8)
#else
        if (abs(1.-t2) > 1e-8)
#endif
            r_err = sqrt(err);
        else
            r_err = std::numeric_limits<double>::quiet_NaN();
    }
};

// this will calculate the _scalar_ assortativity coefficient, based on the
// scalar property pointed by 'deg'

struct get_scalar_assortativity_coefficient
{
    template <class Graph, class DegreeSelector, class Eweight>
    void operator()(const Graph& g, DegreeSelector deg, Eweight eweight,
                    double& r, double& r_err) const
    {
        typedef typename property_traits<Eweight>::value_type val_t;
        val_t n_edges = 0;
        double e_xy = 0;
        double a = 0, b = 0, da = 0, db = 0;

        #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh()) \
            reduction(+:e_xy,n_edges,a,b,da,db)
        parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 auto k1 = deg(v, g);
                 for (auto e : out_edges_range(v, g))
                 {
                     auto u = target(e, g);
                     auto w = eweight[e];
                     auto k2 = deg(u, g);
                     a += k1 * w;
                     da += k1 * k1 * w;
                     b += k2 * w;
                     db += k2 * k2 * w;
                     e_xy += k1 * k2 * w;
                     n_edges += w;
                 }
             });

        double t1 = e_xy/n_edges;
        a /= n_edges;
        b /= n_edges;
        double stda;
        double stdb;

#if (BOOST_VERSION >= 106000)
        if (boost::math::relative_difference(da/n_edges, a*a) < 1e-8)
            stda = 0;
        else
            stda = sqrt(da/n_edges - a*a);
        if (boost::math::relative_difference(db/n_edges, b*b) < 1e-8)
            stdb = 0;
        else
            stdb = sqrt(db/n_edges - b*b);
#else
        if (sqrt(abs(da/n_edges - a*a)) < 1e-8)
            stda = 0;
        else
            stda = sqrt(da/n_edges - a*a);
        if (sqrt(abs(db/n_edges - b*b)) < 1e-8)
            stdb = 0;
        else
            stdb = sqrt(db/n_edges - b*b);
#endif

        if (stda*stdb > 0)
            r = (t1 - a*b)/(stda*stdb);
        else
            r = std::numeric_limits<double>::quiet_NaN();

        // "jackknife" variance
        r_err = 0.0;

        double err = 0.0;
        size_t one = (graph_tool::is_directed(g)) ? 1 : 2;
        #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh()) \
            reduction(+:err)
        parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 double k1 = double(deg(v, g));
                 double al = (a * n_edges - k1) / (n_edges - one);
                 double dal = sqrt((da - k1 * k1) / (n_edges - one) - al * al);

                 for (auto e : out_edges_range(v, g))
                 {
                     auto u = target(e, g);
                     auto w = eweight[e];
                     double k2 = deg(u, g);
                     double bl = (b * n_edges - k2 * one * w) / (n_edges - one * w);
                     double dbl = sqrt((db - k2 * k2 * one * w) / (n_edges - one * w) - bl * bl);
                     double t1l = (e_xy - k1 * k2 * one * w)/(n_edges - one * w);
                     double rl;
                     if (dal * dbl > 0)
                         rl = (t1l - al * bl)/(dal * dbl);
                     else
                         rl = (t1l - al * bl);
                     err += (r - rl) * (r - rl);
                 }
             });
        if (!graph_tool::is_directed(g))
            err /= 2;
        if (stda*stdb > 0)
            r_err = sqrt(err);
        else
            r_err = std::numeric_limits<double>::quiet_NaN();
    }
};

} // graph_tool namespace

#endif //GRAPH_ASSORTATIVITY_HH
