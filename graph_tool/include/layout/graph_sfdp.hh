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

#ifndef GRAPH_FDP_HH
#define GRAPH_FDP_HH

#include <limits>
#include <iostream>

#include "idx_map.hh"

#include "../generation/sampler.hh"
#include "../inference/support/util.hh"

#include "quad_tree.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class Pos1, class Pos2>
double dist(const Pos1& p1, const Pos2& p2)
{
    double r = 0;
    for (size_t i = 0; i < 2; ++i)
        r += pow2(double(p1[i] - p2[i]));
    return sqrt(r);
}

template <class Pos1, class Pos2>
double f_r(double C, double K, double p, const Pos1& p1, const Pos2& p2)
{
    double d = dist(p1, p2);
    if (d == 0)
        return 0;
    return -C * pow(K, 1 + p) / pow(d, p);
}

template <class Pos1, class Pos2>
double f_a(double K, const Pos1& p1, const Pos2& p2)
{
    return pow2(dist(p1, p2)) / K;
}

template <class Pos1, class Pos2, class Pos3>
double get_diff(const Pos1& p1, const Pos2& p2, Pos3& r)
{
    double abs = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        r[i] = p1[i] - p2[i];
        abs += pow2(r[i]);
    }
    abs = sqrt(abs);
    if (abs > 0)
    {
        for (size_t i = 0; i < 2; ++i)
            r[i] /= abs;
    }
    return abs;
}

template <class Pos>
double norm(const Pos& x)
{
    double abs = 0;
    for (size_t i = 0; i < 2; ++i)
        abs += pow2(x[i]);
    return sqrt(abs);
}

template <class Graph, class PosMap, class VertexWeightMap,
          class EdgeWeightMap, class PinMap, class GroupMaps, class CMap,
          class OrderMap,
          class RNG>
void get_sfdp_layout(Graph& g, PosMap pos, VertexWeightMap vweight,
                     EdgeWeightMap eweight, PinMap pin, GroupMaps& groups,
                     double C, double K, double p, double theta,
                     std::vector<double> gamma, double r, size_t kc, CMap c, double R,
                     OrderMap yorder, double init_step, double step_schedule,
                     size_t max_level, double epsilon, size_t max_iter,
                     bool simple, bool verbose, RNG& rng)
{
    typedef typename property_traits<PosMap>::value_type::value_type val_t;
    typedef std::array<val_t, 2> pos_t;

    typedef typename property_traits<VertexWeightMap>::value_type vweight_t;

    vector<size_t> vertices;

    double ocenter = 0;
    double omax = -std::numeric_limits<double>::max();
    double omin = std::numeric_limits<double>::max();

    int HN = 0;
    vweight_t W = 0;
    for (auto v : vertices_range(g))
    {
        if (pin[v] == 0)
            vertices.push_back(v);
        pos[v].resize(2, 0);
        HN++;

        ocenter += yorder[v] * get(vweight, v);
        omax = max(yorder[v], omax);
        omin = min(yorder[v], omin);
        W += get(vweight, v);
    }

    ocenter /= W;
    double o_rg = omax - omin;

    val_t delta = epsilon * K + 1, E = 0, E0;
    E0 = numeric_limits<val_t>::max();
    size_t n_iter = 0;
    val_t step = init_step;
    size_t progress = 0;

    std::vector<idx_map<size_t, vector<size_t>>> rvs(groups.size());
    std::vector<idx_map<size_t, pos_t>> rcm(groups.size());
    std::vector<idx_map<size_t, pos_t>> rftot(groups.size());

    vector<size_t> cs;
    vector<double> cprobs;
    for (auto v : vertices_range(g))
    {
        size_t r = c[v];
        if (r >= cprobs.size())
            cprobs.resize(r + 1, 0);
        cprobs[r] += 1;

        r = v;
        for (size_t l = 0; l < groups.size(); ++l)
        {
            r = groups[l][r];
            rvs[l][r].push_back(v);
        }
    }
    for (size_t r = 0; r < cprobs.size(); ++r)
        cs.push_back(r);
    Sampler<size_t> csample(cs, cprobs);

    vector<pos_t> ccm;
    vector<size_t> csize;

    vector<pos_t> ftots(num_vertices(g));

    while (delta > epsilon * K && (max_iter == 0 || n_iter < max_iter))
    {
        delta = 0;
        E0 = E;
        E = 0;

        pos_t ll{numeric_limits<val_t>::max(), numeric_limits<val_t>::max()},
            ur{-numeric_limits<val_t>::max(), -numeric_limits<val_t>::max()};

        ccm.clear();
        csize.clear();

        double ycenter = 0;
        val_t ymax = -std::numeric_limits<val_t>::max();
        val_t ymin = std::numeric_limits<val_t>::max();

        for (auto v : vertices_range(g))
        {
            for (size_t j = 0; j < 2; ++j)
            {
                ll[j] = min(pos[v][j], ll[j]);
                ur[j] = max(pos[v][j], ur[j]);
            }

            size_t s = c[v];
            if (s >= ccm.size())
            {
                ccm.resize(s + 1);
                csize.resize(s + 1, 0);
            }

            csize[s] += get(vweight, v);
            for (size_t j = 0; j < 2; ++j)
                ccm[s][j] += pos[v][j] * get(vweight, v);

            ycenter += pos[v][1] * get(vweight, v);
            ymin = min(ymin, pos[v][1]);
            ymax = max(ymax, pos[v][1]);
        }
        ycenter /= W;
        double y_rg = ymax - ymin;

        for (size_t s = 0; s < ccm.size(); ++s)
        {
            if (csize[s] == 0)
                continue;
            for (size_t j = 0; j < 2; ++j)
                ccm[s][j] /= csize[s];
        }

        for (size_t l = 0; l < groups.size(); ++l)
        {
            rcm[l].clear();
            rftot[l].clear();
            for (auto& [r, vs] : rvs[l])
            {
                auto& cm = rcm[l][r];
                for (auto v : vs)
                {
                    for (size_t j = 0; j < 2; ++j)
                        cm[j] += pos[v][j];
                }
                for (size_t j = 0; j < 2; ++j)
                    cm[j] /= vs.size();
                rftot[l][r] = {0, 0};
            }
        }

        QuadTree<val_t, vweight_t> qt(ll, ur, max_level, num_vertices(g));
        for (auto v : vertices_range(g))
            qt.put_pos(0, pos[v], vweight[v]);

        size_t nmoves = 0;
        vector<size_t> Q;
        Q.reserve(num_vertices(g));

        size_t nopen = 0;

        auto get_rf_bh =
            [&](size_t v, auto& qt, auto& Q, auto&& f)
            {
                pos_t cm{0, 0}, diff{0, 0};

                Q.push_back(0);
                while (!Q.empty())
                {
                    size_t q = Q.back();
                    Q.pop_back();

                    auto& dleaves = qt.get_dense_leaves(q);
                    if (!dleaves.empty())
                    {
                        for (auto& dleaf : dleaves)
                        {
                            f(get<0>(dleaf), get<1>(dleaf));
                            nopen++;
                        }
                    }
                    else
                    {
                        double w = qt[q].get_w();
                        qt[q].get_cm(cm);
                        double d = get_diff(cm, pos[v], diff);
                        if (w > theta * d)
                        {
                            auto leaf = qt.get_leaves(q);
                            for (size_t i = 0; i < 4; ++i)
                            {
                                if (qt[leaf].get_count() > 0)
                                    Q.push_back(leaf);
                                ++leaf;
                            }
                        }
                        else
                        {
                            f(cm, qt[q].get_count());
                            nopen++;
                        }
                    }
                }
            };

        double adist = 0;

        #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh())   \
            firstprivate(Q) reduction(+:adist)
        parallel_loop_no_spawn
            (vertices,
             [&](size_t, auto v)
             {
                 pos_t diff{0, 0};
                 auto& ftot = ftots[v];
                 ftot = {0, 0};

                 auto& pos_v = pos[v];

                 // global repulsive forces
                 get_rf_bh(v, qt, Q,
                           [&](auto& lpos, auto w)
                           {
                               auto f = f_r(C, K, p, pos_v, lpos);
                               if (f == 0)
                                   return;
                               f *= w * get(vweight, v);
                               auto d = get_diff(lpos, pos_v, diff);
                               for (size_t l = 0; l < 2; ++l)
                                   ftot[l] += f * diff[l];
                               adist += d * w;
                           });

                 // local attractive forces
                 for (auto e : out_edges_range(v, g))
                 {
                     auto u = target(e, g);
                     if (u == v)
                         continue;
                     auto& pos_u = pos[u];
                     get_diff(pos_u, pos_v, diff);
                     val_t f = f_a(K, pos_u, pos_v);
                     f *= get(eweight, e) * get(vweight, u) * get(vweight, v);
                     for (size_t l = 0; l < 2; ++l)
                         ftot[l] += f * diff[l];
                 }

                 // inter-component attractive forces
                 if (r > 0 && ccm.size() > 1)
                 {
                     for (size_t i = 0; i < std::min(kc, ccm.size() - 1); ++i)
                     {
                         auto s = csample(rng);
                         if (csize[s] == 0)
                             continue;
                         if (s == size_t(c[v]))
                             continue;
                         val_t d = get_diff(ccm[s], pos_v, diff);
                         if (d == 0)
                             continue;
                         double Kp = K * pow2(HN);
                         val_t f = f_a(Kp, ccm[s], pos_v) * r * csize[s] * get(vweight, v);
                         for (size_t l = 0; l < 2; ++l)
                             ftot[l] += f * diff[l];
                     }
                 }

                 // group average forces
                 size_t s = v;
                 for (size_t l = 0; l < groups.size(); ++l)
                 {
                     s = groups[l][s];
                     for (size_t j = 0; j < 2; ++j)
                         rftot[l][s][j] += ftot[j];
                 }
             });

        adist /= vertices.size();
        for (size_t l = 0; l < groups.size(); ++l)
        {
            for (auto& [r, vs] : rvs[l])
            {
                for (size_t j = 0; j < 2; ++j)
                    rftot[l][r][j] /= vs.size();
            }
        }

        #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh())   \
            reduction(+:E, delta, nmoves)
        parallel_loop_no_spawn
            (vertices,
             [&](size_t, auto v)
             {
                 auto& ftot = ftots[v];
                 auto& pos_v = pos[v];

                 // group segregation force
                 pos_t diff;
                 size_t s = v;
                 for (size_t l = 0; l < groups.size(); ++l)
                 {
                     s = groups[l][s];
                     get_diff(rcm[l][s], pos_v, diff);
                     for (size_t j = 0; j < 2; ++j)
                         ftot[j] += diff[j] * gamma[l] * adist;

                     // coherent move
                     for (size_t j = 0; j < 2; ++j)
                         ftot[j] += rftot[l][s][j] * 10;
                 }

                 // yorder repulsive force
                 if (R > 0)
                 {
                     double dz = (yorder[v] - ocenter)/o_rg;
                     double dp = (pos_v[1] - ycenter)/y_rg;
                     ftot[1] += adist * R * (dz - dp);
                 }

                 double n = norm(ftot);

                 for (size_t l = 0; l < 2; ++l)
                     pos[v][l] += (ftot[l] / n) * step;

                 E += pow2(n);
                 delta += step;
                 nmoves++;
             });

        n_iter++;
        delta /= nmoves;

        if (verbose)
            cout << n_iter << " " << E << " " << step << " "
                 << delta << " " << max_level << " "
                 << nopen / double(HN) << endl;

        if (simple)
        {
            step *= step_schedule;
        }
        else
        {
            if (E < E0)
            {
                ++progress;
                if (progress >= 5)
                {
                    progress = 0;
                    step /= step_schedule;
                }
            }
            else
            {
                progress = 0;
                step *= step_schedule;
            }
        }
    }
}

} // namespace graph_tool


#endif // GRAPH_FDP_HH
