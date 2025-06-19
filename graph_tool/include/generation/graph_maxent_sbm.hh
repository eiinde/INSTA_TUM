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

#ifndef GRAPH_MAXENT_SBM_HH
#define GRAPH_MAXENT_SBM_HH

#include <tuple>

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "sampler.hh"

#include "random.hh"

#include "hash_map_wrap.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;


class SBMFugacities
{
public:

    template <class IVec, class FVec, class Vec, class Bvec>
    SBMFugacities(IVec&& rs, IVec&& ss, FVec&& ers, Vec&& degs_in, Vec&& degs_out,
                  Bvec&& b, bool directed, bool multigraph, bool self_loops)
        : _directed(directed), _multigraph(multigraph), _self_loops(self_loops)
    {

        size_t N = degs_in.size();
        double E2 = 0;
        _B = *std::max_element(b.begin(), b.end()) + 1;
        for (size_t i = 0; i < rs.size(); ++i)
        {
            size_t r = rs[i];
            size_t s = ss[i];
            if (r >= _mrs.size())
            {
                _mrs.resize(r + 1);
                _ers.resize(r + 1);
            }
            if (s >= _msr.size())
                _msr.resize(s + 1);
            E2 += _mrs[r][s] = _msr[s][r] = _ers[r][s] = ers[i];
            _B = std::max(_B, std::max(r, s) + 1);
        }

        _mrs.resize(_B);
        _msr.resize(_B);
        _ers.resize(_B);

        if (multigraph)
        {
            for (auto& mr : _mrs)
                for (auto& m : mr)
                    m.second /= E2;
            for (auto& mr : _msr)
                for (auto& m : mr)
                    m.second /= E2;
        }

        std::vector<gt_hash_map<double, size_t>> vertices_in(_B),
            vertices_out(_B);
        std::vector<size_t> nr(_B);

        for (size_t i = 0; i < N; ++i)
        {
            vertices_in[b[i]][degs_in[i]]++;
            vertices_out[b[i]][degs_out[i]]++;
            nr[b[i]]++;
        }

        for (size_t r = 0; r < _B; ++r)
        {
            _rtheta_in.emplace_back();
            _rdegs_in.emplace_back();
            _in_pos.emplace_back();

            double S = 0;
            for (auto& rc : vertices_in[r])
            {
                double t = rc.first;
                _in_pos[r][t] = _rtheta_in[r].size();
                if (multigraph && t > sqrt(nr[r]-1))
                    t = sqrt(nr[r]-1);
                _rtheta_in[r].emplace_back(t, rc.second);
                _rdegs_in[r].push_back(rc.first);
                S += t * rc.second;
            }
            if (multigraph)
            {
                for (auto& rc : _rtheta_in[r])
                    rc.first /= sqrt(S);
            }
            else
            {
                for (auto& rc : _rtheta_in[r])
                    rc.first /= S;
            }

            _rtheta_out.emplace_back();
            _rdegs_out.emplace_back();
            _out_pos.emplace_back();
            S = 0;
            for (auto rc : vertices_out[r])
            {
                double t = rc.first;
                _out_pos[r][t] = _rtheta_out[r].size();
                if (multigraph && t > sqrt(nr[r]-1))
                    t = sqrt(nr[r]-1);
                _rtheta_out[r].emplace_back(t, rc.second);
                _rdegs_out[r].push_back(rc.first);
                S += t * rc.second;
            }
            if (multigraph)
            {
                for (auto& rc : _rtheta_out[r])
                    rc.first /= sqrt(S);
            }
            else
            {
                for (auto& rc : _rtheta_out[r])
                    rc.first /= sqrt(S);
            }
        }
    }

    void norm()
    {
        std::vector<double> t_in(_B), t_out(_B);
        for (size_t r = 0; r < _B; ++r)
        {
            t_in[r] = 0;
            for (auto& rc : _rtheta_in[r])
                t_in[r] += rc.first * rc.second;
            for (auto& rc : _rtheta_in[r])
                rc.first /= t_in[r];
            t_out[r] = 0;
            for (auto& rc : _rtheta_out[r])
                t_out[r] += rc.first * rc.second;
            for (auto& rc : _rtheta_out[r])
                rc.first /= t_out[r];
        }
    }

    void pack(std::vector<double>& x)
    {
        x.clear();
        for (size_t r = 0; r < _B; ++r)
        {
            for (auto& rc : _rtheta_out[r])
                x.push_back(rc.first);
            if (_directed)
            {
                for (auto& rc : _rtheta_in[r])
                    x.push_back(rc.first);
            }
        }

        for (size_t r = 0; r < _B; ++r)
        {
            for (auto& m : _mrs[r])
            {
                auto s = m.first;
                if (!_directed && r > s)
                    continue;
                x.push_back(m.second);
            }
        }
    }

    void unpack(std::vector<double>& x)
    {
        size_t pos = 0;
        for (size_t r = 0; r < _B; ++r)
        {
            for (auto& rc : _rtheta_out[r])
                rc.first = x[pos++];
            if (_directed)
            {
                for (auto& rc : _rtheta_in[r])
                    rc.first = x[pos++];
            }
            else
            {
                _rtheta_in[r] = _rtheta_out[r];
            }
        }

        for (size_t r = 0; r < _B; ++r)
            for (auto& m : _mrs[r])
            {
                auto s = m.first;
                if (!_directed && r > s)
                    continue;
                m.second = x[pos++];
            }

        if (!_directed)
        {
            for (size_t r = 0; r < _B; ++r)
                for (auto& m : _mrs[r])
                {
                    auto s = m.first;
                    if (r > s)
                        m.second = _mrs[s][r];
                }
        }

        for (size_t r = 0; r < _B; ++r)
            for (auto& m : _mrs[r])
                _msr[m.first][r] = m.second;
    }

    double get_f()
    {
        double L = 0;

        for (size_t r = 0; r < _B; ++r)
        {
            if (_directed)
            {
                for (size_t i = 0; i < _rtheta_in[r].size(); ++i)
                    L += _rdegs_in[r][i] * log(_rtheta_in[r][i].first) * double(_rtheta_in[r][i].second);
            }
            for (size_t i = 0; i < _rtheta_out[r].size(); ++i)
                L += _rdegs_out[r][i] * log(_rtheta_out[r][i].first) * double(_rtheta_out[r][i].second);

            for (auto& m : _mrs[r])
            {
                auto s = m.first;
                L += _ers[r][s] * log(m.second) / (_directed ? 1. : 2.);

                for (size_t i = 0; i < _rtheta_out[r].size(); ++i)
                {
                    double tout = _rtheta_out[r][i].first;
                    double nout = _rtheta_out[r][i].second;

                    for (auto& tc : _rtheta_in[s])
                    {
                        double n = nout * ((s != r || tout != tc.first) ? tc.second : tc.second - 1);
                        if (!_directed)
                            n /= 2;
                        if (_self_loops && !_directed && s == r && tout == tc.first)
                            n += tc.second;
                        double p = tout * tc.first * _mrs[r][s];
                        if (!_multigraph)
                            L -= log1p(p) * n;
                        else
                            L += log1p(-p) * n;
                    }
                }
            }
        }

        return L;
    }

    void get_diff(std::vector<double>& x)
    {
        std::vector<std::vector<std::pair<double, size_t>>> diff_rtheta_in(_rtheta_in),
            diff_rtheta_out(_rtheta_out);

        vector<gt_hash_map<size_t, double>> diff_mrs(_B);

        for (size_t r = 0; r < _B; ++r)
        {
            for (size_t i = 0; i < _rtheta_in[r].size(); ++i)
            {
                double tin = _rtheta_in[r][i].first;
                double S = 0;
                for (auto& m : _msr[r])
                {
                    auto s = m.first;
                    for (auto& tc : _rtheta_out[s])
                    {
                        double p;
                        if (!_multigraph)
                            p = (tc.first * m.second) / (1. + tin * tc.first * m.second);
                        else
                            p = (tc.first * m.second) / (1. - tin * tc.first * m.second);

                        double n;
                        if (s == r && tin == tc.first)
                            n = (!_self_loops) ? tc.second - 1 : (_directed ? tc.second : tc.second + 1);
                        else
                            n = tc.second;
                        S += p * n;
                    }
                }
                diff_rtheta_in[r][i].first =  _rdegs_in[r][i] / tin - S;
            }

            for (size_t i = 0; i < _rtheta_out[r].size(); ++i)
            {
                double tout = _rtheta_out[r][i].first;
                double S = 0;
                for (auto& m : _mrs[r])
                {
                    auto s = m.first;
                    for (auto& tc : _rtheta_in[s])
                    {
                        double p;
                        if (!_multigraph)
                            p = (tc.first * m.second) / (1. + tout * tc.first * m.second);
                        else
                            p = (tc.first * m.second) / (1. - tout * tc.first * m.second);

                        double n;
                        if (s == r && tout == tc.first)
                            n = (!_self_loops) ? tc.second - 1 : (_directed ? tc.second : tc.second + 1);
                        else
                            n = tc.second;

                        S += p * n;
                    }
                }
                diff_rtheta_out[r][i].first = _rdegs_out[r][i] / tout - S;
            }
        }

        for (size_t r = 0; r < _B; ++r)
        {
            for (auto& m : _mrs[r])
            {
                auto s = m.first;
                double S = 0;
                for (auto& tc_out : _rtheta_out[r])
                    for (auto& tc_in : _rtheta_in[s])
                    {
                        double n = (r == s && tc_out.first == tc_in.first
                                    && !_self_loops) ?
                            tc_out.second * (tc_in.second - 1) :
                            tc_out.second * tc_in.second;
                        if (_self_loops && !_directed && s == r &&
                            tc_out.first == tc_in.first)
                            n += tc_in.second;
                        double p = tc_out.first * tc_in.first;
                        if (!_multigraph)
                            p = p * n / (1. + p * m.second);
                        else
                            p = p * n / (1. - p * m.second);
                        S += p;
                    }
                diff_mrs[r][s] = _ers[r][s] / m.second - S;
                if (r == s)
                    diff_mrs[r][s] /= 2;
            }
        }

        x.clear();
        for (size_t r = 0; r < _B; ++r)
        {
            for (auto& rc : diff_rtheta_out[r])
                x.push_back(rc.first);
            if (_directed)
            {
                for (auto& rc : diff_rtheta_in[r])
                    x.push_back(rc.first);
            }
        }

        for (size_t r = 0; r < _B; ++r)
        {
            for (auto& m : diff_mrs[r])
            {
                auto s = m.first;
                if (!_directed && r > s)
                    continue;
                x.push_back(m.second);
            }
        }
    }

    void new_x(std::vector<double>& x)
    {
        std::vector<std::vector<std::pair<double, size_t>>> new_rtheta_in(_rtheta_in),
            new_rtheta_out(_rtheta_out);

        std::vector<gt_hash_map<size_t, double>> new_mrs(_B);

        for (size_t r = 0; r < _B; ++r)
        {
            for (size_t i = 0; i < _rtheta_in[r].size(); ++i)
            {
                double tin = _rtheta_in[r][i].first;
                double S = 0;
                for (auto& m : _msr[r])
                {
                    auto s = m.first;
                    for (auto& tc : _rtheta_out[s])
                    {
                        double n = (s == r && tin == tc.first && !_self_loops)
                            ? tc.second - 1 : tc.second;
                        if (!_multigraph)
                        {
                            S += (tc.first * m.second * n) / (1. + tin * tc.first * m.second);
                        }
                        else
                        {
                            S += (tc.first * m.second * n) / (1. - tin * tc.first * m.second);
                        }
                    }
                }
                new_rtheta_in[r][i].first = _rdegs_in[r][i] / S;
            }

            for (size_t i = 0; i < _rtheta_out[r].size(); ++i)
            {
                double tout = _rtheta_out[r][i].first;
                double S = 0;
                for (auto& m : _mrs[r])
                {
                    auto s = m.first;
                    for (auto& tc : _rtheta_in[s])
                    {
                        double n = (s == r && tout == tc.first && !_self_loops)
                            ? tc.second - 1 : tc.second;
                        if (!_multigraph)
                            S += (tc.first * m.second * n) / (1. + tout * tc.first * m.second);
                        else
                            S += (tc.first * m.second * n) / (1. - tout * tc.first * m.second);
                    }
                }
                new_rtheta_out[r][i].first = _rdegs_out[r][i] / S;
            }
        }

        for (size_t r = 0; r < _B; ++r)
        {
            for (auto& m : _mrs[r])
            {
                auto s = m.first;
                double S = 0;
                for (auto& tc_out : _rtheta_out[r])
                    for (auto& tc_in : _rtheta_in[s])
                    {
                        double n = (r == s && tc_out.first == tc_in.first
                                    && !_self_loops) ?
                            tc_out.second * (tc_in.second - 1) :
                            tc_out.second * tc_in.second;
                        double p = tc_out.first * tc_in.first;
                        if (!_multigraph)
                            p = (p * n) / (1. + p * m.second);
                        else
                            p = (p * n) / (1. - p * m.second);
                        S += p;
                    }
                new_mrs[r][s] = _ers[r][s] / S;
            }
        }

        x.clear();
        for (size_t r = 0; r < _B; ++r)
        {
            for (auto& rc : new_rtheta_out[r])
                x.push_back(rc.first);
            if (_directed)
            {
                for (auto& rc : new_rtheta_in[r])
                    x.push_back(rc.first);
            }
        }

        for (size_t r = 0; r < _B; ++r)
        {
            for (auto& m : new_mrs[r])
            {
                auto s = m.first;
                if (!_directed && r > s)
                    continue;
                x.push_back(m.second);
            }
        }
    }

    template <class IVec, class MVec, class Vec, class Bvec>
    void export_args(IVec&& rs, IVec&& ss, MVec&& mrs, Vec&& degs_in,
                     Vec&& degs_out, Vec&& theta_in, Vec&& theta_out, Bvec&& b)
    {
        for (size_t i = 0; i < rs.size(); ++i)
        {
            auto r = rs[i];
            auto s = ss[i];
            mrs[i] = _mrs[r][s];
        }

        size_t N = theta_in.size();
        for (size_t i = 0; i < N; ++i)
        {
            theta_in[i] = _rtheta_in[b[i]][_in_pos[b[i]][degs_in[i]]].first;
            theta_out[i] = _rtheta_out[b[i]][_out_pos[b[i]][degs_out[i]]].first;
        }
    }

private:
    bool _directed;
    bool _multigraph;
    bool _self_loops;

    std::vector<std::vector<double>> _rdegs_in, _rdegs_out;
    std::vector<std::vector<std::pair<double, size_t>>> _rtheta_in, _rtheta_out;

    std::vector<gt_hash_map<double, size_t>> _in_pos, _out_pos;

    size_t _B;
    std::vector<gt_hash_map<size_t, double>> _ers, _mrs, _msr;
};

template <bool multigraph, class Graph, class VProp, class IVec, class MVec,
          class VDProp, class RNG>
void gen_maxent_sbm(Graph& g, VProp b, IVec&& rs, IVec&& ss, MVec& mrs,
                    VDProp theta_in, VDProp theta_out, bool self_loops,
                    RNG& rng)
{
    size_t B = std::max(*std::max_element(rs.begin(), rs.end()),
                        *std::max_element(ss.begin(), ss.end())) + 1;

    std::vector<gt_hash_map<double, std::vector<size_t>>> vertices_in(B),
        vertices_out(B);
    gt_hash_set<std::tuple<double, double>> theta_l;

    for (auto v : vertices_range(g))
    {
        auto r = b[v];
        if (theta_in[v] > 0)
            vertices_in[r][theta_in[v]].push_back(v);
        if (theta_out[v] > 0)
            vertices_out[r][theta_out[v]].push_back(v);
        theta_l.insert({theta_out[v], theta_in[v]});
    }

    gt_hash_set<std::pair<size_t, size_t>> sampled;
    for (size_t i = 0; i < mrs.size(); ++i)
    {
        auto r = rs[i];
        auto s = ss[i];
        auto m = mrs[i];

        for (auto& vout : vertices_out[r])
        {
            for (auto& vin : vertices_in[s])
            {
                if (!graph_tool::is_directed(g) && r == s && vin.first > vout.first)
                    continue;

                double p = vout.first * vin.first * m;

                if (!multigraph)
                    p /= (1+p);
                else if (p >= 1)
                    throw GraphException("Invalid probability: " + lexical_cast<string>(p));

                size_t n;
                if (r == s && theta_l.find({vout.first, vin.first}) != theta_l.end())
                {
                    n = vout.second.size() * (vin.second.size() - 1);
                    if (!graph_tool::is_directed(g))
                        n /= 2;
                    if (self_loops)
                        n += vin.second.size();
                }
                else
                {
                    n = vout.second.size() * vin.second.size();
                }

                std::binomial_distribution<size_t> d(n, p);
                size_t nedges = d(rng);

                for (size_t i = 0; i < nedges; ++i)
                {
                    size_t u,v;
                    do
                    {
                         u = uniform_sample(vout.second, rng);
                         v = uniform_sample(vin.second, rng);
                         if (!graph_tool::is_directed(g) && u > v)
                             std::swap(u, v);
                    }
                    while ((u == v && !self_loops) ||
                           sampled.find({u,v}) != sampled.end());

                    add_edge(u, v, g);

                    if (multigraph)
                    {
                        std::bernoulli_distribution coin(p);
                        while (coin(rng))
                            add_edge(u, v, g);
                    }
                    sampled.insert({u,v});
                }
                sampled.clear();
            }
        }
    }
}


} // graph_tool namespace

#endif // GRAPH_MAXENT_SBM_HH
