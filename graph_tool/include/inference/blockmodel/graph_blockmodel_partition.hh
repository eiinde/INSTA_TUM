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

#ifndef GRAPH_BLOCKMODEL_PARTITION_HH
#define GRAPH_BLOCKMODEL_PARTITION_HH

#include "../support/util.hh"
#include "../support/int_part.hh"

#include "hash_map_wrap.hh"

#include <boost/range/counting_range.hpp>

namespace graph_tool
{

// ===============
// Partition stats
// ===============

constexpr size_t null_group = std::numeric_limits<size_t>::max();

typedef vprop_map_t<std::tuple<size_t, size_t>>::type degs_map_t;

struct simple_degs_t {};

template <class Graph, class Eprop>
[[gnu::always_inline]] [[gnu::flatten]] inline
std::tuple<size_t, size_t> get_deg(size_t v, Eprop& eweight,
                                   const simple_degs_t&, Graph& g)
{
    return {in_degreeS()(v, g, eweight), out_degreeS()(v, g, eweight)};
}

template <class Graph, class Eprop>
[[gnu::always_inline]] [[gnu::flatten]] inline
std::tuple<size_t, size_t>& get_deg(size_t v, Eprop&,
                                    const typename degs_map_t::unchecked_t& degs,
                                    Graph&)
{
    return degs[v];
}

template <bool use_rmap>
class partition_stats_base
{
public:

    typedef gt_hash_map<size_t, int> map_t;

    template <class Graph, class Vprop, class VWprop, class Eprop, class Degs,
              class Vlist>
    partition_stats_base(Graph& g, Vprop& b, Vlist&& vlist, size_t E, size_t B,
                         VWprop& vweight, Eprop& eweight, Degs& degs)
        : _directed(graph_tool::is_directed(g)), _N(0), _E(E), _total_B(B)
    {
        if constexpr (!use_rmap)
        {
            if (_directed)
                _hist_in.resize(B, nullptr);
            _hist_out.resize(B, nullptr);
            _total.resize(B);
            _ep.resize(B);
            _em.resize(B);
        }

        for (auto v : vlist)
        {
            if (vweight[v] == 0)
                continue;

            auto r = get_r(b[v]);

            auto [kin, kout] = get_deg(v, eweight, degs, g);
            auto n = vweight[v];
            if (_directed)
                get_hist<false>(r)[kin] += n;
            get_hist<true>(r)[kout] += n;
            _em[r] += kin * n;
            _ep[r] += kout * n;
            _total[r] += n;
            _N += n;
        }

        _actual_B = 0;
        for (auto n : _total)
        {
            if (n > 0)
                _actual_B++;
        }
    }

    ~partition_stats_base()
    {
        for (auto* h : _hist_in)
        {
            if (h != nullptr)
                delete h;
        }
        for (auto* h : _hist_out)
        {
            if (h != nullptr)
                delete h;
        }
    }

    template <bool out, bool create=true>
    auto& get_hist(size_t r)
    {
        auto& h = (out) ? _hist_out[r] : _hist_in[r];
        if (h == nullptr)
        {
            if constexpr (!create)
                return _dummy;
            h = new map_t();
        }
        return *h;
    };

    size_t get_r(size_t r)
    {
        if constexpr (use_rmap)
        {
            constexpr size_t null =
                std::numeric_limits<size_t>::max();
            if (r >= _bmap.size())
                _bmap.resize(r + 1, null);
            size_t nr = _bmap[r];
            if (nr == null)
                nr = _bmap[r] = _hist_out.size();
            r = nr;
        }
        if (r >= _hist_out.size())
        {
            if (_directed)
                _hist_in.resize(r + 1, nullptr);
            _hist_out.resize(r + 1, nullptr);
            _total.resize(r + 1);
            _ep.resize(r + 1);
            _em.resize(r + 1);
        }
        return r;
    }

    double get_partition_dl()
    {
        if (_N == 0)
            return 0;
        double S = 0;
        S += lbinom_fast(_N - 1, _actual_B - 1);
        S += lgamma_fast(_N + 1);
        for (auto nr : _total)
            S -= lgamma_fast(nr + 1);
        S += safelog_fast(_N);
        return S;
    }

    template <class Rs, class Kins, class Kouts>
    double get_deg_dl_ent(Rs&& rs, Kins&& kins, Kouts&& kouts)
    {
        double S = 0;
        for (auto r : rs)
        {
            r = get_r(r);
            size_t total = 0;
            if (kins.empty() && kouts.empty())
            {
                if (_directed)
                {
                    for (auto& k_c : get_hist<false, false>(r))
                        S -= xlogx_fast(k_c.second);
                }

                for (auto& k_c : get_hist<true, false>(r))
                {
                    S -= xlogx_fast(k_c.second);
                    total += k_c.second;
                }
            }
            else
            {
                auto& h_out = get_hist<true, false>(r);
                auto& h_in = (_directed) ? get_hist<false, false>(r) : h_out;

                if (_directed)
                {
                    for (auto& kin_d : kins)
                    {
                        auto& [kin, delta] = kin_d;
                        if (kin == numeric_limits<size_t>::max())
                            continue;
                        auto iter = h_in.find(kin);
                        auto k_c = (iter != h_in.end()) ? iter->second : 0;
                        S -= xlogx(k_c + delta);
                    }
                }

                for (auto& kout_d : kouts)
                {
                    auto& [kout, delta] = kout_d;
                    if (kout == numeric_limits<size_t>::max())
                        continue;
                    auto iter = h_out.find(kout);
                    auto k_c = (iter != h_out.end()) ? iter->second : 0;
                    S -= xlogx(k_c + delta);
                }

                total = _total[r];
            }
            if (_directed)
                S += 2 * xlogx_fast(total);
            else
                S += xlogx_fast(total);
        }
        return S;
    }

    template <class Rs, class Kins, class Kouts>
    double get_deg_dl_uniform(Rs&& rs, Kins&& kins, Kouts&& kouts)
    {
        double S = 0;
        for (auto r : rs)
        {
            r = get_r(r);
            if (kins.empty() && kouts.empty())
            {
                S += lbinom_fast(_total[r] + _ep[r] - 1, _ep[r]);
                if (_directed)
                    S += lbinom_fast(_total[r] + _em[r] - 1, _em[r]);
            }
            else
            {
                int dp = 0;
                int dm = 0;

                if (_directed)
                {
                    for (auto& kin_d : kins)
                    {
                        auto& [kin, delta] = kin_d;
                        if (kin == numeric_limits<size_t>::max())
                            continue;
                        dm +=  delta * kin;
                    }
                }

                for (auto& kout_d : kouts)
                {
                    auto& [kout, delta] = kout_d;
                    if (kout == numeric_limits<size_t>::max())
                        continue;
                    dp += delta * kout;
                }

                S += lbinom_fast(_total[r] + _ep[r] + dp - 1, _ep[r] + dp);
                if (_directed)
                    S += lbinom_fast(_total[r] + _em[r] + dm - 1, _em[r] + dm);
            }
        }
        return S;
    }

    template <class Rs, class Kins, class Kouts>
    double get_deg_dl_dist(Rs&& rs, Kins&& kins, Kouts&& kouts)
    {
        double S = 0;
        for (auto r : rs)
        {
            r = get_r(r);
            int dm = 0;
            int dp = 0;
            size_t total = 0;
            if (kins.empty() && kouts.empty())
            {
                if (_directed)
                {
                    for (auto& k_c : get_hist<false, false>(r))
                        S -= lgamma_fast(k_c.second + 1);
                }

                for (auto& k_c : get_hist<true, false>(r))
                {
                    S -= lgamma_fast(k_c.second + 1);
                    total += k_c.second;
                }
            }
            else
            {
                auto& h_out = get_hist<true, false>(r);
                auto& h_in = (_directed) ? get_hist<false, false>(r) : h_out;

                if (_directed)
                {
                    for (auto& kin_d : kins)
                    {
                        auto& [kin, delta] = kin_d;
                        if (kin == numeric_limits<size_t>::max())
                            continue;
                        auto iter = h_in.find(kin);
                        auto k_c = (iter != h_in.end()) ? iter->second : 0;
                        S -= lgamma_fast(k_c + delta + 1);
                        dm += delta * kin;
                    }
                }

                for (auto& kout_d : kouts)
                {
                    auto& [kout, delta] = kout_d;
                    if (kout == numeric_limits<size_t>::max())
                        continue;
                    auto iter = h_out.find(kout);
                    auto k_c = (iter != h_out.end()) ? iter->second : 0;
                    S -= lgamma_fast(k_c + delta + 1);
                    dp += delta * kout;
                }

                total = _total[r];
            }

            S += log_q(_ep[r] + dp, _total[r]);
            if (_directed)
                S += log_q(_em[r] + dm, _total[r]);

            if (_directed)
                S += 2 * lgamma_fast(total + 1);
            else
                S += lgamma_fast(total + 1);
        }
        return S;
    }

    template <class Rs, class Kins, class Kouts>
    double get_deg_dl(int kind, Rs&& rs, Kins&& kins, Kouts&& kouts)
    {
        if (_N == 0)
            return 0;

        switch (kind)
        {
        case deg_dl_kind::ENT:
            return get_deg_dl_ent(rs, kins, kouts);
        case deg_dl_kind::UNIFORM:
            return get_deg_dl_uniform(rs, kins, kouts);
        case deg_dl_kind::DIST:
            return get_deg_dl_dist(rs, kins, kouts);
        default:
            return numeric_limits<double>::quiet_NaN();
        }
    }

    double get_deg_dl(int kind)
    {
        return get_deg_dl(kind, boost::counting_range(size_t(0), _total_B),
                          std::array<std::pair<size_t,int>,0>(),
                          std::array<std::pair<size_t,int>,0>());
    }

    template <class Graph>
    double get_edges_dl(size_t B, Graph& g, int dE = 0)
    {
        size_t BB = (graph_tool::is_directed(g)) ? B * B : (B * (B + 1)) / 2;
        return lbinom_fast(BB + _E + dE - 1, _E + dE);
    }

    template <class VProp>
    double get_delta_partition_dl(size_t v, size_t r, size_t nr, VProp& vweight)
    {
        if (r == nr)
            return 0;

        if (r != null_group)
            r = get_r(r);

        if (nr != null_group)
            nr = get_r(nr);

        int n = vweight[v];
        if (n == 0)
        {
            if (r == null_group)
                n = 1;
            else
                return 0;
        }

        double S_b = 0, S_a = 0;

        if (r != null_group)
        {
            S_b += -lgamma_fast(_total[r] + 1);
            S_a += -lgamma_fast(_total[r] - n + 1);
        }

        if (nr != null_group)
        {
            S_b += -lgamma_fast(_total[nr] + 1);
            S_a += -lgamma_fast(_total[nr] + n + 1);
        }

        int dN = 0;
        if (r == null_group)
            dN += n;
        if (nr == null_group)
            dN -= n;

        S_b += lgamma_fast(_N + 1);
        S_a += lgamma_fast(_N + dN + 1);

        int dB = 0;
        if (r != null_group && _total[r] == n)
            dB--;
        if (nr != null_group && _total[nr] == 0)
            dB++;

        if ((dN != 0 || dB != 0))
        {
            S_b += lbinom_fast(_N - 1, _actual_B - 1);
            S_a += lbinom_fast(_N - 1 + dN, _actual_B + dB - 1);
        }

        if (dN != 0)
        {
            S_b += safelog_fast(_N);
            S_a += safelog_fast(_N + dN);
        }

        return S_a - S_b;
    }

    template <class VProp, class Graph>
    double get_delta_edges_dl(size_t v, size_t r, size_t nr, VProp& vweight,
                              size_t actual_B, Graph& g)
    {
        if (r == nr)
            return 0;

        if (r != null_group)
            r = get_r(r);
        if (nr != null_group)
            nr = get_r(nr);

        double S_b = 0, S_a = 0;

        int n = vweight[v];

        if (n == 0)
        {
            if (r == null_group)
                n = 1;
            else
                return 0;
        }

        int dB = 0;
        if (r != null_group && _total[r] == n)
            dB--;
        if (nr != null_group && _total[nr] == 0)
            dB++;

        if (dB != 0)
        {
            S_b += get_edges_dl(actual_B, g);
            S_a += get_edges_dl(actual_B + dB, g);
        }

        return S_a - S_b;
    }

    template <class Graph, class VProp, class EProp, class Degs>
    double get_delta_deg_dl(size_t v, size_t r, size_t nr, VProp& vweight,
                            EProp& eweight, Degs& degs, Graph& g, int kind)
    {
        if (r == nr || vweight[v] == 0)
            return 0;
        if (r != null_group)
            r = get_r(r);
        if (nr != null_group)
            nr = get_r(nr);

        auto dop =
            [&](auto&& f)
            {
                auto [kin, kout] = get_deg(v, eweight, degs, g);
                f(kin, kout, vweight[v]);
            };

        double dS = 0;
        switch (kind)
        {
        case deg_dl_kind::ENT:
            if (r != null_group)
                dS += get_delta_deg_dl_ent_change(r,  dop, -1);
            if (nr != null_group)
                dS += get_delta_deg_dl_ent_change(nr, dop, +1);
            break;
        case deg_dl_kind::UNIFORM:
            if (r != null_group)
                dS += get_delta_deg_dl_uniform_change(r,  dop, -1);
            if (nr != null_group)
                dS += get_delta_deg_dl_uniform_change(nr, dop, +1);
            break;
        case deg_dl_kind::DIST:
            if (r != null_group)
                dS += get_delta_deg_dl_dist_change(r,  dop, -1);
            if (nr != null_group)
                dS += get_delta_deg_dl_dist_change(nr, dop, +1);
            break;
        default:
            dS = numeric_limits<double>::quiet_NaN();
        }
        return dS;
    }

    template <class DegOP>
    double get_delta_deg_dl_ent_change(size_t r, DegOP&& dop, int diff)
    {
        int nr = _total[r];
        auto get_Sk = [&](size_t s, pair<size_t, size_t>& deg, int delta)
            {
                double S = 0;
                int nd = 0;
                if (_directed)
                {
                    if (_hist_in[s] != nullptr)
                    {
                        auto& h = *_hist_in[s];
                        auto iter = h.find(get<0>(deg));
                        if (iter != h.end())
                            nd = iter->second;
                    }
                    assert(nd + delta >= 0);
                    S -= xlogx_fast(nd + delta);
                }

                nd = 0;
                if (_hist_out[s] != nullptr)
                {
                    auto& h = *_hist_out[s];
                    auto iter = h.find(get<1>(deg));
                    if (iter != h.end())
                        nd = iter->second;
                }

                return S - xlogx_fast(nd + delta);
            };

        double S_b = 0, S_a = 0;
        int dn = 0;

        dop([&](size_t kin, size_t kout, int nk)
            {
                dn += diff * nk;
                auto deg = make_pair(kin, kout);
                S_b += get_Sk(r, deg,         0);
                S_a += get_Sk(r, deg, diff * nk);
            });

        if (_directed)
        {
            S_b += 2 * xlogx_fast(nr);
            S_a += 2 * xlogx_fast(nr + dn);
        }
        else
        {
            S_b += xlogx_fast(nr);
            S_a += xlogx_fast(nr + dn);
        }

        return S_a - S_b;
    }

    template <class DegOP>
    double get_delta_deg_dl_uniform_change(size_t r, DegOP&& dop, int diff)
    {
        auto total_r = _total[r];
        auto ep_r = _ep[r];
        auto em_r = _em[r];

        auto get_Se = [&](int dn, int dkin, int dkout)
            {
                double S = 0;
                S += lbinom_fast(total_r + dn + ep_r - 1 + dkout, ep_r + dkout);
                if (_directed)
                    S += lbinom_fast(total_r + dn + em_r - 1 + dkin,  em_r + dkin);
                return S;
            };

        double S_b = 0, S_a = 0;
        int tkin = 0, tkout = 0, n = 0;
        dop([&](auto kin, auto kout, int nk)
            {
                tkin += kin * nk;
                tkout += kout * nk;
                n += nk;
            });

        S_b += get_Se(       0,           0,            0);
        S_a += get_Se(diff * n, diff * tkin, diff * tkout);
        return S_a - S_b;
    }

    template <class DegOP>
    double get_delta_deg_dl_dist_change(size_t r, DegOP&& dop, int diff)
    {
        auto total_r = _total[r];
        auto ep_r = _ep[r];
        auto em_r = _em[r];

        auto get_Se = [&](int delta, int kin, int kout)
            {
                double S = 0;
                assert(total_r + delta >= 0);
                assert(em_r + kin >= 0);
                assert(ep_r + kout >= 0);
                if (_directed)
                    S += log_q(em_r + kin, total_r + delta);
                S += log_q(ep_r + kout, total_r + delta);
                return S;
            };

        auto get_Sr = [&](int delta)
            {
                assert(total_r + delta + 1 >= 0);
                if (_directed)
                    return 2 * lgamma_fast(total_r + delta + 1);
                else
                    return lgamma_fast(total_r + delta + 1);
            };

        auto get_Sk = [&](pair<size_t, size_t>& deg, int delta)
            {
                double S = 0;
                int nd = 0;
                if (_directed)
                {
                    if (_hist_in[r] != nullptr)
                    {
                        auto& h = *_hist_in[r];
                        auto iter = h.find(get<0>(deg));
                        if (iter != h.end())
                            nd = iter->second;
                    }
                    S -= lgamma_fast(nd + delta + 1);
                }

                nd = 0;
                if (_hist_out[r] != nullptr)
                {
                    auto& h = *_hist_out[r];
                    auto iter = h.find(get<1>(deg));
                    if (iter != h.end())
                        nd = iter->second;
                }

                return S - lgamma_fast(nd + delta + 1);
            };

        double S_b = 0, S_a = 0;
        int tkin = 0, tkout = 0, n = 0;
        dop([&](size_t kin, size_t kout, int nk)
            {
                tkin += kin * nk;
                tkout += kout * nk;
                n += nk;

                auto deg = make_pair(kin, kout);
                S_b += get_Sk(deg,         0);
                S_a += get_Sk(deg, diff * nk);
            });

        S_b += get_Se(       0,           0,            0);
        S_a += get_Se(diff * n, diff * tkin, diff * tkout);

        S_b += get_Sr(       0);
        S_a += get_Sr(diff * n);

        return S_a - S_b;
    }

    template <class VWeight>
    void change_vertex(size_t v, size_t r, VWeight& vweight, int diff)
    {
        int vw = vweight[v];
        int dv = vw * diff;

        if (_total[r] == 0 && dv > 0)
            _actual_B++;

        if (_total[r] == vw && dv < 0)
            _actual_B--;

        _total[r] += dv;
        _N += dv;

        assert(_total[r] >= 0);
    }

    template <class Graph, class VWeight, class EWeight, class Degs>
    void change_vertex_degs(size_t v, size_t r, Graph& g, VWeight& vweight,
                            EWeight& eweight, Degs& degs, int diff)
    {
        auto [kin, kout] = get_deg(v, eweight, degs, g);
        auto n = vweight[v];
        int dk = diff * n;

        auto change_hist =
            [&](auto& hist, auto& h, size_t k)
            {
                auto iter = h.insert({k, 0}).first;
                iter->second += dk;
                if (iter->second == 0)
                {
                    h.erase(iter);
                    if (h.empty())
                    {
                        delete hist[r];
                        hist[r] = nullptr;
                    }
                }
            };

        if (_directed)
            change_hist(_hist_in, get_hist<false>(r), kin);
        change_hist(_hist_out, get_hist<true>(r), kout);

        if (_directed)
            _em[r] += dk * kin;
        _ep[r] += dk * kout;
    }

    template <class Graph, class VWeight, class EWeight, class Degs>
    void remove_vertex(size_t v, size_t r, bool deg_corr, Graph& g,
                       VWeight& vweight, EWeight& eweight, Degs& degs)
    {
        if (r == null_group || vweight[v] == 0)
            return;
        r = get_r(r);
        change_vertex(v, r, vweight, -1);
        if (deg_corr)
            change_vertex_degs(v, r, g, vweight, eweight, degs, -1);
    }

    template <class Graph, class VWeight, class EWeight, class Degs>
    void add_vertex(size_t v, size_t nr, bool deg_corr, Graph& g,
                    VWeight& vweight, EWeight& eweight, Degs& degs)
    {
        if (nr == null_group || vweight[v] == 0)
            return;
        nr = get_r(nr);
        change_vertex(v, nr, vweight, 1);
        if (deg_corr)
            change_vertex_degs(v, nr, g, vweight, eweight, degs, 1);
    }

    void change_E(int dE)
    {
        _E += dE;
    }

    size_t get_N()
    {
        return _N;
    }

    size_t get_E()
    {
        return _E;
    }

    size_t get_actual_B()
    {
        return _actual_B;
    }

    void add_block()
    {
        _total_B++;
        if constexpr (!use_rmap)
        {
            if (_directed)
                _hist_in.resize(_total_B);
            _hist_out.resize(_total_B);
            _total.resize(_total_B);
            _ep.resize(_total_B);
            _em.resize(_total_B);
        }
    }

    void copy_hists()
    {
        typedef decltype(_hist_out) hist_t;
        for (auto* h : std::array<hist_t*,2>({&_hist_out, &_hist_in}))
        {
            auto& hist = *h;
            for (size_t r = 0; r < hist.size(); ++r)
            {
                if (hist[r] != nullptr)
                    hist[r] = new map_t(*hist[r]);
            }
        }
    }

private:
    bool _directed;
    vector<size_t> _bmap;
    size_t _N;
    size_t _E;
    size_t _actual_B;
    size_t _total_B;
    vector<map_t*> _hist_in, _hist_out;
    vector<int> _total;
    vector<int> _ep;
    vector<int> _em;
    map_t _dummy;
};

template <bool use_rmap>
class partition_stats:
        public partition_stats_base<use_rmap>
{
public:
    typedef partition_stats_base<use_rmap> base_t;

    template <class... Args>
    partition_stats(Args&&... args)
        : base_t(args...)
    {
    }

    partition_stats(const partition_stats& o)
        : base_t(o)
    {
        base_t::copy_hists();
    }

    partition_stats& operator=(const partition_stats& other)
    {
        if (this == &other)
            return *this;
        base_t::operator=(other);
        base_t::copy_hists();
        return *this;
    }
};


} //namespace graph_tool
#endif // GRAPH_BLOCKMODEL_PARTITION_HH
