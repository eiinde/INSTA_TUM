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

#ifndef MULTILEVEL_HH
#define MULTILEVEL_HH

#include <queue>
#include <cmath>

#include "mcmc_loop.hh"

#include <boost/range/combine.hpp>
#include "../../topology/graph_bipartite_weighted_matching.hh"
#include "../support/fibonacci_search.hh"
#include "../support/contingency.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

template <class State, class Node, class Group,
          template <class> class VSet,
          template <class, class> class VMap,
          template <class> class GSet,
          template <class, class> class GMap,
          class GSMap, bool allow_empty=false,
          bool relabel=false>
struct Multilevel: public State
{
    enum class move_t { multilevel=0, null };

    template <class... TS>
    Multilevel(TS&&... as)
        : State(as...)
        {
            State::iter_nodes
                ([&](const auto& v)
                {
                    auto r = State::get_group(v);
                    _groups[r].insert(v);
                    _N++;
                    _nodes.insert(v);
                });

            State::iter_groups
                ([&](const auto& r)
                {
                    _rlist.insert(r);
                });
        }

    VSet<Node> _nodes;
    GSMap _groups;

    using State::_state;
    using State::_merge_sweeps;
    using State::_mh_sweeps;
    using State::_parallel;
    using State::_init_r;
    using State::_init_min_iter;
    using State::_gibbs;
    using State::_null_group;
    using State::_beta;
    using State::_init_beta;
    using State::_cache_states;
    using State::_verbose;
    using State::_force_accept;

    using State::_B_min;
    using State::_B_max;

    size_t _nmoves = 0;

    std::vector<std::vector<std::tuple<Node,Group>>> _bstack;

    Sampler<move_t, mpl::false_> _move_sampler;

    template <class Vs>
    void push_b(Vs& vs)
    {
        _bstack.emplace_back();
        auto& back = _bstack.back();
        for (const auto& v : vs)
            back.emplace_back(v, State::get_group(v));
        State::push_state(vs);
    }

    void pop_b()
    {
        auto& back = _bstack.back();
        for (auto& vb : back)
        {
            auto& [v, s] = vb;
            auto r = State::get_group(v);
            if (r == s)
                continue;
            move_node(v, s);
        }
        _bstack.pop_back();
        State::pop_state();
    }

    GSet<Group> _rlist;

    std::vector<Node> _vs;
    move_t _move;
    GSet<Group> _rs, _rs_prev;
    GMap<Group, pair<Group, double>> _best_merge;
    GMap<Group, Group> _root;

    VSet<Node> _visited;

    VMap<Node, Group> _bnext, _bprev, _btemp;

    constexpr static size_t _null_move = 1;

    size_t _N = 0;

    double _dS;
    double _a;

    Group node_state(const Node&)
    {
        return Group();
    }

    constexpr bool skip_node(const Node&)
    {
        return false;
    }

    void move_node(const Node& v, const Group& r, bool cache = false)
    {
        Group s = State::get_group(v);
        if (s == r)
            return;
        State::move_node(v, r, cache);
        auto& vs = _groups[s];
        vs.erase(v);
        if (vs.empty())
            _groups.erase(s);
        _groups[r].insert(v);
        _nmoves++;
    }

    template <bool clear=true>
    void get_group_vs(const Group& r, std::vector<Node>& vs)
    {
        if constexpr (clear)
            vs.clear();
        auto iter = _groups.find(r);
        if (iter != _groups.end())
            vs.insert(vs.end(), iter->second.begin(), iter->second.end());
    }

    size_t get_wr(const Group& r)
    {
        auto iter = _groups.find(r);
        if (iter != _groups.end())
            return iter->second.size();
        return 0;
    }

    std::vector<size_t> _vis;

    template <bool smart, class RNG>
    double mh_sweep(std::vector<Node>& vs, GSet<Group>& rs, double beta,
                    RNG& rng, size_t B_min = 0,
                    size_t B_max = std::numeric_limits<size_t>::max(),
                    bool init_heuristic = false)
    {
        if (rs.size() == 1 || (rs.size() == vs.size() && B_min == rs.size()))
            return 0;

        _vis.resize(vs.size());
        std::iota(_vis.begin(), _vis.end(), 0);
        std::shuffle(_vis.begin(), _vis.end(), rng);

        double S = 0;
        for (size_t vi : _vis)
        {
            const auto& v = vs[vi];

            auto r = State::get_group(v);
            Group s;

            if constexpr (smart)
            {
                s = State::sample_group(v, false, false, init_heuristic, rng); // c == 0!
                if (rs.find(s) == rs.end())
                    continue;
            }
            else
            {
                rs.erase(r);
                s = uniform_sample(rs, rng);
                rs.insert(r);
            }

            int dB = 0;
            if (get_wr(s) == 0)
                dB++;
            if (get_wr(r) == 1)
                dB--;

            double dS;
            if (rs.size() + dB < B_min || rs.size() + dB > B_max)
                dS = std::numeric_limits<double>::infinity();
            else
                dS = State::virtual_move(v, r, s);

            double pf = 0, pb = 0;
            if constexpr (smart)
            {
                if (!std::isinf(beta) && s != r)
                {
                    pf = State::get_move_prob(v, r, s, false, rs.size() > B_min, false);
                    pb = State::get_move_prob(v, s, r, false, rs.size() > B_min, true);
                }
            }

            double ap = 0, rp = 0;
            bool accept;
            if constexpr (smart)
            {
                accept = metropolis_accept(dS, pb - pf, beta, rng);
            }
            else
            {
                if (!std::isinf(beta))
                {
                    double logZ = log_sum_exp(-beta * dS, 0.);
                    ap = -beta * dS - logZ;
                    rp = -logZ;

                    std::bernoulli_distribution u(exp(ap));
                    accept = u(rng);
                }
                else
                {
                    accept = dS < 0;
                    ap = 0;
                    rp = -std::numeric_limits<double>::infinity();
                    if (!accept)
                        std::swap(ap, rp);
                }
            }

            if (accept)
            {
                move_node(v, s, true);
                S += dS;

                if (get_wr(r) == 0)
                    rs.erase(r);

                assert(r != s || dS == 0);
            }
        }

        return S;
    }

    std::vector<Group> _mh_moves;

    template <bool smart, class RNG>
    double pseudo_mh_sweep(std::vector<Node>& vs, GSet<Group>& rs, double beta,
                           RNG& rng_, size_t B_min = 0,
                           size_t B_max = std::numeric_limits<size_t>::max(),
                           bool init_heuristic = false)
    {
        if (rs.size() == 1 || (rs.size() == vs.size() && B_min == rs.size()))
            return 0;

        if (_vis.size() < vs.size())
        {
            _vis.resize(vs.size());
            std::iota(_vis.begin(), _vis.end(), 0);
            std::shuffle(_vis.begin(), _vis.end(), rng_);
        }

        auto& moves = _mh_moves;
        moves.resize(vs.size());

        double S = 0;

        parallel_rng<rng_t> prng(rng_);

        #pragma omp parallel for schedule(runtime) reduction(+:S)
        for (size_t i = 0; i < _vis.size(); ++i)
        {
            auto& rng = prng.get(rng_);

            size_t vi = _vis[i];
            const auto& v = vs[vi];

            auto r = State::get_group(v);
            Group s;

            if constexpr (smart)
            {
                s = State::sample_group(v, false, false, init_heuristic, rng); // c == 0!
                if (rs.find(s) == rs.end() || s == r)
                {
                    moves[vi] = r;
                    continue;
                }
            }
            else
            {
                s = r;
                while (s == r)
                    s = uniform_sample(rs, rng);
            }

            double dS = State::virtual_move(v, r, s);

            if (metropolis_accept(dS, 0, beta, rng))
            {
                S += dS;
                moves[vi] = s;
            }
            else
            {
                moves[vi] = r;
            }
        }

        for (auto vi : _vis)
        {
            const auto& v = vs[vi];
            auto r = State::get_group(v);
            auto s = moves[vi];
            if (r == s || get_wr(s) == 0)
                continue;
            int dB = 0;
            if (get_wr(s) == 0)
                dB++;
            if (get_wr(r) == 1)
                dB--;
            if (rs.size() + dB < B_min || rs.size() + dB > B_max)
                continue;
            move_node(v, s, false);
            if (get_wr(r) == 0)
                rs.erase(r);
        }

        return S;
    }

    template <class RNG>
    double gibbs_sweep(std::vector<Node>& vs, GSet<Group>& rs, RNG& rng)
    {
        if (rs.size() == 1 || rs.size() == vs.size())
            return 0;

        _vis.resize(vs.size());
        std::iota(_vis.begin(), _vis.end(), 0);
        std::shuffle(_vis.begin(), _vis.end(), rng);

        double S = 0;

        std::vector<double> dS(rs.size());
        std::vector<double> probs(rs.size());
        std::vector<double> ps(rs.size());
        std::vector<size_t> ris(rs.size());
        std::iota(ris.begin(), ris.end(), 0);

        for (size_t vi : _vis)
        {
            const auto& v = vs[vi];

            auto r = State::get_group(v);

            for (size_t j = 0; j < rs.size(); ++j)
            {
                auto iter = rs.begin();
                std::advance(iter, j);
                Group s = *iter;
                if (s != r && get_wr(State::get_group(v)) == 1)
                    dS[j] = std::numeric_limits<double>::infinity();
                else
                    dS[j] = State::virtual_move(v, r, s);
            }

            double Z = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < rs.size(); ++j)
            {
                if (!std::isinf(_beta) && !std::isinf(dS[j]))
                {
                    ps[j] = -dS[j] * _beta;
                }
                else
                {
                    if (dS[j] < 0)
                        ps[j] = 0;
                    else
                        ps[j] = -std::numeric_limits<double>::infinity();
                }
                Z = log_sum_exp(Z, ps[j]);
            }

            size_t si = rs.size();
            for (size_t j = 0; j < rs.size(); ++j)
                probs[j] = exp(ps[j] - Z);

            Sampler<size_t> sampler(ris, probs);
            si = sampler.sample(rng);

            if (si >= rs.size() || std::isinf(dS[si]))
                break;

            auto iter = rs.begin();
            std::advance(iter, si);
            move_node(v, *iter);
            S += dS[si];
        }

        return S;
    }

    template <class RNG>
    double relabel_sweep(GSet<Group>& rs, double beta, RNG& rng)
    {
        double S = 0;
        for (auto r : rs)
        {
            auto u = State::sample_group_label(r, rng);
            double dS = State::relabel_group_dS(r, u);
            if (metropolis_accept(dS, 0, beta, rng))
            {
                State::relabel_group(r, u);
                S += dS;
            }
        }
        return S;
    }

    template <class RNG>
    double pseudo_relabel_sweep(GSet<Group>& rs_, double beta, RNG& rng_)
    {
        typedef decltype(State::group_label(Group())) label_t;
        std::vector<label_t> moves(rs_.size());
        std::vector<Group> rs(rs_.begin(), rs_.end());

        double S = 0;

        parallel_rng<rng_t> prng(rng_);

        #pragma omp parallel for schedule(runtime) reduction(+:S)
        for (size_t i = 0; i < _vis.size(); ++i)
        {
            auto r = rs[i];
            auto& rng = prng.get(rng_);
            auto u = State::sample_group_label(r, rng);
            double dS = State::relabel_group_dS(r, u);
            if (metropolis_accept(dS, 0, beta, rng))
            {
                moves[i] = u;
                S += dS;
            }
            else
            {
                moves[i] = State::group_label(r);
            }
        }

        for (size_t i = 0; i < _vis.size(); ++i)
        {
            auto r = rs[i];
            auto u = moves[i];
            if (u != State::group_label(r))
                State::relabel_group(r, u);
        }

        return S;
    }

    double virtual_merge_dS(const Group& r, const Group& s)
    {
        std::vector<Node> mvs;
        State::relax_update(true);

        double dS = 0;
        for (auto& v : _groups[r])
        {
            assert(State::get_group(v) == r);
            double ddS = State::virtual_move(v, r, s);
            dS += ddS;
            if (std::isinf(ddS))
                break;
            State::move_node(v, s, true);
            mvs.push_back(v);
        }

        for (auto& v : mvs)
            State::move_node(v, r, false);

        State::relax_update(false);

        return dS;
    }

    void merge(const Group& r, const Group& s)
    {
        assert(r != s);

        std::vector<Node> mvs;
        get_group_vs(r, mvs);
        for (auto& v : mvs)
            move_node(v, s);
    }

    template <class RNG>
    double merge_sweep(GSet<Group>& rs, size_t B, size_t niter, RNG& rng_)
    {
        double S = 0;

        std::vector<Group> rlist;
        _best_merge.clear();
        for (const auto& r : rs)
        {
            _best_merge[r] = std::make_pair(r, numeric_limits<double>::infinity());
            rlist.push_back(r);
        }

        gt_hash_set<Group> past_merges;
        parallel_rng<rng_t> prng(rng_);

        if (_parallel)
            State::split_parallel();

        #pragma omp parallel for schedule(runtime) if (_parallel)   \
            firstprivate(past_merges)
        for (size_t j = 0; j < rlist.size(); ++j)
        {
            auto& rng = prng.get(rng_);
            auto& r = rlist[j];
            auto find_candidates = [&](bool allow_random)
                {
                    for (size_t i = 0; i < niter; ++i)
                    {
                        auto v = uniform_sample(_groups[r], rng);
                        auto s = State::sample_group(v, allow_random, false, false, rng);
                        if (s != r &&
                            rs.find(s) != rs.end() &&
                            past_merges.find(s) == past_merges.end())
                        {
                            double dS = virtual_merge_dS(r, s);
                            if (!std::isinf(dS) && dS < _best_merge[r].second)
                                _best_merge[r] = {s, dS};
                            past_merges.insert(s);
                        }
                    }
                };

            // Prefer smart constrained moves. If no candidates were found, the
            // group is likely to be "stuck" (e.g. isolated or constrained by
            // clabel); attempt random movements instead.

            find_candidates(false);

            if (_best_merge[r].first == r)
                find_candidates(true);

            past_merges.clear();
        }

        if (_parallel)
            State::unsplit_parallel();

        std::vector<std::pair<Group, Group>> pairs;
        std::vector<double> dS;
        for (const auto& r : rs)
        {
            auto& m = _best_merge[r];
            if (m.first == r)
                continue;
            pairs.emplace_back(r, m.first);
            dS.emplace_back(m.second);
        }

        auto cmp = [&](size_t i, size_t j) { return dS[i] > dS[j]; };
        std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)>
            queue(cmp);

        std::vector<size_t> pis(pairs.size());
        std::iota(pis.begin(), pis.end(), 0);
        std::shuffle(pis.begin(), pis.end(), rng_);

        for (auto i : pis)
            queue.push(i);

        _root.clear();
        auto get_root = [&](Group r)
        {
            Group s = r;
            if (_root.find(r) == _root.end())
                _root[r] = r;
            while (_root[r] != r)
                r = _root[r];
            _root[s] = r;
            return r;
        };

        while (rs.size() > B && !queue.empty())
        {
            auto i = queue.top();
            queue.pop();

            std::pair<Group, Group>& m = pairs[i];
            m.first = get_root(m.first);
            m.second = get_root(m.second);
            if (m.first == m.second)
                continue;

            double ndS = virtual_merge_dS(m.first, m.second);
            if (!queue.empty() && ndS > dS[queue.top()])
            {
                dS[i] = ndS;
                queue.push(i);
                continue;
            }

            _root[m.first] = m.second;
            merge(m.first, m.second);
            S += ndS;
            rs.erase(m.first);
            assert(get_wr(m.first) == 0);
        }

        assert(rs.size() >= B);
        return S;
    }

    template <class RNG>
    double stage_multilevel(GSet<Group>& rs, std::vector<Node>& vs, RNG& rng)
    {
        size_t N = vs.size();

        if (N == 1)
            return 0;

        if (_verbose)
            cout << "staging multilevel, N = " << N << endl;

        size_t B_max = State::_global_moves ? std::min(N, _B_max) : std::min(N, State::_M);
        size_t B_min = State::_global_moves ? std::max(size_t(1), _B_min) : 1;

        B_min = std::min(std::max(B_min, State::get_Bmin(vs)), B_max);

        size_t B_mid;

        size_t B_max_init = B_max;
        size_t B_min_init = B_min;

        double S_best = std::numeric_limits<double>::infinity();

        map<size_t, std::pair<double, std::vector<Group>>> cache;

        auto has_cache =
            [&](size_t B)
            {
                return cache.find(B) != cache.end();
            };

        auto put_cache = [&](size_t B, double S)
        {
            assert(cache.find(B) == cache.end());

            auto& c = cache[B];
            c.first = S;
            c.second.resize(vs.size());
            for (size_t i = 0; i < vs.size(); ++i)
                c.second[i] = State::get_group(vs[i]);
            if (S < S_best)
                S_best = S;
        };

        auto get_cache = [&](size_t B, GSet<Group>& rs)
        {
            assert(cache.find(B) != cache.end());

            rs.clear();
            auto& c = cache[B];
            for (size_t i = 0; i < vs.size(); ++i)
            {
                auto& s = c.second[i];
                move_node(vs[i], s);
                rs.insert(s);
            }
            assert(rs.size() == B);
            return c.first;
        };

        auto get_S = [&](size_t B, bool keep_cache=true)
        {
            auto iter = cache.lower_bound(B);
            if (iter->first == B)
                return iter->second.first;
            assert(iter != cache.end());

            double S = get_cache(iter->first, rs);

            if (_verbose)
            {
                cout << "bracket B = [ " << B_min
                     << ", " << B_mid
                     << ", " << B_max << " ]"
                     << endl;
                cout << "shrinking from: " << iter->first << " -> " << B << endl;
            }

            // merge & sweep
            while (rs.size() > B)
            {
                size_t Bprev = rs.size();
                auto Bnext =
                    std::max(std::min(rs.size() - 1,
                                      size_t(round(rs.size() * State::_r))),
                             B);

                while (rs.size() != Bnext)
                    S += merge_sweep(rs, Bnext, _merge_sweeps, rng);

                double Sb = _parallel ? State::entropy() - S: 0;

                for (size_t i = 0; i < _mh_sweeps; ++i)
                {
                    double dS = 0;
                    if constexpr (relabel)
                    {
                        if (_parallel)
                            dS += pseudo_relabel_sweep(rs, _beta, rng);
                        else
                            dS += relabel_sweep(rs, _beta, rng);
                    }
                    if (_parallel)
                        dS += pseudo_mh_sweep<true>(vs, rs, _beta, rng, B);
                    else
                        dS += mh_sweep<true>(vs, rs, _beta, rng, B);
                    S += dS;
                    if (std::isinf(_beta) && abs(dS) < 1e-8)
                        break;
                }

                double Sa = _parallel ? State::entropy() : S;

                S = Sa - Sb;

                if ((keep_cache && _cache_states) || rs.size() == B)
                    put_cache(rs.size(), S);

                if (_verbose)
                    cout << "    " << Bprev << " -> "
                         << rs.size() << ": " << S << endl;
            }

            assert(rs.size() == B);
            return S;
        };

        if (State::_has_b_min)
        {
            if (B_min == _B_min)
            {
                push_b(vs);
                double S = 0;
                if (rs.size() != B_min)
                {
                    for (auto& v : vs)
                    {
                        auto r = State::get_group(v);
                        Group t = State::get_b_min(v);
                        if (r == t)
                            continue;
                        S += State::virtual_move(v, r, t);
                        move_node(v, t, true);
                    }
                }
                assert(rs.size() == B_min);
                put_cache(B_min, S);
                pop_b();
            }
        }
        else if (B_min == 1)
        {
            push_b(vs);
            double S = 0;
            if (rs.size() > 1)
            {
                auto u = uniform_sample(vs, rng);
                auto t = State::get_group(u);
                for (auto& v : vs)
                {
                    auto r = State::get_group(v);
                    if (r == t)
                        continue;
                    S += State::virtual_move(v, r, t);
                    move_node(v, t, true);
                }
            }
            put_cache(B_min, S);
            pop_b();
        }

        if (!State::_has_b_max)
        {
            double S = 0;
            push_b(vs);
            State::relax_update(true);
            rs.clear();
            for (auto& v : vs)
            {
                auto s = State::get_group(v);
                if (get_wr(s) == 1)
                {
                    rs.insert(s);
                    continue;
                }
                auto t = State::get_new_group(v, true, rng);
                S += State::virtual_move(v, s, t);
                move_node(v, t, true);
                rs.insert(t);
            }

            put_cache(rs.size(), S);

            // single-node sweep initialization with B = N. This is faster than
            // using merges!
            if (std::isinf(_beta) && _init_r < 1.)
            {
                size_t Bprev;
                size_t i = 0;
                double Sb = _parallel ? State::entropy() - S: 0;
                do
                {
                    Bprev = rs.size();
                    double dS = 0;
                    if constexpr (relabel)
                    {
                        if (_parallel)
                            dS += pseudo_relabel_sweep(rs, (i == 0) ? _init_beta : _beta, rng);
                        else
                            dS += relabel_sweep(rs, (i == 0) ? _init_beta : _beta, rng);
                    }

                    if (_parallel)
                        dS += pseudo_mh_sweep<true>(vs, rs,
                                                    (i == 0) ? _init_beta : _beta,
                                                    rng, B_min, B_max, true);
                    else
                        dS += mh_sweep<true>(vs, rs,
                                             (i == 0) ? _init_beta : _beta,
                                             rng, B_min, B_max, true);
                    S += dS;
                    if (_verbose)
                        cout << i << " " << ((i == 0) ? _init_beta : _beta)
                             << " " << dS << " " << rs.size() << " "
                             << rs.size()/double(Bprev) << endl;
                    ++i;
                }
                while (i < _init_min_iter || rs.size()/double(Bprev) < _init_r);

                double Sa = _parallel ? State::entropy() : S;
                S = Sa - Sb;

                B_max = B_max_init = std::min(rs.size(), B_max);
            }

            if (rs.size() >= B_min && !has_cache(rs.size()))
                put_cache(rs.size(), S);

            pop_b();
            State::relax_update(false);
            get_cache(rs.size(), rs);
        }
        else if (B_max != B_min)
        {
            double S = 0;
            for (auto& v : vs)
            {
                auto r = State::get_group(v);
                Group t = State::get_b_max(v);
                if (r == t)
                    continue;
                S += State::virtual_move(v, r, t);
                move_node(v, t, true);
            }
            put_cache(B_max, S);
        }

        FibonacciSearch<size_t> fb;
        if (State::_random_bisect)
            fb.search(B_min, B_mid, B_max, get_S, 0, 0, rng);
        else
            fb.search(B_min, B_mid, B_max, get_S);

        // add midpoint
        if (!std::isinf(_beta))
        {
            size_t Br = (State::_random_bisect ?
                         fb.get_mid(B_min_init, B_max_init, rng) :
                         fb.get_mid(B_min_init, B_max_init));
            get_S(Br, false);
        }

        // remove out of bounds
        auto iter = cache.upper_bound(B_max);
        if (iter != cache.end())
            cache.erase(iter, cache.end());
        iter = cache.lower_bound(B_min);
        if (iter != cache.end())
            cache.erase(cache.begin(), iter);

        // Sample partition
        iter = max_element(cache.begin(), cache.end(),
                           [&](auto& x, auto& y)
                           {
                               return x.second.first > y.second.first;
                           });

        std::vector<size_t> Bs;
        std::vector<double> probs;
        for (auto& BS : cache)
        {
            Bs.push_back(BS.first);
            if (!std::isinf(_beta))
            {
                probs.push_back(exp(_beta * (-BS.second.first +
                                             iter->second.first)));
            }
            else
            {
                if (BS.second.first == iter->second.first)
                    probs.push_back(1);
                else
                    probs.push_back(0);
            }
        }

        Sampler<size_t> B_sampler(Bs, probs);
        size_t B = B_sampler.sample(rng);
        double S = get_cache(B, rs);

        assert(rs.size() == B);

        return S;
    }

    template <class RNG>
    void sample_rs(GSet<Group>& rs, RNG& rng)
    {
        if (State::_global_moves)
        {
            rs.clear();
            for (auto r : _rlist)
                rs.insert(r);
        }
        else
        {
            std::uniform_int_distribution<size_t>
                sample(1, std::min(State::_M, _rlist.size()));

            auto M = sample(rng);

            rs.clear();
            while (rs.size() < M)
            {
                auto r = uniform_sample(_rlist, rng);
                _rlist.erase(r);
                rs.insert(r);

                if (get_wr(r) == 0)
                    abort();
            }

            for (const auto& r : rs)
                _rlist.insert(r);
        }
    }

    template <class RNG>
    std::tuple<size_t, size_t>
    move_proposal(const Node&, RNG& rng)
    {
        _dS = _a = 0;
        _vs.clear();
        _nmoves = 0;

        sample_rs(_rs, rng);
        size_t nr = _rs.size();

        _vs.clear();
        for (const auto& r : _rs)
            get_group_vs<false>(r, _vs);

        //push_b(_vs);
        for (auto& v : _vs)
            _bprev[v] = State::get_group(v);

        _dS = stage_multilevel(_rs, _vs, rng);

        size_t nnr = _rs.size();

        for (auto& v : _vs)
            _bnext[v] = State::get_group(v);

        if (_verbose)
            cout << "multilevel proposal: " << nr << "->" << nnr
                 << " (" << _vs.size() << "), dS: " << _dS << endl;

        for (auto& v : _vs)
            move_node(v, _bprev[v]);

        if (_force_accept)
            _dS = -numeric_limits<double>::infinity();

        return {0, _nmoves};
    }

    std::tuple<double, double>
    virtual_move_dS(const Node&, size_t)
    {
        return {_dS, _a};
    }

    void perform_move(const Node&, size_t)
    {
        for (auto& v : _vs)
        {
            auto r = State::get_group(v);
            auto s = _bnext[v];
            if (r == s)
                continue;

            if (get_wr(s) == 0)
                _rlist.insert(s);

            move_node(v, s);

            if (get_wr(r) == 0)
                _rlist.erase(r);
        }
    }

    constexpr bool is_deterministic()
    {
        return true;
    }

    constexpr bool is_sequential()
    {
        return false;
    }

    std::array<Node, 1> _vlist = {Node()};

    auto& get_vlist()
    {
        return _vlist;
    }

    size_t get_N()
    {
        return 1; //_N;
    }

    double get_beta()
    {
        return _beta;
    }

    size_t get_niter()
    {
        return State::_niter;
    }

    constexpr void step(const Node&, size_t)
    {
    }

    template <class RNG>
    void init_iter(RNG&)
    {
    }
};

} // graph_tool namespace

#endif // MULTILEVEL_HH
