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

#ifndef MERGE_SPLIT_HH
#define MERGE_SPLIT_HH

#include "parallel_rng.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

struct MergeSplitStateBase
{
    template <class Group>
    bool can_swap(const Group&, const Group&) { return true; }

    template <class Group>
    bool allow_move(const Group&, const Group&) { return true; }

    void relax_update(bool) {}

    template <class V>
    void store_next_state(V) {}

    void clear_next_state() {}

    template <class V>
    void push_state(V&&) {}

    void pop_state() {}

    template <class Group, class VS>
    std::tuple<Group, double> relabel_group(const Group& r, VS&) { return {r, 0}; }

    template <class V, class Rs>
    void virtual_move_lock(V, V, Rs&&) {}

    template <class V>
    void virtual_move_unlock(V) {}

    constexpr static bool _parallel = false;
    constexpr static bool _relabel = false;
};

template <class State, class Node, class Group,
          template <class> class VSet,
          template <class, class> class VMap,
          template <class> class GSet,
          class GMap, bool allow_empty=false,
          bool labelled=false>
struct MergeSplit: public State
{
    enum class move_t { single = 0, split, merge, mergesplit, movelabel, null };
    enum class stage_t { random = 0, scatter, coalesce };

    template <class... TS>
    MergeSplit(TS&&... as)
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

        std::vector<move_t> moves
            = {move_t::single, move_t::split, move_t::merge,
               move_t::mergesplit, move_t::movelabel};
        std::vector<double> probs
            = {State::_psingle, State::_psplit, State::_pmerge,
               State::_pmergesplit, State::_pmovelabel};
        _move_sampler = Sampler<move_t, mpl::false_>(moves, probs);

        std::vector<stage_t> stages
            = {stage_t::random, stage_t::scatter, stage_t::coalesce};
        std::vector<double> sprobs
            = {State::_psrandom, State::_psscatter, State::_pscoalesce};
        _stage_sampler = Sampler<stage_t, mpl::false_>(stages, sprobs);
    }

    VSet<Node> _nodes;
    GMap _groups;

    using State::_null_group;
    using State::_beta;
    using State::_gibbs_sweeps;
    using State::_verbose;
    using State::_parallel;
    using State::_relabel;

    size_t _nmoves = 0;

    std::vector<std::vector<std::tuple<Node,Group>>> _bstack;

    Sampler<move_t, mpl::false_> _move_sampler;
    Sampler<stage_t, mpl::false_> _stage_sampler;

    void _push_b_dispatch() {}

    template <class V, class... Vs>
    void _push_b_dispatch(const V& vs, Vs&&... vvs)
    {
        auto& back = _bstack.back();
        for (const auto& v : vs)
            back.emplace_back(v, State::get_group(v));
        State::push_state(vs);
        _push_b_dispatch(std::forward<Vs>(vvs)...);
    }

    template <class... Vs>
    void push_b(Vs&&... vvs)
    {
        _bstack.emplace_back();
        _push_b_dispatch(std::forward<Vs>(vvs)...);
    }

    void pop_b()
    {
        auto& back = _bstack.back();
        #pragma omp parallel for schedule(runtime) if (_parallel)
        for (size_t i = 0; i < back.size(); ++i)
        {
            auto& [v, s] = back[i];
            State::virtual_move_lock(v, State::get_group(v), s);
            move_node(v, s);
        }
        _bstack.pop_back();
        State::pop_state();
    }

    void check_rlist()
    {
#ifndef NDEBUG
        for (auto r : _rlist)
            assert(get_wr(r) > 0);
#endif
    }

    GSet<Group> _rlist;

    std::vector<Node> _vs;
    move_t _move;

    VMap<Node, Group> _bnext, _btemp;

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

    void move_node(const Node& v, const Group& r, bool cache=false)
    {
        Group s = State::get_group(v);
        if (s != r)
        {
            #pragma omp critical (move_node)
            {
                auto& vs = _groups[s];
                vs.erase(v);
                if (vs.empty())
                    _groups.erase(s);
                _groups[r].insert(v);
                _nmoves++;
            }
        }
        State::move_node(v, r, cache); // will unlock if needed
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

    size_t get_wr_(Group r)
    {
        return get_wr(r);
    }

    double relabel_group(Group& r)
    {
        Group s;
        double dS;
        std::tie(s, dS) = State::relabel_group(r, _groups[r]);
        if (dS >= 0 || (s != r && get_wr(s) > 0))
            return 0;
        if (s != r)
        {
            std::vector<Node> vrs;
            get_group_vs<>(r, vrs);
            #pragma omp parallel for schedule(runtime) if (_parallel)
            for (size_t i = 0; i < vrs.size(); ++i)
                move_node(vrs[i], s);
            r = s;
        }
        return dS;
    }

    template <class RNG>
    std::tuple<double, double>
    gibbs_sweep(std::vector<Node>& vs, const Group& r, const Group& s,
                double beta, RNG& rng_)
    {
        double lp = 0, dS = 0;
        std::array<double,2> p = {0,0};
        std::shuffle(vs.begin(), vs.end(), rng_);

        parallel_rng<rng_t> prng(rng_);
        #pragma omp parallel for schedule(runtime) if (_parallel) reduction(+: dS, lp) firstprivate(p)
        for (size_t i = 0; i < vs.size(); ++i)
        {
            // parallel
            auto& rng = prng.get(rng_);
            const auto& v = vs[i];
            Group bv = State::get_group(v);
            Group nbv = (bv == r) ? s : r;
            double ddS;
            State::virtual_move_lock(v, bv, nbv);

            // serial
            if (allow_empty || get_wr(bv) > 1)
                ddS = State::virtual_move(v, bv, nbv);
            else
                ddS = std::numeric_limits<double>::infinity();

            if (!std::isinf(beta) && !std::isinf(ddS))
            {
                double Z = log_sum_exp(0., -ddS * beta);
                p[0] = -ddS * beta - Z;
                p[1] = -Z;
            }
            else
            {
                if (ddS < 0)
                {
                    p[0] = 0;
                    p[1] = -std::numeric_limits<double>::infinity();
                }
                else
                {
                    p[0] = -std::numeric_limits<double>::infinity();;
                    p[1] = 0;
                }
            }

            std::bernoulli_distribution sample(exp(p[0]));
            if (sample(rng))
            {
                move_node(v, nbv, true);
                lp += p[0];
                dS += ddS;
            }
            else
            {
                lp += p[1];
                State::virtual_move_unlock(v);
            }

            assert(!std::isnan(lp));
        }
        return {dS, lp};
    }

    template <bool forward=true, class RNG>
    std::tuple<double, Group, Group>
    stage_split_random(std::vector<Node>& vs, const Group& r,
                       const Group& s, RNG& rng_)
    {
        std::array<Group, 2> rt = {_null_group, _null_group};
        double dS = 0;

        std::uniform_real_distribution<> unit(0, 1);
        double p0 = unit(rng_);
        std::bernoulli_distribution sample(p0);

        parallel_rng<rng_t> prng(rng_);
        std::shuffle(vs.begin(), vs.end(), rng_);
        #pragma omp parallel for schedule(runtime) if (_parallel) reduction(+: dS)
        for (size_t i = 0; i < vs.size(); ++i)
        {
            //parallel
            auto& rng = prng.get(rng_);
            const auto& v = vs[i];

            size_t j = sample(rng);

            #pragma omp critical (split_random)
            {
                if (rt[0] == _null_group)
                {
                    rt[0] = r;
                    j = 0;
                }
                else if (rt[1] == _null_group)
                {
                    if constexpr (forward)
                        rt[1] = (s == _null_group) ? State::sample_new_group(v, rng) : s;
                    else
                        rt[1] = s;
                    j = 1;
                }
            }

            State::virtual_move_lock(v, State::get_group(v), rt[j]);

            // serial
            dS += State::virtual_move(v, State::get_group(v), rt[j]);
            move_node(v, rt[j], true);
        }

        if (allow_empty && vs.size() == 1)
            rt[1] = State::sample_new_group(vs.front(), rng_);

        return {dS, rt[0], rt[1]};
    }

    template <bool forward=true, class RNG>
    std::tuple<double, Group, Group>
    stage_split_scatter(std::vector<Node>& vs, const Group& r, const Group& s, RNG& rng_)
    {
        std::array<Group, 2> rt = {_null_group, _null_group};
        std::array<double, 2> ps;
        double dS = 0;

        std::array<Group, 2> except = {r, s};
        Group t;
        if (_rlist.size() < (forward ? _N - 1 : _N))
            t = State::template sample_new_group<false>(*_groups[forward ? r : s].begin(), rng_, except);
        else
            t = r;

        std::vector<Node> vrs;
        get_group_vs<>(r, vrs);

        #pragma omp parallel for schedule(runtime) if (_parallel) reduction(+: dS)
        for (size_t i = 0; i < vrs.size(); ++i)
        {
            const auto& v = vrs[i];
            State::virtual_move_lock(v, State::get_group(v), t);
            dS += State::virtual_move(v, State::get_group(v), t);
            move_node(v, t, true);
        }

        if constexpr (!forward)
        {
            get_group_vs<>(s, vrs);

            #pragma omp parallel for schedule(runtime) if (_parallel) reduction(+: dS)
            for (size_t i = 0; i < vrs.size(); ++i)
            {
                const auto& v = vrs[i];
                State::virtual_move_lock(v, State::get_group(v), t);
                dS += State::virtual_move(v, State::get_group(v), t);
                move_node(v, t, true);
            }
        }

        std::shuffle(vs.begin(), vs.end(), rng_);

        parallel_rng<rng_t> prng(rng_);
        #pragma omp parallel for schedule(runtime) if (_parallel) reduction(+: dS) firstprivate(ps)
        for (size_t i = 0; i < vs.size(); ++i)
        {
            //parallel
            auto& rng = prng.get(rng_);
            const auto& v = vs[i];

            ps = {numeric_limits<double>::quiet_NaN(),
                  numeric_limits<double>::quiet_NaN()};

            #pragma omp critical (split_scatter)
            {
                if (rt[0] == _null_group)
                {
                    rt[0] = r;
                    //ps[0] = State::virtual_move(v, State::get_group(v), rt[0]);
                    ps[1] = -numeric_limits<double>::infinity();
                }
                else if (rt[1] == _null_group)
                {
                    if constexpr (forward)
                        rt[1] = (s == _null_group) ? State::sample_new_group(v, rng) : s;
                    else
                        rt[1] = s;
                    ps[0] = -numeric_limits<double>::infinity();
                    //ps[1] = State::virtual_move(v, State::get_group(v), rt[1]);
                }
            }

            State::virtual_move_lock(v, State::get_group(v), rt);

            //serial
            if (std::isnan(ps[0]))
                ps[0] = State::virtual_move(v, State::get_group(v), rt[0]);
            if (std::isnan(ps[1]))
                ps[1] = State::virtual_move(v, State::get_group(v), rt[1]);;

            double Z = log_sum_exp(ps[0], ps[1]);
            double p0 = ps[0] - Z;
            std::bernoulli_distribution sample(exp(p0));
            if (sample(rng))
            {
                dS += ps[0];
                move_node(v, rt[0]);
            }
            else
            {
                dS += ps[1];
                move_node(v, rt[1], true);
            }
        }

        if (allow_empty && vs.size() == 1)
            rt[1] = State::sample_new_group(vs.front(), rng_);

        return {dS, rt[0], rt[1]};
    }

    template <bool forward=true, class RNG>
    std::tuple<double, Group, Group>
    stage_split_coalesce(std::vector<Node>& vs, const Group& r,
                         const Group& s, RNG& rng_)
    {
        std::array<Group, 2> rt = {_null_group, _null_group};
        std::array<double, 2> ps;
        double dS = 0;

        std::array<Group, 2> except = {r, s};

        size_t nB = get_wr(r);
        if constexpr (!forward)
            nB += get_wr(s);

        State::reserve_empty_groups(nB);

        std::vector<Node> vrs;
        get_group_vs<>(r, vrs);

        parallel_rng<rng_t> prng(rng_);
        #pragma omp parallel for schedule(runtime) if (_parallel) reduction(+: dS)
        for (size_t i = 0; i < vrs.size(); ++i)
        {
            //parallel
            auto& rng = prng.get(rng_);
            const auto& v = vrs[i];

            Group t;
            if (_rlist.size() + i < (forward ? _N - 1 : _N))
                t = State::template sample_new_group<false>(v, rng, except);
            else
                t = r;
            State::virtual_move_lock(v, State::get_group(v), t);

            //serial
            dS += State::virtual_move(v, State::get_group(v), t);
            move_node(v, t, true);
        }

        if constexpr (!forward)
        {
            get_group_vs<>(s, vrs);

            #pragma omp parallel for schedule(runtime) if (_parallel) reduction(+: dS)
            for (size_t i = 0; i < vrs.size(); ++i)
            {
                // parallel
                auto& rng = prng.get(rng_);
                const auto& v = vrs[i];
                Group t;
                if (_rlist.size() + i < (forward ? _N - 1 : _N))
                    t = State::template sample_new_group<false>(v, rng, except);
                else
                    t = s;
                State::virtual_move_lock(v, State::get_group(v), t);

                // serial
                dS += State::virtual_move(v, State::get_group(v), t);
                move_node(v, t, true);
            }
        }

        std::shuffle(vs.begin(), vs.end(), rng_);
        #pragma omp parallel for schedule(runtime) if (_parallel) reduction(+: dS) firstprivate(ps)
        for (size_t i = 0; i < vs.size(); ++i)
        {
            // parallel
            auto& rng = prng.get(rng_);
            const auto& v = vs[i];

            ps = {numeric_limits<double>::quiet_NaN(),
                  numeric_limits<double>::quiet_NaN()};

            #pragma omp critical (split_coalesce)
            {
                if (rt[0] == _null_group)
                {
                    rt[0] = r;
                    //ps[0] = State::virtual_move(v, State::get_group(v), rt[0]);
                    ps[1] = -numeric_limits<double>::infinity();
                }
                else if (rt[1] == _null_group)
                {
                    if constexpr (forward)
                        rt[1] = (s == _null_group) ? State::sample_new_group(v, rng) : s;
                    else
                        rt[1] = s;
                    ps[0] = -numeric_limits<double>::infinity();
                    //ps[1] = State::virtual_move(v, State::get_group(v), rt[1]);
                }
            }

            State::virtual_move_lock(v, State::get_group(v), rt);

            // serial
            if (std::isnan(ps[0]))
                ps[0] = State::virtual_move(v, State::get_group(v), rt[0]);
            if (std::isnan(ps[1]))
                ps[1] = State::virtual_move(v, State::get_group(v), rt[1]);

            double Z = log_sum_exp(ps[0], ps[1]);
            double p0 = ps[0] - Z;
            std::bernoulli_distribution sample(exp(p0));
            if (sample(rng))
            {
                dS += ps[0];
                move_node(v, rt[0]);
            }
            else
            {
                dS += ps[1];
                move_node(v, rt[1], true);
            }
        }

        if (allow_empty && vs.size() == 1)
            rt[1] = State::sample_new_group(vs.front(), rng_);

        return {dS, rt[0], rt[1]};
    }

    template <class RNG, bool forward=true>
    std::tuple<Group, Group, double, double> split(const Group& r,
                                                   const Group& s, RNG& rng)
    {
        std::vector<Node> vs;
        get_group_vs<>(r, vs);

        if constexpr (!forward)
            get_group_vs<false>(s, vs);

        double dS = 0;
        std::array<Group, 2> rt = {_null_group, _null_group};

        switch (_stage_sampler.sample(rng))
        {
        case stage_t::random:
            std::tie(dS, rt[0], rt[1]) = stage_split_random<forward>(vs, r, s, rng);
            break;
        case stage_t::scatter:
            std::tie(dS, rt[0], rt[1]) = stage_split_scatter<forward>(vs, r, s, rng);
            break;
        case stage_t::coalesce:
            std::tie(dS, rt[0], rt[1]) = stage_split_coalesce<forward>(vs, r, s, rng);
            break;
        default:
            break;
        }

        if constexpr (_relabel)
        {
            if (std::isinf(_beta))
            {
                dS += relabel_group(rt[0]);
                dS += relabel_group(rt[1]);
            }
        }

        for (size_t i = 0; i < _gibbs_sweeps - 1; ++i)
        {
            auto ret = gibbs_sweep(vs, rt[0], rt[1],
                                   (i < _gibbs_sweeps / 2) ? 1 : _beta,
                                   rng);
            dS += get<0>(ret);

            if constexpr (_relabel)
            {
                if (std::isinf(_beta))
                {
                    dS += relabel_group(rt[0]);
                    dS += relabel_group(rt[1]);
                }
            }

            if (std::isinf(_beta) && abs(get<0>(ret)) < 1e-6)
                break;
        }

        double lp = 0;
        if constexpr (forward)
        {
            if constexpr (labelled)
            {
                auto ret = gibbs_sweep(vs, rt[0], rt[1], _beta, rng);
                dS += get<0>(ret);
                lp = get<1>(ret);
            }
            else
            {
                if (std::isinf(_beta) || !State::can_swap(rt[0], rt[1]))
                {
                    auto ret = gibbs_sweep(vs, rt[0], rt[1], _beta, rng);
                    dS += get<0>(ret);
                    lp = get<1>(ret);
                }
                else
                {
                    push_b(vs);

                    auto ret = gibbs_sweep(vs, rt[0], rt[1], _beta, rng);
                    dS += get<0>(ret);
                    double lp1 = get<1>(ret);

                    for (const auto& v : vs)
                        _btemp[v] = State::get_group(v);

                    pop_b();

                    #pragma omp parallel for schedule(runtime) if (_parallel)
                    for (size_t i = 0; i < vs.size(); ++i)
                    {
                        const auto& v = vs[i];
                        if (State::get_group(v) == rt[0])
                            move_node(v, rt[1]);
                        else
                            move_node(v, rt[0]);
                    }

                    double lp2 = split_prob_gibbs(rt[0], rt[1], vs);

                    lp = log_sum_exp(lp1, lp2) - log(2);

                    #pragma omp parallel for schedule(runtime) if (_parallel)
                    for (size_t i = 0; i < vs.size(); ++i)
                    {
                        const auto& v = vs[i];
                        move_node(v, _btemp[v]);
                    }

                    assert(!std::isnan(lp));
                }
            }
        }

        return {rt[0], rt[1], dS, lp};
    }

    template <class RNG>
    double split_prob(const Group& r, const Group& s, RNG& rng)
    {
        std::vector<Node> vs;
        get_group_vs<false>(r, vs);
        get_group_vs<false>(s, vs);

        for (const auto& v : vs)
            _btemp[v] = State::get_group(v);

        split<RNG, false>(r, s, rng);

        std::shuffle(vs.begin(), vs.end(), rng);

        double lp;
        if constexpr (labelled)
        {
            lp = split_prob_gibbs(r, s, vs);
        }
        else
        {
            if (!State::can_swap(r, s))
            {
                lp = split_prob_gibbs(r, s, vs);
            }
            else
            {
                push_b(vs);

                double lp1 = split_prob_gibbs(r, s, vs);

                pop_b();

                #pragma omp parallel for schedule(runtime) if (_parallel)
                for (size_t i = 0; i < vs.size(); ++i)
                {
                    const auto& v = vs[i];
                    if (State::get_group(v) == r)
                        move_node(v, s);
                    else
                        move_node(v, r);
                }

                double lp2 = split_prob_gibbs(r, s, vs);

                lp = log_sum_exp(lp1, lp2) - log(2);

                #pragma omp parallel for schedule(runtime) if (_parallel)
                for (size_t i = 0; i < vs.size(); ++i)
                {
                    const auto& v = vs[i];
                    move_node(v, _btemp[v]);
                }
            }
        }

        #pragma omp parallel for schedule(runtime) if (_parallel)
        for (size_t i = 0; i < vs.size(); ++i)
        {
            const auto& v = vs[i];
            move_node(v, _btemp[v]);
        }

        return lp;
    }

    double split_prob_gibbs(const Group& r, const Group& s,
                            const std::vector<Node>& vs)
    {
        double lp = 0;
        #pragma omp parallel for schedule(runtime) if (_parallel) reduction(+: lp)
        for (size_t i = 0; i < vs.size(); ++i)
        {
            if (std::isinf(lp))
                continue;

            // parallel
            const auto& v = vs[i];

            Group bv = State::get_group(v);
            Group nbv = (bv == r) ? s : r;

            State::virtual_move_lock(v, bv, nbv);

            //serial
            double ddS;
            if (allow_empty || get_wr(bv) > 1)
                ddS = State::virtual_move(v, bv, nbv);
            else
                ddS = std::numeric_limits<double>::infinity();

            Group tbv = _btemp[v];

            if (!std::isinf(ddS))
            {
                ddS *= _beta;
                double Z = log_sum_exp(0., -ddS);

                if (tbv == nbv)
                {
                    move_node(v, nbv);
                    lp += -ddS - Z;
                }
                else
                {
                    lp += -Z;
                    State::virtual_move_unlock(v);
                }
            }
            else
            {
                if (tbv == nbv)
                {
                    #pragma omp critical (split_prob_gibbs)
                    lp = -std::numeric_limits<double>::infinity();
                }
                State::virtual_move_unlock(v);
            }
            assert(!std::isnan(lp));
        }

        assert(!std::isnan(lp));

        return lp;
    }

    bool allow_merge(const Group& r, const Group& s)
    {
        return State::allow_move(r, s);
    }

    double merge(const Group& r, Group& s)
    {
        assert(r != s);

        double dS = 0;

        std::vector<Node> vrs;
        get_group_vs<>(r, vrs);

        #pragma omp parallel for schedule(runtime) if (_parallel) reduction(+:dS)
        for (size_t i = 0; i < vrs.size(); ++i)
        {
            // parallel
            const auto& v = vrs[i];
            State::virtual_move_lock(v, State::get_group(v), s);
            // serial
            dS += State::virtual_move(v, State::get_group(v), s); // FIXME: State::get_group(v) -> r
            move_node(v, s, true);
        }

        if constexpr (_relabel)
        {
            if (std::isinf(_beta))
            {
                for (size_t i = 0; i < _gibbs_sweeps; ++i)
                    dS += relabel_group(s);
            }
        }

        return dS;
    }

    template <class RNG>
    Group sample_move(const Group& r, RNG& rng)
    {
        Node v = uniform_sample(_groups[r], rng);
        auto s = r;
        while (s == r)
            s = State::sample_group(v, allow_empty, rng);   // FIXME: eliminate rejection
        return s;
    }

    double get_move_prob(const Group& r, const Group& s)
    {
        double prs = -numeric_limits<double>::infinity();
        double prr = -numeric_limits<double>::infinity();
        auto& vs = _groups[r];

        std::vector<Node> vs_;
        for (const auto& v : vs)
            vs_.push_back(v);

        #pragma omp parallel for schedule(runtime) if (_parallel)
        for (size_t i = 0; i < vs_.size(); ++i)
        {
            const auto& v = vs_[i];
            auto ps = State::get_move_prob(v, r, s, allow_empty, false);
            auto pr = State::get_move_prob(v, r, r, allow_empty, false);
            #pragma omp critical (get_move_prob)
            {
                prs = log_sum_exp(prs, ps);
                prr = log_sum_exp(prr, pr);
            }
        }
        prs -= safelog_fast(vs.size());
        prr -= safelog_fast(vs.size());

        auto lp = prs - log1p(-exp(prr));
        assert(!std::isnan(lp));
        return lp;
    }

    double merge_prob(const Group& r, const Group& s)
    {
        return get_move_prob(r, s);
    }

    template <class RNG>
    std::tuple<Group, double, double, double>
    sample_merge(const Group& r, RNG& rng)
    {
        Group s = sample_move(r, rng);

        if (s == r || !allow_merge(r, s))
            return {_null_group, 0., 0., 0.};

        push_b(_groups[s]);

        double pf = 0, pb = 0;
        if (!std::isinf(_beta))
        {
            pf = merge_prob(r, s);
            pb = split_prob(s, r, rng);
        }

        if (_verbose)
            cout << "merge " << get_wr(r) << " " << get_wr(s);

        double dS = merge(r, s);

        if (_verbose)
            cout << " " << dS << " " << pf << "  " << pb << " " << endl;

        return {s, dS, pf, pb};
    }

    template <class RNG>
    std::tuple<Group, double, double, double>
    sample_split(Group& r, Group s, RNG& rng)
    {
        double dS, pf, pb=0;
        std::tie(r, s, dS, pf) = split(r, s, rng);
        if (!std::isinf(_beta))
            pb = merge_prob(s, r);

        if (_verbose)
            cout << "split " << get_wr(r) << " " << get_wr(s)
                 << " " << dS << " " << pf << " " << pb << endl;

        return {s, dS, pf, pb};
    }

    template <class RNG>
    std::tuple<size_t, size_t>
    move_proposal(const Node&, RNG& rng)
    {
        double pf = 0, pb = 0;
        _dS = _a = 0;
        _vs.clear();
        _nmoves = 0;

        check_rlist();
        auto move = _move_sampler.sample(rng);

        switch (move)
        {
        case move_t::single:
            {
                auto v = uniform_sample(_nodes, rng);
                auto r = State::get_group(v);
                auto s = State::sample_group(v, true, rng);
                if (r == s || !State::allow_move(r, s))
                {
                    move = move_t::null;
                    break;
                }
                _dS = State::virtual_move(v, r, s);
                if (!std::isinf(_beta))
                {
                    pf = State::get_move_prob(v, r, s, true, false);
                    pb = State::get_move_prob(v, s, r, true, true);
                }
                _vs.clear();
                _vs.push_back(v);
                _bnext[v] = s;
                _nmoves++;
            }
            break;

        case move_t::split:
            {
                check_rlist();
                auto r = uniform_sample(_rlist, rng);

                if (get_wr(r) < 2 && !allow_empty)
                {
                    move = move_t::null;
                    break;
                }

                State::relax_update(true);

                auto& vrs = _groups[r];
                _vs.clear();
                _vs.insert(_vs.begin(), vrs.begin(), vrs.end());

                push_b(_vs);

                Group s;
                std::tie(r, s, _dS, pf) = split(r, _null_group, rng);

                if (!std::isinf(_beta))
                {
                    pf += log(State::_psplit);
                    pf += -safelog_fast(_rlist.size());

                    pb = merge_prob(s, r);
                    pb += -safelog_fast(_rlist.size()+1); //FIXME: empty groups!

                    pb += log(State::_pmerge);
                }

                if (_verbose)
                    cout << "split proposal: " << get_wr(r) << " "
                         << get_wr(s) << " " << _dS << " " << pb << " " << pf
                         << " " << -_dS + pb - pf << endl;

                for (const auto& v : _vs)
                    _bnext[v] = State::get_group(v);

                pop_b();
                check_rlist();
                State::relax_update(false);
            }
            break;

        case move_t::merge:
            {
                check_rlist();
                if (_rlist.size() == 1 && !allow_empty)
                {
                    move = move_t::null;
                    break;
                }
                auto r = uniform_sample(_rlist, rng);
                auto s = sample_move(r, rng);
                auto r_ = r;
                auto s_ = s;
                if (s == r || !allow_merge(r, s))
                {
                    move = move_t::null;
                    break;
                }

                State::relax_update(true);

                if (!std::isinf(_beta))
                {
                    pf += log(State::_pmerge);
                    pf += -safelog_fast(_rlist.size());
                    pf += merge_prob(r, s);

                    pb = -safelog_fast(_rlist.size()-1);
                    pb += split_prob(s, r, rng);
                    pb += log(State::_psplit);
                }

                auto* vrs = &_groups[r];
                _vs.clear();
                _vs.insert(_vs.end(), vrs->begin(), vrs->end());
                vrs = &_groups[s];
                _vs.insert(_vs.end(), vrs->begin(), vrs->end());

                push_b(_vs);

                _dS = merge(r, s);

                for (const auto& v : _vs)
                    _bnext[v] = State::get_group(v);

                pop_b();

                check_rlist();

                State::relax_update(false);

                if (_verbose)
                    cout << "merge proposal: " <<  get_wr(r_) << " "
                         << get_wr(s_) << " " << _dS << " " << pb << " " << pf
                         << " " << -_dS + pb - pf << endl;
            }
            break;

        case move_t::mergesplit:
            {
                if (_rlist.size() == 1)
                {
                    move = move_t::null;
                    break;
                }

                check_rlist();

                auto r = uniform_sample(_rlist, rng);

                if (get_wr(r) < 2 && !allow_empty)
                {
                    move = move_t::null;
                    break;
                }

                State::relax_update(true);

                push_b(_groups[r]);

                auto ret = sample_merge(r, rng);
                auto s = get<0>(ret);

                if (s == _null_group)
                {
                    while (!_bstack.empty())
                        pop_b();
                    State::relax_update(false);
                    move = move_t::null;
                    break;
                }

                _dS += get<1>(ret);
                pf += get<2>(ret);
                pb += get<3>(ret);

                ret = sample_split(s, r, rng);
                r = get<0>(ret);
                _dS += get<1>(ret);
                pf += get<2>(ret);
                pb += get<3>(ret);

                for (auto& vs : _bstack)
                    for (auto& vb : vs)
                    {
                        auto v = get<0>(vb);
                        _vs.push_back(v);
                        _bnext[v] = State::get_group(v);
                    }

                while (!_bstack.empty())
                    pop_b();

                check_rlist();
                State::relax_update(false);

                if (_verbose)
                    cout << "mergesplit proposal: " << _dS << " "
                         << pb << " " << pf  << " "
                         << -_dS + pb - pf << endl;
            }
            break;

        case move_t::movelabel:
            {
                check_rlist();
                auto r = uniform_sample(_rlist, rng);

                auto s = State::sample_new_group(*_groups[r].begin(), rng);

                if (!allow_merge(r, s) || r == s)
                {
                    move = move_t::null;
                    break;
                }

                State::relax_update(true);

                auto& vrs = _groups[r];
                _vs.clear();
                _vs.insert(_vs.begin(), vrs.begin(), vrs.end());

                push_b(_vs);

                _dS = merge(r, s);

                for (const auto& v : _vs)
                    _bnext[v] = State::get_group(v);

                pop_b();
                check_rlist();

                State::relax_update(false);

                if (_verbose)
                    cout << "movelabel proposal: " <<  get_wr(r) << " "
                         << get_wr(s) << " " << _dS << " " << pb << " " << pf
                         << " " << -_dS + pb - pf << endl;
            }
            break;

        default:
            move = move_t::null;
            break;
        }

        if (move == move_t::null)
            return {_null_move, _nmoves ? _nmoves : 1};

        _move = move;

        _a = pb - pf;

        if (size_t(move) >= State::_nproposal.size())
        {
            State::_nproposal.resize(size_t(move) + 1);
            State::_nacceptance.resize(size_t(move) + 1);
        }
        State::_nproposal[size_t(move)]++;

        if (State::_force_move)
        {
            _nmoves = std::numeric_limits<size_t>::max();
            _a = _dS * _beta + 1;
        }

        check_rlist();

        return {0, _nmoves};
    }

    std::tuple<double, double>
    virtual_move_dS(const Node&, size_t)
    {
        return {_dS, _a};
    }

    void perform_move(const Node&, size_t)
    {
        check_rlist();
        for (const auto& v : _vs)
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
        check_rlist();

        State::_nacceptance[size_t(_move)]++;
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
        return std::round(_N * std::min(State::_niter, 1.));
    }

    double get_beta()
    {
        return _beta;
    }

    size_t get_niter()
    {
        return std::max(State::_niter, 1.);
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

#endif // MERGE_SPLIT_HH
