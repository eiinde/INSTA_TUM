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

#ifndef GRAPH_SBM_SAMPLE_EDGE_HH
#define GRAPH_SBM_SAMPLE_EDGE_HH

#include <tuple>

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "../../generation/sampler.hh"

#include "random.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class State>
class SBMEdgeSampler
{
public:
    SBMEdgeSampler(State& state, double esample=.25, double usample=.25)
        : _state(state),
          _tgts(vertices(state._g).first,
                vertices(state._g).second),
          _N(num_vertices(state._g)),
          _sample_vertex(0, _N - 1),
          _v_in_sampler(graph_tool::is_directed(state._g) ?
                        __v_in_sampler : _v_out_sampler),
          _esample(esample),
          _usample(usample)
    {
        for (auto v : _tgts)
        {
            for (auto e : in_or_out_edges_range(v, state._g))
            {
                size_t u = source(e, state._g);
                if (_edge_pos.find(get_edge(u, v)) != _edge_pos.end())
                    continue;
                _edges.push_back(get_edge(u, v));
                _edge_pos[_edges.back()] = _edges.size() - 1;
                _E += _state._eweight[e];
            }
        }

        if (_esample == 1 || _usample == 1)
            return;

        for (auto me : edges_range(state._bg))
        {
            auto r = source(me, state._bg);
            auto s = target(me, state._bg);

            std::tie(r, s) = get_edge(r, s);

            size_t mrs = state._mrs[me];
            if (mrs == 0)
                continue;
            _rs_pos[me] = _rs_sampler.insert({r, s}, mrs);

            if (!graph_tool::is_directed(_state._g) && r == s)
                mrs *= 2;

            if (r >= _r_out_sampler.size())
                _r_out_sampler.resize(r + 1);
            _r_out_pos[me] = _r_out_sampler[r].insert(s, mrs);

            if (graph_tool::is_directed(_state._g) || r != s)
            {
                auto& r_in_sampler = graph_tool::is_directed(_state._g) ?
                    _r_in_sampler : _r_out_sampler;
                if (s >= r_in_sampler.size())
                    r_in_sampler.resize(s + 1);
                _r_in_pos[me] = r_in_sampler[s].insert(r, mrs);
            }
        }

        for (auto v : _tgts)
        {
            size_t r = state._b[v];
            if (r >= _v_out_sampler.size())
            {
                if (graph_tool::is_directed(_state._g))
                    _v_in_sampler.resize(r+1);
                _v_out_sampler.resize(r+1);
            }

            auto [kin, kout] = (state._deg_corr) ?
                get_deg(v, state._eweight, state._degs, state._g) :
                std::make_tuple(size_t(0), size_t(0));

            if (graph_tool::is_directed(_state._g))
                _v_in_pos[v] = _v_in_sampler[r].insert(v, kin + 1);
            _v_out_pos[v] = _v_out_sampler[r].insert(v, kout + 1);
        }
    }

    SBMEdgeSampler(const SBMEdgeSampler&) = delete;

    std::tuple<size_t, size_t> get_edge(size_t u, size_t v)
    {
        if (!graph_tool::is_directed(_state._g) && u > v)
            return {v, u};
        return {u, v};
    }

    void update_edge(size_t u, size_t v, size_t m, int delta)
    {
        if (m == 0 && delta > 0)
        {
            _edges.push_back(get_edge(u, v));
            _edge_pos[_edges.back()] = _edges.size() - 1;
        }

        _E += delta;

        if (m > 0 && (m + delta) == 0)
        {
            auto iter = _edge_pos.find(get_edge(u, v));
            size_t pos = iter->second;
            _edge_pos.erase(iter);
            if (pos < _edges.size() - 1)
            {
                _edges[pos] = _edges.back();
                _edge_pos[_edges.back()] = pos;
            }
            _edges.pop_back();
        }

        if (_esample == 1 || _usample == 1)
            return;

        size_t r = _state._b[u];
        size_t s = _state._b[v];

        std::tie(r, s) = get_edge(r, s);

        auto me = _state._emat.get_me(r, s);
        assert (me != _state._emat.get_null_edge());

        int ers = _state._mrs[me] + ((delta < 0) ? delta : 0);
        if (ers == 0)
        {
            _rs_sampler.remove(_rs_pos[me]);
            _rs_pos[me] = std::numeric_limits<size_t>::max();

            _r_out_sampler[r].remove(_r_out_pos[me]);
            _r_out_pos[me] = std::numeric_limits<size_t>::max();

            if (graph_tool::is_directed(_state._g) || r != s)
            {
                auto& r_in_sampler = graph_tool::is_directed(_state._g) ?
                    _r_in_sampler : _r_out_sampler;

                r_in_sampler[s].remove(_r_in_pos[me]);
                _r_in_pos[me] = std::numeric_limits<size_t>::max();
            }
        }
        else if (delta == ers)
        {
            _rs_pos[me] = _rs_sampler.insert({r, s}, ers);

            if (!graph_tool::is_directed(_state._g) && r == s)
                ers *= 2;

            if (r >= _r_out_sampler.size())
                _r_out_sampler.resize(r + 1);
            _r_out_pos[me] = _r_out_sampler[r].insert(s, ers);

            if (graph_tool::is_directed(_state._g) || r != s)
            {
                auto& r_in_sampler = graph_tool::is_directed(_state._g) ?
                    _r_in_sampler : _r_out_sampler;

                if (s >= r_in_sampler.size())
                    r_in_sampler.resize(s + 1);
                _r_in_pos[me] = r_in_sampler[s].insert(r, ers);
            }
        }
        else
        {
            _rs_sampler.update(_rs_pos[me], delta, true);

            int dm = delta;
            if (!graph_tool::is_directed(_state._g) && r == s)
                dm *= 2;

            _r_out_sampler[r].update(_r_out_pos[me], dm, true);

            if (graph_tool::is_directed(_state._g) || r != s)
            {
                auto& r_in_sampler = graph_tool::is_directed(_state._g) ?
                    _r_in_sampler : _r_out_sampler;

                r_in_sampler[s].update(_r_in_pos[me], dm, true);
            }
        }

        if (_state._deg_corr)
        {
            size_t ku = get<1>(get_deg(u, _state._eweight, _state._degs, _state._g));
            size_t kv = (graph_tool::is_directed(_state._g)) ?
                get<0>(get_deg(v, _state._eweight, _state._degs, _state._g)) :
                get<1>(get_deg(v, _state._eweight, _state._degs, _state._g));

            if (delta < 0)
            {
                if (u != v || graph_tool::is_directed(_state._g))
                {
                    ku += delta;
                    kv += delta;
                }
                else
                {
                    ku += 2 * delta;
                }
            }

            r = _state._b[u];
            _v_out_sampler[r].remove(_v_out_pos[u]);
            _v_out_pos[u] = _v_out_sampler[r].insert(u, ku + 1);

            if (u != v || graph_tool::is_directed(_state._g))
            {
                s = _state._b[v];
                if (graph_tool::is_directed(_state._g))
                {
                    _v_in_sampler[s].remove(_v_in_pos[v]);
                    _v_in_pos[v] = _v_in_sampler[s].insert(v, kv + 1);
                }
                else
                {
                    _v_out_sampler[s].remove(_v_out_pos[v]);
                    _v_out_pos[v] = _v_out_sampler[s].insert(v, kv + 1);
                }
            }
        }
    }

    template <class RNG>
    std::tuple<size_t, size_t> sample(RNG& rng, bool edges_only = false)
    {
        // if (_edges.empty())
        //     return _null_edge;
        // return uniform_sample(_edges, rng);

        if (_esample == 1 || edges_only)
        {
            if (_edges.empty())
                return _null_edge;

            std::bernoulli_distribution coin(_E / double(_E + _N));
            if (edges_only || coin(rng))
            {
                return uniform_sample(_edges, rng);
            }
            else
            {
                size_t v = uniform_sample(_tgts, rng);
                return {v, v};
            }
        }

        std::bernoulli_distribution esample(_esample);

        if (!_edges.empty() && esample(rng))
            return uniform_sample(_edges, rng);

        std::bernoulli_distribution usample(_usample);

        if (_edges.empty() || usample(rng))
            return get_edge(_sample_vertex(rng),
                            uniform_sample(_tgts, rng));

        auto [r, s] = _rs_sampler.sample(rng);
        auto& r_sampler = _v_out_sampler[r];
        auto& s_sampler = _v_in_sampler[s];
        return get_edge(r_sampler.sample(rng),
                        s_sampler.sample(rng));
    }

    double log_prob(size_t u, size_t v, size_t m, int delta,
                    bool edges_only = false)
    {
        // if (_edges.size() + delta == 0)
        //     return -numeric_limits<double>::infinity();
        // return -log(_edges.size() + delta);

        if (_esample == 1 || edges_only)
            return -log(_edges.size() + delta);

        auto& g = _state._g;
        auto E = graph_tool::is_directed(g) ? (_E + delta) :
            2 * (_E + delta);

        double lu = -safelog_fast(_N) - safelog_fast(_tgts.size());
        if (!graph_tool::is_directed(g) && u != v)
            lu += log(2);

        if (_usample == 1 || E == 0)
            return lu;

        size_t r = _state._b[u];
        size_t s = _state._b[v];

        size_t ku = 0, kv = 0;
        if (_state._deg_corr)
        {
            ku = get<1>(get_deg(u, _state._eweight, _state._degs, _state._g));
            kv = (graph_tool::is_directed(_state._g)) ?
                get<0>(get_deg(v, _state._eweight, _state._degs, _state._g)) :
                get<1>(get_deg(v, _state._eweight, _state._degs, _state._g));
        }

        auto&& me = _state._emat.get_me(r, s);
        size_t ers = (me == _state._emat.get_null_edge()) ? 0 : _state._mrs[me];
        ers += delta;

        if (!graph_tool::is_directed(g) && r == s)
            ers *= 2;

        size_t nr = _state._wr[r];
        size_t ns = _state._wr[s];
        size_t er = _state._mrp[r];
        size_t es = graph_tool::is_directed(g) ? _state._mrm[s] : _state._mrp[s];

        if (_state._deg_corr)
        {
            if (r != s || graph_tool::is_directed(g))
            {
                er += delta;
                es += delta;
            }
            else
            {
                er += 2 * delta;
                es += 2 * delta;
            }

            if (u != v || graph_tool::is_directed(g))
            {
                ku += delta;
                kv += delta;
            }
            else
            {
                ku += 2 * delta;
                kv += 2 * delta;
            }
        }
        else
        {
            er = es = 0;
        }

        double lp = 0;

        if (E > 0)
        {
            if (ers > 0)
            {
                lp = ((safelog_fast(ers) - safelog_fast(E)) +
                      (safelog_fast(ku + 1) - safelog_fast(er + nr)) +
                      (safelog_fast(kv + 1) - safelog_fast(es + ns)));

                if (!graph_tool::is_directed(g) && u != v)
                    lp += log(2);

                lp = log_sum_exp(lp + log1p(-_usample),
                                 lu + log(_usample));
            }
            else
            {
                lp = lu + log(_usample);
            }
        }
        else
        {
            lp = lu;
        }

        if (_esample > 0)
        {
            if (m + delta > 0)
            {
                double rp;
                if (m == 0)
                    rp = -safelog_fast(_edges.size() + 1);
                else
                    rp = -safelog_fast(_edges.size());
                return log_sum_exp(rp + log(_esample),
                                   lp + log1p(-_esample));
            }
            else
            {
                return lp + log1p(-_esample);
            }
        }

        return lp;
    }

    template <class RNG>
    size_t sample_out_neighbor(size_t v, RNG& rng)
    {
        auto r = _state._b[v];
        std::bernoulli_distribution usample(_usample);
        if (_state._mrp[r] == 0 || usample(rng))
            return _sample_vertex(rng);
        auto s = _r_out_sampler[r].sample(rng);
        return _v_in_sampler[s].sample(rng);
    }

    template <class RNG>
    size_t sample_in_neighbor(size_t v, RNG& rng)
    {
        auto r = _state._b[v];
        std::bernoulli_distribution usample(_usample);
        if (_state._mrm[r] == 0 || usample(rng))
            return _sample_vertex(rng);
        auto s = _r_in_sampler[r].sample(rng);
        return _v_out_sampler[s].sample(rng);
    }

    double log_prob_out(size_t v, size_t u)
    {
        if (_usample == 1)
            return -safelog_fast(_N);

        auto& g = _state._g;
        size_t r = _state._b[v];
        size_t s = _state._b[u];

        size_t ku = 0;
        if (_state._deg_corr)
            ku = (graph_tool::is_directed(_state._g)) ?
                get<0>(get_deg(u, _state._eweight, _state._degs, _state._g)) :
                get<1>(get_deg(u, _state._eweight, _state._degs, _state._g));

        auto&& me = _state._emat.get_me(r, s);
        size_t ers = (me == _state._emat.get_null_edge()) ? 0 : _state._mrs[me];

        if (!graph_tool::is_directed(g) && r == s)
            ers *= 2;

        size_t ns = _state._wr[s];
        size_t er = _state._mrp[r];
        size_t es = (_state._deg_corr) ?
            (graph_tool::is_directed(g) ? _state._mrm[s] : _state._mrp[s]) : 0;

        double lN = safelog_fast(_N);

        if (_state._mrp[r] == 0)
            return -lN;

        if (ers > 0)
        {
            double lp = (safelog_fast(ers) - safelog_fast(er) +
                         safelog_fast(ku + 1) - safelog_fast(es + ns));

            return log_sum_exp(lp + log1p(-_usample), -lN + log(_usample));
        }
        else
        {
            return -lN + log(_usample);
        }
    }

    double log_prob_in(size_t v, size_t u)
    {
        if (_usample == 1)
            return -safelog_fast(_N);

        size_t r = _state._b[v];
        size_t s = _state._b[u];

        size_t ku = 0;
        if (_state._deg_corr)
            ku = get<1>(get_deg(u, _state._eweight, _state._degs, _state._g));

        auto&& me = _state._emat.get_me(s, r);
        size_t esr = (me == _state._emat.get_null_edge()) ? 0 : _state._mrs[me];

        size_t ns = _state._wr[s];
        size_t er = _state._mrm[r];
        size_t es = (_state._deg_corr) ? _state._mrp[s] : 0;

        double lN = safelog_fast(_N);

        if (_state._mrp[r] == 0)
            return -lN;

        if (esr > 0)
        {
            double lp = (safelog_fast(esr) - safelog_fast(er) +
                         safelog_fast(ku + 1) - safelog_fast(es + ns));

            return log_sum_exp(lp + log1p(-_usample), -lN + log(_usample));
        }
        else
        {
            return -lN + log(_usample);
        }
    }

    void check_counts()
    {
        for (auto me : edges_range(_state._bg))
        {
            size_t mrs = _state._mrs[me];
            assert(_rs_sampler.get_prob(_rs_pos[me]) == mrs);

            size_t r = source(me, _state._bg);
            size_t s = target(me, _state._bg);
            std::tie(r, s) = get_edge(r, s);

            if (!graph_tool::is_directed(_state._g) && r == s)
                mrs *= 2;

            assert(_r_out_sampler[r].get_prob(_r_out_pos[me]) == mrs);
            if (graph_tool::is_directed(_state._g) || r != s)
            {
                auto& r_in_sampler = graph_tool::is_directed(_state._g) ?
                    _r_in_sampler : _r_out_sampler;
                assert(r_in_sampler[s].get_prob(_r_in_pos[me]) == mrs);
            }
        }

        for (size_t i = 0; i < _rs_sampler.size(); ++i)
        {
            auto& [r,s] = _rs_sampler[i];
            auto ers = _rs_sampler.get_prob(i);
            auto&& me = _state._emat.get_me(r, s);
            size_t mrs = 0;
            if (me != _state._emat.get_null_edge())
                mrs = _state._mrs[me];
            assert(mrs == ers);
        }

        for (size_t r = 0; r < _r_out_sampler.size(); ++r)
        {
            auto& out_sampler = _r_out_sampler[r];
            for (size_t i = 0; i < out_sampler.size(); ++i)
            {
                if (!out_sampler.is_valid(i))
                    continue;
                auto s = out_sampler[i];
                auto ers = out_sampler.get_prob(i);
                auto&& me = _state._emat.get_me(r, s);
                size_t mrs = 0;
                if (me != _state._emat.get_null_edge())
                    mrs = _state._mrs[me];
                if (!graph_tool::is_directed(_state._g) && r == s)
                    mrs *= 2;
                assert(mrs == ers);
            }
        }

        auto& r_in_sampler = graph_tool::is_directed(_state._g) ?
            _r_in_sampler : _r_out_sampler;

        for (size_t s = 0; s < r_in_sampler.size(); ++s)
        {
            auto& in_sampler = r_in_sampler[s];
            for (size_t i = 0; i < in_sampler.size(); ++i)
            {
                if (!in_sampler.is_valid(i))
                    continue;
                auto r = in_sampler[i];
                auto ers = in_sampler.get_prob(i);
                auto&& me = _state._emat.get_me(r, s);
                size_t mrs = 0;
                if (me != _state._emat.get_null_edge())
                    mrs = _state._mrs[me];
                if (!graph_tool::is_directed(_state._g) && r == s)
                    mrs *= 2;
                assert(mrs == ers);
            }
        }
    }

    std::tuple<size_t, size_t> get_null_edge()
    {
        return _null_edge;
    }

    size_t num_edges()
    {
        return _edges.size();
    }

//private:
    State& _state;
    std::vector<size_t> _tgts;
    size_t _N;
    std::uniform_int_distribution<size_t> _sample_vertex;

    typedef DynamicSampler<std::tuple<size_t, size_t>> rs_sampler_t;
    rs_sampler_t _rs_sampler;
    eprop_map_t<size_t>::type _rs_pos;

    typedef DynamicSampler<size_t> r_sampler_t;
    std::vector<r_sampler_t> _r_out_sampler;
    eprop_map_t<size_t>::type _r_out_pos;

    std::vector<r_sampler_t> _r_in_sampler;
    eprop_map_t<size_t>::type _r_in_pos;

    typedef DynamicSampler<size_t> vsampler_t;
    vector<vsampler_t> __v_in_sampler, _v_out_sampler;
    vector<vsampler_t>& _v_in_sampler;

    vprop_map_t<size_t>::type _v_in_pos;
    vprop_map_t<size_t>::type _v_out_pos;

    std::vector<std::tuple<size_t, size_t>> _edges;
    gt_hash_map<std::tuple<size_t, size_t>, size_t> _edge_pos;
    size_t _E = 0;

    std::vector<size_t> _vertices;

    double _esample;
    double _usample;

    std::tuple<size_t, size_t> _null_edge = \
            {std::numeric_limits<size_t>::max(),
             std::numeric_limits<size_t>::max()};

};


} // graph_tool namespace

#endif // GRAPH_SBM_SAMPLE_EDGE_HH
