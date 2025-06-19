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

#ifndef GRAPH_BLOCKMODEL_ELIST_HH
#define GRAPH_BLOCKMODEL_ELIST_HH

#include "../../generation/sampler.hh"
#include "../../generation/dynamic_sampler.hh"

namespace graph_tool
{

// ====================================
// Construct and manage half-edge lists
// ====================================

class EGroups
{
public:
    template <class BGraph, class Eprop>
    EGroups(BGraph& bg, Eprop& mrs)
        : _egroups(num_vertices(bg)),
          _pos(num_vertices(bg))
    {
        for (auto e : edges_range(bg))
        {
            insert_edge(source(e, bg), target(e, bg), mrs[e]);
            insert_edge(target(e, bg), source(e, bg), mrs[e]);
        }

        check(bg, mrs);
    }

    void add_block()
    {
        _egroups.emplace_back();
        _pos.emplace_back();
    }

    void insert_edge(size_t s, size_t t, int weight)
    {
        auto& pos = _pos[s];
        auto iter = pos.find(t);
        if (iter == pos.end())
            iter = pos.insert({t, _null_pos}).first;
        insert_edge(t, weight, _egroups[s], iter->second);
        if (iter->second == _null_pos)
            pos.erase(iter);
    }

    template <class EV>
    void insert_edge(size_t t, int weight, EV& elist, size_t& pos)
    {
        if (pos != _null_pos)
        {
            assert(elist.is_valid(pos) && elist[pos] == t);
            elist.update(pos, weight, true);
            if (elist.get_prob(pos) == 0)
            {
                elist.remove(pos);
                pos = _null_pos;
            }
        }
        else
        {
            if (weight > 0)
                pos = elist.insert(t, weight);
            else
                pos = _null_pos;
        }
    }

    template <class RNG>
    size_t sample_edge(size_t r, RNG& rng)
    {
        auto s = _egroups[r].sample(rng);
        assert(s != numeric_limits<size_t>::max());
        return s;
    }

    template <class Eprop, class BGraph>
    void check([[maybe_unused]] BGraph& bg, [[maybe_unused]] Eprop& mrs)
    {
#ifndef NDEBUG
        for (auto e : edges_range(bg))
        {
            auto r = source(e, bg);
            auto s = target(e, bg);

            assert(r < _pos.size());
            auto& pos = _pos[r];
            auto iter = pos.find(s);
            assert(iter != pos.end());

            auto p = _egroups[r].get_prob(iter->second);

            if (!graph_tool::is_directed(bg) || r == s)
            {
                assert(p == mrs[e] * (r == s ? 2 : 1));
            }
            else
            {
                auto ne = edge(s, r, bg);
                if (ne.second)
                    assert(p == mrs[e] + mrs[ne.first]);
                else
                    assert(p == mrs[e]);
            }
        }
#endif
    }

private:
    vector<DynamicSampler<size_t>> _egroups;
    vector<gt_hash_map<size_t, size_t>> _pos;
    static constexpr size_t _null_pos = numeric_limits<size_t>::max();
};

} // namespace graph_tool

#endif //GRAPH_BLOCKMODEL_ELIST_HH
