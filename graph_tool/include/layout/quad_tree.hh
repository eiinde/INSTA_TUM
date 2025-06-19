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

#ifndef QUAD_TREE_HH
#define QUAD_TREE_HH

#include <array>
#include <vector>
#include <tuple>
#include <cmath>

template <class Value>
Value pow2(Value x)
{
    return x * x;
}

template <class Val, class Weight>
class QuadTree
{
public:

    typedef typename std::array<Val, 2> pos_t;

    class TreeNode
    {
    public:
        template <class Pos>
        TreeNode(const Pos& ll, const Pos& ur, size_t level)
            : _ll(ll), _ur(ur), _cm{0,0}, _level(level), _count(0) {}

        double get_w()
        {
            return std::sqrt(pow2(_ur[0] - _ll[0]) +
                             pow2(_ur[1] - _ll[1]));
        }

        template <class Pos>
        void get_cm(Pos& cm)
        {
            for (size_t i = 0; i < 2; ++i)
                cm[i] = _cm[i] / _count;
        }

        Weight get_count()
        {
            return _count;
        }

        friend class QuadTree;
    private:
        pos_t _ll;
        pos_t _ur;
        std::array<double,2> _cm;
        size_t _level;
        Weight _count;
        size_t _leaves = std::numeric_limits<size_t>::max();
    };

    QuadTree(): _max_level(0) {}

    template <class Pos>
    QuadTree(const Pos& ll, const Pos& ur, int max_level, size_t n)
        : _tree(1, {ll, ur, 0}), _dense_leaves(1), _max_level(max_level)
    {
        _tree.reserve(n);
        _dense_leaves.reserve(n);
    }

    auto& operator[](size_t pos)
    {
        return _tree[pos];
    }

    size_t get_leaves(size_t pos)
    {
        auto& node = _tree[pos];

        size_t level = _tree[pos]._level;
        if (level >= _max_level)
            return _tree.size();

        if (node._leaves >= _tree.size())
        {
            auto ll = node._ll;
            auto ur = node._ur;
            auto level = node._level;

            node._leaves = _tree.size();
            //_tree.reserve(_tree.size() + 4);
            for (size_t i = 0; i < 4; ++i)
            {
                pos_t lll = ll, lur = ur;
                if (i % 2)
                    lll[0] += (ur[0] - ll[0]) / 2;
                else
                    lur[0] -= (ur[0] - ll[0]) / 2;
                if (i / 2)
                    lll[1] += (ur[1] - ll[1]) / 2;
                else
                    lur[1] -= (ur[1] - ll[1]) / 2;
                _tree.emplace_back(lll, lur, level + 1);
            }
            _dense_leaves.resize(_tree.size());
        }

        return _tree[pos]._leaves;
    }

    auto& get_dense_leaves(size_t pos)
    {
        return _dense_leaves[pos];
    }

    template <class Pos>
    size_t get_branch(size_t pos, Pos& p)
    {
        auto& n = _tree[pos];
        int i = p[0] > (n._ll[0] + (n._ur[0] - n._ll[0]) / 2);
        int j = p[1] > (n._ll[1] + (n._ur[1] - n._ll[1]) / 2);
        return i + 2 * j;
    }

    template <class Pos>
    void put_pos(size_t pos, Pos& p, Weight w)
    {
        while (pos < _tree.size())
        {
            auto& node = _tree[pos];

            node._count += w;
            node._cm[0] += p[0] * w;
            node._cm[1] += p[1] * w;

            if (node._level >= _max_level || node._count == w)
            {
                _dense_leaves[pos].emplace_back(pos_t{p[0], p[1]}, w);
                pos = _tree.size();
            }
            else
            {
                auto leaves = get_leaves(pos);

                if (!_dense_leaves[pos].empty())
                {
                    // move dense leaves down
                    for (auto& leaf : _dense_leaves[pos])
                    {
                        auto& lp = get<0>(leaf);
                        auto& lw = get<1>(leaf);
                        put_pos(leaves + get_branch(pos, lp), lp, lw);
                    }
                    _dense_leaves[pos].clear();
                }
                pos = leaves + get_branch(pos, p);
            }
        }
    }

    size_t size()
    {
        return _tree.size();
    }

private:
    std::vector<TreeNode> _tree;
    std::vector<std::vector<std::tuple<pos_t,Weight>>> _dense_leaves;
    size_t _max_level;
};

#endif // QUAD_TREE_HH
