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

#ifndef DYNAMIC_SAMPLER_HH
#define DYNAMIC_SAMPLER_HH

#include "random.hh"
#include <boost/mpl/if.hpp>

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class Value>
class DynamicSampler
{
public:
    DynamicSampler() : _back(0), _n_items(0) {}

    DynamicSampler(const vector<Value>& items,
                   const vector<double>& probs)
        : _back(0), _n_items(0)
    {
        for (size_t i = 0; i < items.size(); ++i)
            insert(items[i], probs[i]);
    }

    typedef Value value_type;

    size_t get_left(size_t i)   const { return 2 * i + 1;               }
    size_t get_right(size_t i)  const { return 2 * i + 2;               }
    size_t get_parent(size_t i) const { return i > 0 ? (i - 1) / 2 : 0; }

    template <class RNG>
    size_t sample_idx(RNG& rng) const
    {
        uniform_real_distribution<> sample(0, _tree[0]);
        double u = sample(rng), c = 0;

        size_t pos = 0;
        while (_idx[pos] == _null_idx)
        {
            size_t l = get_left(pos);
            double a = _tree[l];
            if (u < a + c)
            {
                pos = l;
            }
            else
            {
                pos = get_right(pos);
                c += a;
            }
        }
        return _idx[pos];
    }

    template <class RNG>
    const Value& sample(RNG& rng) const
    {
        return _items[sample_idx(rng)];
    }

    template <class RNG>
    const Value& operator()(RNG& rng)
    {
        return sample(rng);
    }

    size_t insert(const Value& v, double w)
    {
        size_t pos;
        if (_free.empty())
        {
            if (_back > 0)
            {
                // move parent to left leaf
                pos = get_parent(_back);
                size_t l = get_left(pos);
                _idx[l] = _idx[pos];
                _ipos[_idx[l]] = l;
                _tree[l] = _tree[pos];
                _idx[pos] = _null_idx;

                // position new item to the right
                _back = get_right(pos);
            }

            pos = _back;
            check_size(pos);

            _idx[pos] = _items.size();
            _items.push_back(v);
            _valid.push_back(true);
            _ipos.push_back(pos);
            _tree[pos] = w;
            _back++;
            check_size(_back);
        }
        else
        {
            pos = _free.back();
            auto i = _idx[pos];
            _items[i] = v;
            _valid[i] = true;
            _tree[pos] = w;
            _free.pop_back();
        }

        insert_leaf_prob(pos);
        _n_items++;

        return _idx[pos];
    }

    void remove(size_t i)
    {
        size_t pos = _ipos[i];
        remove_leaf_prob(pos);
        _tree[pos] = 0;
        _free.push_back(pos);
        _items[i] = Value();
        _valid[i] = false;
        _n_items--;
    }

    void update(size_t i, double w, bool delta)
    {
        size_t pos = _ipos[i];
        assert(_tree[pos] > 0 || w > 0);
        remove_leaf_prob(pos);
        if (delta)
            _tree[pos] += w;
        else
            _tree[pos] = w;
        insert_leaf_prob(pos);
        assert(_tree[pos] >= 0);
    }

    void clear(bool shrink=false)
    {
        _items.clear();
        _ipos.clear();
        _tree.clear();
        _idx.clear();
        _free.clear();
        _valid.clear();
        if (shrink)
        {
            _items.shrink_to_fit();
            _ipos.shrink_to_fit();
            _tree.shrink_to_fit();
            _idx.shrink_to_fit();
            _free.shrink_to_fit();
            _valid.shrink_to_fit();
        }
        _back = 0;
        _n_items = 0;
    }

    void rebuild()
    {
        vector<Value> items;
        vector<double> probs;

        for (size_t i = 0; i < _tree.size(); ++i)
        {
            if (_idx[i] == _null_idx)
                continue;
            size_t j = _idx[i];
            if (!_valid[j])
                continue;
            items.push_back(_items[j]);
            probs.push_back(_tree[i]);
        }

        clear(true);

        for (size_t i = 0; i < items.size(); ++i)
            insert(items[i], probs[i]);
    }

    const Value& operator[](size_t i) const
    {
        return _items[i];
    }

    double get_prob(size_t i) const
    {
        return _tree[_ipos[i]];
    }

    bool is_valid(size_t i) const
    {
        return ((i < _items.size()) && _valid[i]);
    }

    const auto& items() const
    {
        return _items;
    }

    auto begin() const
    {
        return _items.begin();
    }

    auto end() const
    {
        return _items.end();
    }

    size_t size() const
    {
        return _items.size();
    }

    bool empty() const
    {
        return _n_items == 0;
    }

private:
    void check_size(size_t i)
    {
        if (i >= _tree.size())
        {
            _idx.resize(i + 1, _null_idx);
            _tree.resize(i + 1, 0);
        }
    }

    void remove_leaf_prob(size_t i)
    {
        size_t parent = i;
        double w = _tree[i];

        while (parent > 0)
        {
            parent = get_parent(parent);
            _tree[parent] -= w;
            assert(_tree[parent] >= 0);
        }
    }

    void insert_leaf_prob(size_t i)
    {
        size_t parent = i;
        double w = _tree[i];

        while (parent > 0)
        {
            parent = get_parent(parent);
            _tree[parent] += w;
        }
    }

    bool check_probs()
    {
        for (size_t i = 0; i < _tree.size(); ++i)
        {
            if (_idx[i] != _null_idx)
                continue;
            assert(get_right(i) >= _tree.size() || _tree[i] == _tree[get_left(i)] + _tree[get_right(i)]);
        }
        return true;
    }

    vector<Value>  _items;
    vector<size_t> _ipos;  // position of the item in the tree

    vector<double> _tree;  // tree nodes with weight sums
    vector<size_t> _idx;   // index in _items
    int _back;             // last item in tree

    vector<size_t> _free;  // empty leafs
    vector<bool> _valid;   // non-removed items
    size_t _n_items;

    constexpr static size_t _null_idx = numeric_limits<size_t>::max();
};



} // namespace graph_tool

#endif // DYNAMIC_SAMPLER_HH
