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

#ifndef IDX_MAP_HH
#define IDX_MAP_HH

#include <vector>
#include <utility>
#include <limits>

template <class T>
size_t get_size(T n)
{
    size_t k = 1;
    while (k < size_t(n))
        k <<= 1;
    return k;
}

template <class Key, class T, bool shared_pos=false, bool check_size=true>
class idx_map
{
public:
    typedef Key key_type;
    typedef T mapped_type;
    typedef std::pair<const Key, T> value_type;
    typedef typename std::vector<std::pair<Key,T>>::iterator iterator;
    typedef typename std::vector<std::pair<Key,T>>::const_iterator const_iterator;

    idx_map(size_t size = 0)
    {
        if constexpr (shared_pos)
            _pos = nullptr;
        else
            _pos.resize(size, _null);
    }

    idx_map(std::vector<size_t>& pos) : _pos(&pos) {}

    auto& get_pos()
    {
        if constexpr (shared_pos)
            return *_pos;
        else
            return _pos;
    }

    template <class P>
    std::pair<iterator,bool> insert(P&& value)
    {
        auto& pos = get_pos();
        if constexpr (check_size)
        {
            if (pos.size() <= size_t(value.first))
                pos.resize(get_size(value.first + 1), _null);
        }
        size_t& idx = pos[value.first];
        if (idx == _null || shared_pos)
        {
            idx = _items.size();
            _items.push_back(value);
            return std::make_pair(begin() + idx, true);
        }
        else
        {
            _items[idx].second = value.second;
            return std::make_pair(begin() + idx, false);
        }
    }

    size_t erase(const Key& k)
    {
        auto& pos = get_pos();
        size_t& idx = pos[k];
        if (idx == _null && !shared_pos)
            return 0;
        auto& back = _items.back();
        pos[back.first] = idx;
        _items[idx] = back;
        _items.pop_back();
        if constexpr (!shared_pos)
            idx = _null;
        return 1;
    }

    iterator erase(const_iterator pos)
    {
        size_t idx = pos - begin();
        erase(pos->first);
        return begin() + idx;
    }

    [[gnu::always_inline]]
    auto& operator[](const Key& key)
    {
        auto iter = find(key);
        if (iter == end())
            iter = insert(std::make_pair(key, T())).first;
        return iter->second;
    }

    [[gnu::always_inline]]
    iterator find(const Key& key)
    {
        auto& pos = get_pos();
        if constexpr (check_size)
        {
            if (size_t(key) >= pos.size())
                return end();
        }
        size_t idx = pos[key];
        if constexpr (shared_pos)
        {
            if (idx >= _items.size() || _items[idx].first != key)
                return end();
        }
        else
        {
            if (idx == _null)
                return end();
        }
        return begin() + idx;
    }

    const_iterator find(const Key& key) const
    {
        return const_cast<decltype(this)>(this)->find(key);
    }

    void clear()
    {
        auto& pos = get_pos();
        if constexpr (shared_pos || !check_size)
        {
            for (auto k : _items)
                pos[k.first] = _null;
        }
        else
        {
            pos.clear();
        }
        _items.clear();
    }

    void shrink_to_fit()
    {
        auto& pos = get_pos();
        _items.shrink_to_fit();
        if (_items.empty() && !check_size)
            pos.clear();
        pos.shrink_to_fit();
    }

    iterator begin() { return _items.begin(); }
    iterator end() { return _items.end(); }
    const_iterator begin() const { return _items.begin(); }
    const_iterator end() const { return _items.end(); }

    size_t size() { return _items.size(); }
    bool empty() { return _items.empty(); }

private:
    std::vector<std::pair<Key,T>> _items;
    std::conditional_t<shared_pos,
                       std::vector<size_t>*,
                       std::vector<size_t>> _pos;
    inline static constexpr size_t _null = std::numeric_limits<size_t>::max();
};


template <class Key, bool shared_pos=false, bool check_size=true>
class idx_set
{
public:
    typedef Key key_type;
    typedef Key value_type;
    typedef typename std::vector<Key>::iterator iterator;
    typedef typename std::vector<Key>::const_iterator const_iterator;
    typedef std::vector<size_t> pos_t;

    idx_set(size_t size = 0)
    {
        if constexpr (shared_pos)
            _pos = nullptr;
        else
            _pos.resize(size, _null);
    }

    idx_set(std::vector<size_t>& pos) : _pos(&pos) {}

    const pos_t& get_pos() const
    {
        if constexpr (shared_pos)
            return *_pos;
        else
            return _pos;
    }

    std::pair<const_iterator,bool> insert(const Key& k)
    {
        pos_t& pos = const_cast<pos_t&>(get_pos());
        if constexpr (check_size)
        {
            if (pos.size() <= size_t(k))
                pos.resize(get_size(k + 1), _null);
        }
        size_t& idx = pos[k];
        if (idx == _null || shared_pos)
        {
            idx = _items.size();
            _items.push_back(k);
            return std::make_pair(begin() + idx, true);
        }
        else
        {
            return std::make_pair(begin() + idx, false);
        }
    }

    size_t erase(const Key& k)
    {
        pos_t& pos = const_cast<pos_t&>(get_pos());
        size_t& idx = pos[k];
        if (idx == _null && !shared_pos)
            return 0;
        auto& back = _items.back();
        pos[back] = idx;
        _items[idx] = back;
        _items.pop_back();
        if constexpr (!shared_pos)
            idx = _null;
        return 1;
    }

    const_iterator erase(const_iterator pos)
    {
        size_t idx = pos - begin();
        erase(pos->first);
        return begin() + idx;
    }

    const_iterator find(const Key& key) const
    {
        const auto& pos = get_pos();
        if constexpr (check_size)
        {
            if (size_t(key) >= pos.size())
                return end();
        }
        size_t idx = pos[key];
        if (idx == _null)
            return end();
        return begin() + idx;
    }

    void clear()
    {
        auto& pos = const_cast<pos_t&>(get_pos());
        if constexpr (shared_pos || !check_size)
        {
            for (auto k : _items)
                pos[k] = _null;
        }
        else
        {
            pos.clear();
        }
        _items.clear();
    }

    void shrink_to_fit()
    {
        auto& pos = const_cast<pos_t&>(get_pos());
        _items.shrink_to_fit();
        if (_items.empty() && !check_size)
            pos.clear();
        pos.shrink_to_fit();
    }

    const_iterator begin() const { return _items.begin(); }
    const_iterator end() const { return _items.end(); }

    size_t size() { return _items.size(); }
    bool empty() { return _items.empty(); }

private:
    std::vector<Key> _items;
    std::conditional_t<shared_pos, pos_t*, pos_t> _pos;
    inline static constexpr size_t _null = std::numeric_limits<size_t>::max();
};

#endif // IDX_MAP_HH
