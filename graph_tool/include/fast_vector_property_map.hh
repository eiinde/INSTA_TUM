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

// Copyright (C) Vladimir Prus 2003.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/libs/graph/vector_property_map.html for
// documentation.
//

//
// This is a modification of boost's vector property map which optionally
// disables bound checking for better performance.
//

#ifndef FAST_VECTOR_PROPERTY_MAP_HH
#define FAST_VECTOR_PROPERTY_MAP_HH

#include <boost/property_map/property_map.hpp>
#include <memory>
#include <vector>

namespace boost {

template<typename T, typename IndexMap>
class unchecked_vector_property_map;

template<typename T, typename IndexMap = identity_property_map>
class checked_vector_property_map
    : public boost::put_get_helper<
              typename std::iterator_traits<
                  typename std::vector<T>::iterator >::reference,
              checked_vector_property_map<T, IndexMap> >
{
public:
    typedef typename property_traits<IndexMap>::key_type  key_type;
    typedef T value_type;
    typedef typename std::iterator_traits<
        typename std::vector<T>::iterator >::reference reference;
    typedef boost::lvalue_property_map_tag category;

    template<typename Type, typename Index>
    friend class unchecked_vector_property_map;

    typedef unchecked_vector_property_map<T, IndexMap> unchecked_t;
    typedef IndexMap index_map_t;
    typedef checked_vector_property_map<T,IndexMap> self_t;

    checked_vector_property_map(const IndexMap& idx = IndexMap())
        : _store(std::make_shared<std::vector<T>>()), _index(idx) {}

    checked_vector_property_map(unsigned initial_size,
                                const IndexMap& idx = IndexMap())
        : _store(std::make_shared<std::vector<T>>(initial_size)), _index(idx) {}

    typename std::vector<T>::iterator storage_begin()
    {
        return _store->begin();
    }

    typename std::vector<T>::iterator storage_end()
    {
        return _store->end();
    }

    typename std::vector<T>::const_iterator storage_begin() const
    {
        return _store->begin();
    }

    typename std::vector<T>::const_iterator storage_end() const
    {
        return _store->end();
    }

    void reserve(size_t size) const
    {
        if (size > _store->size())
            _store->resize(size);
    }

    void resize(size_t size) const
    {
        _store->resize(size);
    }

    void shrink_to_fit() const
    {
        _store->shrink_to_fit();
    }

    std::vector<T>& get_storage() const { return *_store; }

    void swap(checked_vector_property_map& other)
    {
        _store->swap(*other._store);
    }

    unchecked_t get_unchecked(size_t size = 0) const
    {
        reserve(size);
        return unchecked_t(*this, size);
    }

    // deep copy
    checked_vector_property_map copy() const
    {
        checked_vector_property_map pmap(_index);
        *(pmap._store) = *_store;
        return pmap;
    }

public:
    [[gnu::always_inline]]
    reference operator[](const key_type& v) const {
        auto i = get(_index, v);
        auto& store = *_store;
        if (static_cast<size_t>(i) >= store.size()) {
            store.resize(i + 1);
        }
        return store[i];
    }
protected:
    std::shared_ptr<std::vector<T>> _store;
    IndexMap _index;
};

template<typename T, typename IndexMap = identity_property_map>
class unchecked_vector_property_map
    : public boost::put_get_helper<
                typename std::iterator_traits<
                    typename std::vector<T>::iterator >::reference,
                unchecked_vector_property_map<T, IndexMap> >
{
public:
    typedef typename property_traits<IndexMap>::key_type  key_type;
    typedef T value_type;
    typedef typename std::iterator_traits<
        typename std::vector<T>::iterator >::reference reference;
    typedef boost::lvalue_property_map_tag category;

    typedef checked_vector_property_map<T, IndexMap> checked_t;

    unchecked_vector_property_map(const checked_t& checked = checked_t(),
                                  size_t size = 0)
        : _checked(checked)
    {
        if (size > 0 && _checked._store->size() < size)
            _checked._store->resize(size);
    }

    unchecked_vector_property_map(const IndexMap& index_map,
                                  size_t size = 0)
        : _checked(size, index_map)
    {
    }

    void reserve(size_t size) const { _checked.reserve(size); }
    void resize(size_t size) const { _checked.resize(size); }
    void shrink_to_fit() const { _checked.shrink_to_fit(); }


    [[gnu::always_inline]] [[gnu::flatten]]
    reference operator[](const key_type& v) const
    {
        return (*_checked._store)[get(_checked._index, v)];
    }

    std::vector<T>& get_storage() const { return _checked.get_storage(); }

    void swap(unchecked_vector_property_map& other)
    {
        get_storage().swap(other.get_storage());
    }

    checked_t get_checked() {return _checked;}

    // deep copy
    unchecked_vector_property_map copy() const
    {
        return _checked.copy().get_unchecked();
    }

private:
    checked_t _checked;
};

template<typename T, typename IndexMap>
checked_vector_property_map<T, IndexMap>
make_checked_vector_property_map(IndexMap index)
{
    return checked_vector_property_map<T, IndexMap>(index);
}

template<typename T, typename IndexMap>
unchecked_vector_property_map<T, IndexMap>
make_unchecked_vector_property_map(IndexMap index)
{
    return unchecked_vector_property_map<T, IndexMap>(index);
}


template <class Type, class Index>
unchecked_vector_property_map<Type, Index>
get_unchecked(checked_vector_property_map<Type, Index> prop)
{
    return prop.get_unchecked();
}

template <class Prop>
Prop
get_unchecked(Prop prop)
{
    return prop;
}

template <class Type, class Index>
checked_vector_property_map<Type, Index>
get_checked(unchecked_vector_property_map<Type, Index> prop)
{
    return prop.get_checked();
}

template <class Prop>
Prop
get_checked(Prop prop)
{
    return prop;
}


}

#endif
