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

#ifndef GRAPH_PROPERTIES_HH
#define GRAPH_PROPERTIES_HH

#include <string>
#include <vector>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <boost/python/object.hpp>
#include <boost/python/extract.hpp>
#include <boost/lexical_cast.hpp>

#include <boost/property_map/property_map.hpp>
#include "fast_vector_property_map.hh"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/bind.hpp>

#include "graph_exceptions.hh"

// this file provides general functions for manipulating graph properties

namespace graph_tool
{

// Metaprogramming
// ===============
//
// Metafunctions and data structures to deal with property maps

// Global Property Types
// ---------------------
// global property types. only these types are allowed in property maps
// Note: we must avoid a vector<bool> (and bools in general) since it is quite
//       broken, and use a vector<uint8_t> instead!
//       see: http://www.gotw.ca/publications/N1211.pdf

typedef boost::mpl::vector15<uint8_t, int16_t, int32_t, int64_t, double,
                             long double, std::string, std::vector<uint8_t>,
                             std::vector<int16_t>, std::vector<int32_t>,
                             std::vector<int64_t>, std::vector<double>,
                             std::vector<long double>, std::vector<std::string>,
                             boost::python::object> value_types;

extern const char* type_names[]; // respective type names (defined in
                                 // graph_properties.cc)

// scalar types: types contained in value_types which are scalar
typedef boost::mpl::vector6<uint8_t, int16_t, int32_t, int64_t, double, long double>
    scalar_types;

// integer_types: scalar types which are integer
typedef boost::mpl::vector4<uint8_t, int16_t, int32_t, int64_t> integer_types;

// floating_types: scalar types which are floating point
typedef boost::mpl::vector2<double, long double> floating_types;

struct make_vector
{
    template <class ValueType> struct apply { typedef std::vector<ValueType> type; };
};

// scalar_vector_types: vector types with floating point values
typedef boost::mpl::transform<scalar_types,make_vector>::type scalar_vector_types;

// integer_vector_types: vector types with floating point values
typedef boost::mpl::transform<integer_types,make_vector>::type integer_vector_types;

// floating_vector_types: vector types with floating point values
typedef boost::mpl::transform<floating_types,make_vector>::type floating_vector_types;

//
// Property Map Types
// ------------------

// metafunction to generate the correct property map type given a value type and
// an index map
struct property_map_type
{
    template <class ValueType, class IndexMap>
    struct apply
    {
        typedef boost::checked_vector_property_map<ValueType,IndexMap> type;
    };
};

// metafunction to get the sequence of property map types of ValueTypes and
// IndexMap
struct property_map_types
{
   // this wraps an unary metafunction class Bind into a unary metafunction,
   // i.e., it is an identity operation. I'm not sure why it's necessary, but
   // using pure unary bind expressions below didn't work for me, and this fixed
   // it.
    template <class Bind>
    struct bind_wrap1
    {
        template <class T1> struct apply
        { typedef typename Bind::template apply<T1>::type type; };
    };

    template <class ValueTypes, class IndexMap,
              class IncludeIndexMap = boost::mpl::bool_<true>>
    struct apply
    {
        typedef typename boost::mpl::transform<
            ValueTypes,
            bind_wrap1<boost::mpl::bind2<property_map_type,
                                         boost::mpl::_1,
                                         IndexMap>>
            >::type scalar_properties;

        // put index map itself
        typedef typename boost::mpl::if_<
            IncludeIndexMap,
            typename boost::mpl::push_back<scalar_properties,IndexMap>::type,
            scalar_properties
            >::type type;
    };
};

// Property map manipulation
// =========================
//
// Functions which deal with several aspects of property map manipulation


// this functor tests whether or not a given boost::any object holds a type
// contained in a given type Sequence
template <class Sequence>
struct belongs
{
    struct get_type
    {
        get_type(const boost::any& val, bool& found)
            : _val(val), _found(found) {}

        template <class Type>
        void operator()(Type) const
        {
            const Type* ptr = boost::any_cast<Type>(&_val);
            if (ptr != 0)
                _found = true;
        }

        const boost::any& _val;
        bool& _found;
    };

    bool operator()(const boost::any& prop)
    {
        bool found = false;
        boost::mpl::for_each<Sequence>(get_type(prop, found));
        return found;
    }
};

// this will return the name of a given type
template <class TypeSequence = value_types,
          class NamedSequence = value_types>
class get_type_name
{
public:
    get_type_name(const char* names[] = type_names)
        : _type_names(names)
    {
        if (_all_names.empty())
        {
            boost::mpl::for_each<TypeSequence>
                ([this](auto t){ get_all_names()(t, _type_names, _all_names); });
        }
    }

    const std::string& operator()(const std::type_info& type) const
    {
        std::string const* name;
        boost::mpl::for_each<TypeSequence>
            ([this, &type, &name](auto t){ find_name()(t, type, _all_names, name); });
        return *name;
    }

    const std::vector<std::string>& all_names() const
    {
        return _all_names;
    }

private:
    struct find_name
    {
        template <class Type>
        void operator()(Type, const std::type_info& type,
                        const std::vector<std::string>& all_names,
                        std::string const*& name) const
        {
            size_t index = boost::mpl::find<TypeSequence,Type>::type::pos::value;
            if (type == typeid(Type))
                name = &all_names[index];
        }
    };

    struct get_all_names
    {
        typedef void result_type;
        template <class Type>
        void operator()(Type, const char** t_names,
                        std::vector<std::string>& names) const
        {
            size_t index = boost::mpl::find<NamedSequence,Type>::type::pos::value;
            names.push_back(t_names[index]);
        }
    };

    const char** _type_names;
    std::vector<std::string> _all_names;
};

//
// Extra Property Map Types
// ========================


// handle type convertions

// generic types
template <class Type1, class Type2>
struct convert
{
    Type1 operator()(const Type2& v) const
    {
        if constexpr (std::is_same<Type1, boost::python::object>::value)
        {
            return boost::python::object(v);
        }
        else if constexpr (std::is_convertible<Type2,Type1>::value)
        {
            return Type1(v);
        }
        else if constexpr (std::is_same<Type2, boost::python::object>::value)
        {
            boost::python::extract<Type1> x(v);
            if (x.check())
                return x();
            else
                throw boost::bad_lexical_cast();
        }
        else if constexpr (std::is_same<Type2, std::string>::value)
        {
            //uint8_t is not char, it is bool!
            if (std::is_same<Type1, uint8_t>::value)
                return convert<Type1,int>()(boost::lexical_cast<int>(v));
            else
                return boost::lexical_cast<Type1>(v);
        }
        else if constexpr (std::is_same<Type1, std::string>::value)
        {
            //uint8_t is not char, it is bool!
            if (std::is_same<Type2, uint8_t>::value)
                return boost::lexical_cast<std::string>(convert<int,Type2>()(v));
            else
                return boost::lexical_cast<std::string>(v);
        }
        else
        {
            return specific_convert<Type1, Type2>()(v);
        }
    }

    // default action
    template <class T1, class T2>
    struct specific_convert
    {
        T1 operator()(const T2&) const
        {
            throw boost::bad_lexical_cast();
        }
    };

    // specific specializations

    // vectors
    template <class T1, class T2>
    struct specific_convert<std::vector<T1>, std::vector<T2>>
    {
        std::vector<T1> operator()(const std::vector<T2>& v) const
        {
            std::vector<T1> v2(v.size());
            convert<T1,T2> c;
            for (size_t i = 0; i < v.size(); ++i)
                v2[i] = c(v[i]);
            return v2;
        }
    };

};

// No op
template <class Type1>
struct convert<Type1, Type1>
{
    const Type1& operator()(const Type1& v) const
    {
        return v;
    }
};

// the following class wraps a generic property map, so it can be used as a
// property with a given Key and Value type. The keys and values are converted
// to the desired Key and Value type, which may cause a performance impact,
// since virtual functions are used. Should be used only when property map
// access time is not crucial
template <class Value, class Key,
          template <class T1, class T2> class Converter = convert>
class DynamicPropertyMapWrap
{
public:
    typedef Value value_type;
    typedef Value reference;
    typedef Key key_type;
    typedef boost::read_write_property_map_tag category;

    template <class PropertyTypes>
    DynamicPropertyMapWrap(boost::any pmap, PropertyTypes)
    {
        ValueConverter* converter = 0;
        boost::mpl::for_each<PropertyTypes>
            ([&pmap, &converter](auto t){ choose_converter()(t, pmap, converter); });
        if (converter == 0)
            throw boost::bad_lexical_cast();
        else
            _converter = std::shared_ptr<ValueConverter>(converter);
    }

    DynamicPropertyMapWrap() {}

    Value get(const Key& k) const
    {
        return (*_converter).get(k);
    }

    void put(const Key& k, const Value& val)
    {
        (*_converter).put(k, val);
    }

private:
    class ValueConverter
    {
    public:
        virtual Value get(const Key& k) = 0;
        virtual void put(const Key& k, const Value& val) = 0;
        virtual ~ValueConverter() {}
    };

    template <class PropertyMap>
    class ValueConverterImp: public ValueConverter
    {
    public:
        ValueConverterImp(PropertyMap pmap): _pmap(pmap) {}
        virtual ~ValueConverterImp() {}
        typedef typename boost::property_traits<PropertyMap>::value_type val_t;
        typedef typename boost::property_traits<PropertyMap>::key_type key_t;

        virtual Value get(const Key& k)
        {
            return get_dispatch(_pmap, k,
                                std::is_convertible<typename boost::property_traits<PropertyMap>::category,
                                                    boost::readable_property_map_tag>());
        }

        virtual void put(const Key& k, const Value& val)
        {
            put_dispatch(_pmap, k, _c_put(val),
                         std::is_convertible<typename boost::property_traits<PropertyMap>::category,
                                             boost::writable_property_map_tag>());
        }

        template <class PMap>
        Value get_dispatch(PMap&& pmap, const typename boost::property_traits<std::remove_reference_t<PMap>>::key_type& k,
                           std::true_type)
        {
            return _c_get(boost::get(pmap, k));
        }

        template <class PMap>
        Value get_dispatch(PMap&&, const typename boost::property_traits<std::remove_reference_t<PMap>>::key_type&,
                           std::false_type)
        {
            throw graph_tool::ValueException("Property map is not readable.");
        }

        template <class PMap>
        void put_dispatch(PMap&& pmap, const typename boost::property_traits<std::remove_reference_t<PMap>>::key_type& k,
                          typename boost::property_traits<std::remove_reference_t<PMap>>::value_type val,
                          std::true_type)
        {
            boost::put(pmap, k, val);
        }

        template <class PMap>
        void put_dispatch(PMap&&, const typename boost::property_traits<std::remove_reference_t<PMap>>::key_type&,
                          typename boost::property_traits<std::remove_reference_t<PMap>>::value_type,
                          std::false_type)
        {
            throw ValueException("Property map is not writable.");
        }

    private:
        PropertyMap _pmap;
        Converter<Value, val_t> _c_get;
        Converter<val_t, Value> _c_put;
    };

    struct choose_converter
    {
        typedef void return_type;
        template <class PropertyMap>
        void operator()(PropertyMap, boost::any& dmap,
                        ValueConverter*& converter) const
        {
            if (typeid(PropertyMap) == dmap.type())
                converter = new ValueConverterImp<PropertyMap>
                    (boost::any_cast<PropertyMap>(dmap));
        }
    };

    std::shared_ptr<ValueConverter> _converter;
};

template <class Value, class Key, class ConvKey>
Value get(const graph_tool::DynamicPropertyMapWrap<Value,Key>& pmap,
          ConvKey k)
{
    Key key = k;
    return pmap.get(key);
}

template <class Value, class Key>
void put(graph_tool::DynamicPropertyMapWrap<Value,Key>& pmap,
         Key k, const Value& val)
{
    pmap.put(k, val);
}

// the following is hash functor which, will hash a vertex or edge descriptor
// based on its index
template <class IndexMap>
class DescriptorHash
{
public:
    DescriptorHash() {}
    DescriptorHash(IndexMap index_map): _index_map(index_map) {}
    size_t operator()(typename IndexMap::key_type const& d) const
    {
        std::hash<typename IndexMap::value_type> hash;
        return hash(_index_map[d]);
    }
private:
    IndexMap _index_map;
};

// the following is a property map based on a hashed container, which uses the
// above hash function for vertex or edge descriptors
template <class IndexMap, class Value>
class HashedDescriptorMap
    : public boost::put_get_helper<Value&, HashedDescriptorMap<IndexMap,Value>>
{
public:
    typedef DescriptorHash<IndexMap> hashfc_t;
    typedef std::unordered_map<typename IndexMap::key_type,Value,hashfc_t>
        map_t;
    typedef boost::associative_property_map<map_t> prop_map_t;

    typedef typename boost::property_traits<prop_map_t>::value_type value_type;
    typedef typename boost::property_traits<prop_map_t>::reference reference;
    typedef typename boost::property_traits<prop_map_t>::key_type key_type;
    typedef typename boost::property_traits<prop_map_t>::category category;

    HashedDescriptorMap(IndexMap index_map)
        : _base_map(std::make_shared<map_t>(0, hashfc_t(index_map))),
        _prop_map(*_base_map) {}
    HashedDescriptorMap(){}

    reference operator[](const key_type& k) { return _prop_map[k]; }
    const reference operator[](const key_type& k) const { return _prop_map[k]; }

private:
    std::shared_ptr<map_t> _base_map;
    prop_map_t _prop_map;
};


// this wraps a container as a property map which is automatically initialized
// with a given default value
template <class Container>
class InitializedPropertyMap
    : public boost::put_get_helper<typename Container::value_type::second_type&,
                                   InitializedPropertyMap<Container>>
{
public:
    typedef typename Container::value_type::second_type value_type;
    typedef value_type& reference;
    typedef typename Container::key_type key_type;
    typedef boost::read_write_property_map_tag category;

    InitializedPropertyMap(Container& base_map, value_type def)
        : _base_map(&base_map), _default(def) {}
    InitializedPropertyMap(){}

    reference operator[](const key_type& k)
    {
        return get(k);
    }

    reference operator[](const key_type& k) const
    {
        return get(k);
    }

    reference get(const key_type& k) const
    {
        typename Container::iterator val;
        val = _base_map->find(k);
        if (val == _base_map->end())
            val = _base_map->insert({k, _default}).first;
        return val->second;
    }

private:
    Container* _base_map;
    value_type _default;
};

// the following is a property map which always returns the same value
template <class Value, class Key>
class ConstantPropertyMap
    : public boost::put_get_helper<Value, ConstantPropertyMap<Value,Key>>
{
public:
    typedef Value value_type;
    typedef value_type& reference;
    typedef Key key_type;
    typedef boost::readable_property_map_tag category;

    ConstantPropertyMap(const value_type& c): _c(c) {}
    ConstantPropertyMap(): _c() {}
    ConstantPropertyMap(const ConstantPropertyMap& o): _c(o._c) {}

    const value_type& operator[](const key_type&) const { return _c; }

    ConstantPropertyMap& operator=(const ConstantPropertyMap& other)
    {
        _c = other._c;
        return *this;
    }

private:
    value_type _c;
};

template <class Value, class Key>
void put(const ConstantPropertyMap<Value, Key>&, const Key&, const Value&) {}

// the following is a property map which always returns one
template <class Value, class Key>
class UnityPropertyMap
    : public boost::put_get_helper<Value, UnityPropertyMap<Value, Key>>
{
public:
    typedef Value value_type;
    typedef value_type reference;
    typedef Key key_type;
    typedef boost::readable_property_map_tag category;

    template <class K>
    constexpr value_type operator[](const K&) const { return value_type(1); }
};


template <class Value, class Key>
void put(const UnityPropertyMap<Value, Key>&, const Key&, const Value&) {}

template <class Property>
struct is_constant_property
{
    typedef typename boost::property_traits<Property>::key_type key_type;
    typedef typename boost::property_traits<Property>::value_type value_type;
    typedef typename std::conditional<std::is_same<Property, ConstantPropertyMap<value_type, key_type>>::value,
                                      std::true_type,
                                      typename std::is_same<Property, UnityPropertyMap<value_type, key_type>>::type>::type type;
};

// this wraps an existing property map, but always converts its values to a
// given type
template <class PropertyMap, class Type,
          template <class T1, class T2> class Converter = convert>
class ConvertedPropertyMap
{
public:
    typedef Type value_type;
    typedef typename boost::property_traits<PropertyMap>::value_type orig_type;
    typedef value_type reference;
    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef boost::read_write_property_map_tag category;

    ConvertedPropertyMap(PropertyMap base_map)
        : _base_map(base_map) {}
    ConvertedPropertyMap(){}

    value_type do_get(const key_type& k) const
    {
        return _convert_to(get(_base_map, k));
    }

    void do_put(const key_type& k, const value_type& v)
    {
        put(_base_map, k, _convert_from(v));
    }
private:
    PropertyMap _base_map;
    Converter<value_type, orig_type> _convert_to;
    Converter<orig_type, value_type> _convert_from;
};

template <class PropertyMap, class Type>
Type get(ConvertedPropertyMap<PropertyMap,Type> pmap,
         typename ConvertedPropertyMap<PropertyMap,Type>::key_type k)
{
    return pmap.do_get(k);
}

template <class PropertyMap, class Type>
void put(ConvertedPropertyMap<PropertyMap,Type> pmap,
         typename boost::property_traits<PropertyMap>::key_type k,
         const typename ConvertedPropertyMap<PropertyMap,Type>::value_type& val)
{
    pmap.do_put(k, val);
}

} // graph_tool namespace

#endif
