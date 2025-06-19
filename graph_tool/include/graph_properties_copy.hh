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

#ifndef GRAPH_PROPERTIES_COPY_HH
#define GRAPH_PROPERTIES_COPY_HH

#include "graph_properties.hh"
#include "graph_util.hh"
#include "hash_map_wrap.hh"

namespace graph_tool
{

//
// Property map copying
// ====================

template <class IteratorSel, class PropertyMaps>
struct copy_property
{
    template <class GraphTgt, class GraphSrc, class PropertyTgt>
    void operator()(const GraphTgt& tgt, const GraphSrc& src,
                    PropertyTgt dst_map, boost::any prop_src) const
    {
        try
        {
            auto src_map = boost::any_cast<typename PropertyTgt::checked_t>(prop_src);
            dispatch(tgt, src, dst_map, src_map);
        }
        catch (boost::bad_any_cast&)
        {
            typedef typename boost::property_traits<PropertyTgt>::value_type val_tgt;
            typedef typename IteratorSel::template get_descriptor<GraphSrc>::type src_d;

            DynamicPropertyMapWrap<val_tgt, src_d> src_map(prop_src, PropertyMaps());

            dispatch(tgt, src, dst_map, src_map);
        }
    }

    template <class GraphTgt, class GraphSrc, class PropertyTgt, class PropertySrc>
    void dispatch(const GraphTgt& tgt, const GraphSrc& src,
                  PropertyTgt dst_map, PropertySrc src_map) const
    {
        try
        {
            typename IteratorSel::template apply<GraphSrc>::type vs, vs_end;
            typename IteratorSel::template apply<GraphTgt>::type vt, vt_end;
            std::tie(vt, vt_end) = IteratorSel::range(tgt);
            std::tie(vs, vs_end) = IteratorSel::range(src);
            for (; vs != vs_end; ++vs)
            {
                put(dst_map, *vt++, get(src_map, *vs));
            }
        }
        catch (boost::bad_lexical_cast&)
        {
            throw ValueException("property values are not convertible");
        }
    }

};

template <class IteratorSel, class Graph, class Prop1, class Prop2>
bool compare_props(Graph& g, Prop1 p1, Prop2 p2)
{
    typedef typename boost::property_traits<Prop1>::value_type t1;
    typedef typename boost::property_traits<Prop2>::value_type t2;
    typename IteratorSel::template apply<Graph>::type vi, vi_end;
    std::tie(vi, vi_end) = IteratorSel::range(g);
    try
    {
        for (; vi != vi_end; ++vi)
        {
            auto v = *vi;
            if constexpr (std::is_same_v<t1, t2>)
            {
                if (p1[v] != p2[v])
                    return false;
            }
            else if constexpr (std::is_same_v<t1, boost::python::object>)
            {
                if (p1[v] != boost::python::object(p2[v]))
                    return false;
            }
            else if constexpr (std::is_same_v<t2, boost::python::object>)
            {
                if (p2[v] != boost::python::object(p1[v]))
                    return false;
            }
            else
            {
                if (boost::lexical_cast<t1>(p2[v]) != p1[v])
                    return false;
            }
        }
    }
    catch (boost::bad_lexical_cast&)
    {
        return false;
    }
    return true;
};


struct edge_selector
{
    template <class Graph>
    struct apply
    {
        typedef typename boost::graph_traits<Graph>::edge_iterator type;
    };

    template <class Graph>
    struct get_descriptor
    {
        typedef typename boost::graph_traits<Graph>::edge_descriptor type;
    };

    template <class Graph>
    static std::pair<typename apply<Graph>::type,
                     typename apply<Graph>::type>
    range(Graph& g)
    {
        return edges(g);
    }
};

struct vertex_selector
{
    template <class Graph>
    struct apply
    {
        typedef typename boost::graph_traits<Graph>::vertex_iterator type;
    };

    template <class Graph>
    struct get_descriptor
    {
        typedef typename boost::graph_traits<Graph>::vertex_descriptor type;
    };

    template <class Graph>
    static std::pair<typename apply<Graph>::type,
                     typename apply<Graph>::type>
    range(Graph& g)
    {
        return vertices(g);
    }
};

template <class PropertyMaps>
struct copy_external_edge_property_dispatch
{
    template <class GraphTgt, class GraphSrc, class PropertyTgt>
    void operator()(const GraphTgt& tgt, const GraphSrc& src,
                    PropertyTgt dst_map, boost::any prop_src) const
    {
        try
        {
            auto src_map = boost::any_cast<typename PropertyTgt::checked_t>(prop_src);
            dispatch(tgt, src, dst_map, src_map);
        }
        catch (boost::bad_any_cast&)
        {
            typedef typename boost::property_traits<PropertyTgt>::value_type val_tgt;
            typedef typename boost::graph_traits<GraphSrc>::edge_descriptor edge_t;

            DynamicPropertyMapWrap<val_tgt, edge_t> src_map(prop_src, PropertyMaps());

            dispatch(tgt, src, dst_map, src_map);
        }
    }

    template <class GraphTgt, class GraphSrc, class PropertyTgt,
              class PropertySrc>
    void dispatch(const GraphTgt& tgt, const GraphSrc& src,
                  PropertyTgt dst_map, PropertySrc src_map) const
    {
        typedef typename boost::graph_traits<GraphTgt>::edge_descriptor edge_t;
        gt_hash_map<std::tuple<size_t,size_t>, std::deque<edge_t>> tgt_edges;
        for (auto e : edges_range(tgt))
        {
            auto u = source(e, tgt);
            auto v = target(e, tgt);
            if (!graph_tool::is_directed(tgt) && u > v)
                std::swap(u, v);
            tgt_edges[std::make_tuple(u, v)].push_back(e);
        }

        try
        {
            for (auto e : edges_range(src))
            {
                auto u = source(e, src);
                auto v = target(e, src);
                if (!graph_tool::is_directed(src) && u > v)
                    std::swap(u, v);
                auto& es = tgt_edges[std::make_tuple(u, v)];
                if (es.empty())
                    continue;
                put(dst_map, es.front(), get(src_map, e));
                es.pop_front();
            }
        }
        catch (boost::bad_lexical_cast&)
        {
            throw ValueException("property values are not convertible");
        }
    }

};


} // namespace graph_tool

#endif // GRAPH_PROPERTIES_COPY_HH
