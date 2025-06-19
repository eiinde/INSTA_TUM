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

#ifndef GRAPH_BLOCKMODEL_EMAT_HH
#define GRAPH_BLOCKMODEL_EMAT_HH

#include <boost/multi_array.hpp>
#include "hash_map_wrap.hh"

namespace graph_tool
{

// this structure speeds up the access to the edges between given blocks, by
// using a simple adjacency matrix

template <class BGraph>
class EMat
{
public:
    template <class Graph>
    EMat(Graph&, BGraph& bg)
    {
        sync(bg);
    }

    void sync(BGraph& bg)
    {
        size_t B = num_vertices(bg);
        _mat.resize(boost::extents[B][B]);
        std::fill(_mat.data(), _mat.data() + _mat.num_elements(), _null_edge);

        for (auto e : edges_range(bg))
        {
            assert(get_me(source(e, bg),target(e, bg)) == _null_edge);
            _mat[source(e, bg)][target(e, bg)] = e;
            if (!is_directed_::apply<BGraph>::type::value)
                _mat[target(e, bg)][source(e, bg)] = e;
        }
    }

    void add_block(BGraph& bg)
    {
        sync(bg);
    }

    typedef typename graph_traits<BGraph>::vertex_descriptor vertex_t;
    typedef typename graph_traits<BGraph>::edge_descriptor edge_t;

    const edge_t& get_me(vertex_t r, vertex_t s) const
    {
        return _mat[r][s];
    }

    void put_me(vertex_t r, vertex_t s, const edge_t& e)
    {
        assert(e != _null_edge);
        _mat[r][s] = e;
        if (!is_directed_::apply<BGraph>::type::value && r != s)
            _mat[s][r] = e;
    }

    void remove_me(const edge_t& me, BGraph& bg)
    {
        auto r = source(me, bg);
        auto s = target(me, bg);
        _mat[r][s] = _null_edge;
        if (!is_directed_::apply<BGraph>::type::value)
            _mat[s][r] = _null_edge;
        //remove_edge(me, bg);
    }

    const edge_t& get_null_edge() const { return _null_edge; }

private:
    multi_array<edge_t, 2> _mat;
    static const edge_t _null_edge;
};

template <class BGraph>
const typename EMat<BGraph>::edge_t EMat<BGraph>::_null_edge;


// this structure speeds up the access to the edges between given blocks, since
// we're using an adjacency list to store the block structure (this is like
// EMat above, but takes less space and is slower)

template <class BGraph>
class EHash
{
public:

    template <class Graph>
    EHash(Graph& g, BGraph& bg)
        : _L(num_vertices(g) * 10)
    {
        sync(bg);
    }

    void sync(BGraph& bg)
    {
        if (num_vertices(bg) > _L)
            _L = num_vertices(bg) * 10;

        _h.clear();
        _h.resize(0);
        _h.max_load_factor(.5);
        _h.min_load_factor(.25);

        for (auto e : edges_range(bg))
        {
            assert(get_me(source(e, bg), target(e, bg)) == _null_edge);
            put_me(source(e, bg), target(e, bg), e);
        }
    }

    void add_block(BGraph& bg)
    {
        if (num_vertices(bg) > _L)
            sync(bg);
    }

    typedef typename graph_traits<BGraph>::vertex_descriptor vertex_t;
    typedef typename graph_traits<BGraph>::edge_descriptor edge_t;

    [[gnu::flatten]] __attribute__((hot))
    const edge_t& get_me(vertex_t r, vertex_t s) const
    {
        if (!is_directed_::apply<BGraph>::type::value && r > s)
            std::swap(r, s);
        auto iter = _h.find(r + s * _L);
        if (iter == _h.end())
            return _null_edge;
        return iter->second;
    }

    void put_me(vertex_t r, vertex_t s, const edge_t& e)
    {
        if (!is_directed_::apply<BGraph>::type::value && r > s)
            std::swap(r, s);
        assert(e != _null_edge);
        _h[r + s * _L] = e;
    }

    void remove_me(const edge_t& me, BGraph& bg)
    {
        auto r = source(me, bg);
        auto s = target(me, bg);
        if (!is_directed_::apply<BGraph>::type::value && r > s)
            std::swap(r, s);
        _h.erase(r + s * _L);
    }

    const edge_t& get_null_edge() const { return _null_edge; }

private:
    typedef gt_hash_map<size_t, edge_t> ehash_t;
    ehash_t _h;
    size_t _L;
    static const edge_t _null_edge;
};

template <class BGraph>
const typename EHash<BGraph>::edge_t EHash<BGraph>::_null_edge;

template <class Vertex, class Eprop, class Emat, class BEdge>
inline auto get_beprop(Vertex r, Vertex s, const Eprop& eprop, const Emat& emat,
                       BEdge& me)
{
    typedef typename property_traits<Eprop>::value_type val_t;
    me = emat.get_me(r, s);
    if (me != emat.get_null_edge())
        return eprop[me];
    return val_t();
}

template <class Vertex, class Eprop, class Emat>
inline auto get_beprop(Vertex r, Vertex s, const Eprop& eprop, const Emat& emat)
{
    typedef typename property_traits<Eprop>::key_type bedge_t;
    bedge_t me;
    return get_beprop(r, s, eprop, emat, me);
}

} // namespace graph_tool

#endif // GRAPH_BLOCKMODEL_EMAT_HH
