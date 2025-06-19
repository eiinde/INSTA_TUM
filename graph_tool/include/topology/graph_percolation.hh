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

#ifndef GRAPH_PERCOLATION_HH
#define GRAPH_PERCOLATION_HH

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class Graph, class TreeMap>
auto find_root(size_t vi, TreeMap tree, Graph& g,
               vector<size_t>& temp)
{
    auto parent = vertex(vi, g);
    temp.clear();
    while(size_t(tree[vertex(parent, g)]) != parent)
    {
        temp.push_back(parent);
        parent = tree[vertex(parent, g)];
    }
    // path compression
    for (auto v: temp)
        tree[vertex(v, g)] = parent;
    return vertex(parent, g);
}


template <class Graph, class TreeMap, class SizeMap>
auto join_cluster(const pair<size_t, size_t>& e, TreeMap tree, SizeMap size,
                  Graph& g, vector<size_t>& shist, vector<size_t>& temp)
{
    auto rs = find_root(e.first, tree, g, temp);
    auto rt = find_root(e.second, tree, g, temp);
    if (rt != rs)
    {
        auto srs = size[rs];
        auto srt = size[rt];
        if (srs < srt)
            swap(rs, rt);
        tree[rt] = rs;
        size[rs] += size[rt];
        shist[srs]--;
        shist[srt]--;
        shist[size[rs]]++;
        return size[rs];
    }
    return std::max(size[rs], size[rt]);
}

template <class Graph, class TreeMap, class SizeMap, class MaxSize,
          class Edges>
void edge_percolate(Graph& g, TreeMap tree, SizeMap size, MaxSize& max_size,
                    Edges& edges, bool second)
{
    vector<size_t> temp;
    vector<size_t> shist(num_vertices(g) + 1);
    shist[1] = num_vertices(g);
    size_t ms = 0;
    for (size_t i = 0; i < edges.size(); ++i)
    {
        size_t s = join_cluster({edges[i][0], edges[i][1]},
                                tree, size, g, shist, temp);
        ms = std::max(ms, s);
        if (!second)
        {
            max_size[i] = ms;
        }
        else
        {
            for (size_t s = 1; s < ms; ++s)
            {
                if (shist[s] > 0)
                    max_size[i] = s;
            }
        }
    }

    boost::multi_array_ref<typename Edges::element, 1>
        vertices(edges.data(),
                 boost::extents[edges.num_elements()]);

    for (auto& v : vertices)
    {
        auto root = find_root(v, tree, g, temp);
        size[v] = size[root];
    }
};

template <class Graph,  class TreeMap, class SizeMap, class VisitedMap,
          class MaxSize,  class Vertices>
void vertex_percolate(Graph& g, TreeMap tree, SizeMap size, VisitedMap visited,
                      MaxSize& max_size, Vertices& vertices, bool second)
{
    vector<size_t> temp;
    vector<size_t> shist(num_vertices(g) + 1);
    shist[1] = num_vertices(g);
    size_t ms = 0;
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        auto v = vertex(vertices[i], g);
        if (v == graph_traits<Graph>::null_vertex())
        {
            max_size[i] = ms;
            continue;
        }

        for (auto a : adjacent_vertices_range(v, g))
        {
            if (!visited[a])
                continue;
            size_t s = join_cluster({v, a}, tree, size, g,
                                    shist, temp);
            ms = std::max(ms, s);
        }
        if (!second)
        {
            max_size[i] = std::max(ms, size_t(1));
        }
        else
        {
            for (size_t s = 1; s < ms; ++s)
            {
                if (shist[s] > 0)
                    max_size[i] = s;
            }
        }
        visited[v] = true;
    }

    //flatten tree
    for (auto vi : vertices)
    {
        auto v = vertex(vi, g);
        if (v == graph_traits<Graph>::null_vertex())
            continue;
        auto root = find_root(vi, tree, g, temp);
        size[v] = size[root];
    }
}

} // graph_tool namespace

#endif // GRAPH_PERCOLATION_HH
