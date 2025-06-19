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

#ifndef GRAPH_PARTITION_MODE_HH
#define GRAPH_PARTITION_MODE_HH

#include "config.h"

#include <vector>

#include <boost/graph/maximum_weighted_matching.hpp>
#include "../blockmodel/graph_blockmodel_util.hh"
#include "../../topology/graph_bipartite_weighted_matching.hh"

#include "../support/contingency.hh"

namespace graph_tool
{

template <class BV>
size_t partition_overlap(BV& x, BV& y)
{
    adj_list<> g;
    typename vprop_map_t<int32_t>::type label(get(vertex_index_t(), g));
    typename vprop_map_t<bool>::type partition(get(vertex_index_t(), g));
    typename eprop_map_t<double>::type mrs(get(edge_index_t(), g));

    get_contingency_graph<false>(g, partition, label, mrs, x, y);

    typedef typename graph_traits<adj_list<>>::vertex_descriptor vertex_t;
    typename vprop_map_t<vertex_t>::type match(get(vertex_index_t(), g));

    auto u = undirected_adaptor<adj_list<>>(g);
    //maximum_weighted_matching(u, mrs, match);
    maximum_bipartite_weighted_matching(u, partition, mrs, match);

    size_t m = 0;
    for (auto v : vertices_range(g))
    {
        if (partition[v])
            continue;
        auto w = match[v];
        if (w == graph_traits<adj_list<>>::null_vertex())
            continue;
        m += mrs[edge(v, w, u).first];
    }
    return m;
}

template <class BX, class BY>
void partition_align_labels(BX&& x, BY&& y)
{
    adj_list<> g;
    typename vprop_map_t<int32_t>::type label(get(vertex_index_t(), g));
    typename vprop_map_t<bool>::type partition(get(vertex_index_t(), g));
    typename eprop_map_t<double>::type mrs(get(edge_index_t(), g));

    get_contingency_graph<false>(g, partition, label, mrs, x, y);

    typedef typename graph_traits<adj_list<>>::vertex_descriptor vertex_t;
    typename vprop_map_t<vertex_t>::type match(get(vertex_index_t(), g));

    auto u = undirected_adaptor<adj_list<>>(g);
    maximum_bipartite_weighted_matching(u, partition, mrs, match);

    idx_set<int> used;
    idx_map<size_t, size_t> vertices, umatch;
    for (auto v : vertices_range(u))
    {
        if (partition[v])
            continue;
        vertices[label[v]] = v;
        auto w = match[v];
        if (w == graph_traits<adj_list<>>::null_vertex())
            continue;
        used.insert(label[w]);
    }

    std::vector<int> unused;
    for (size_t r = 0; r < x.size(); ++r)
    {
        if (used.find(r) != used.end())
            continue;
        unused.push_back(r);
    }
    auto iter = unused.begin();
    for (auto v : vertices_range(u))
    {
        if (partition[v])
            continue;
        auto w = match[v];
        if (w == graph_traits<adj_list<>>::null_vertex())
            umatch[label[v]] = *iter++;
    }

    for (auto& r : x)
    {
        if (r == -1)
            continue;
        auto w = match[vertices[r]];
        if (w == graph_traits<adj_list<>>::null_vertex())
            r = umatch[r];
        else
            r = label[w];
    }
}

template <class BT>
void relabel_nested(BT& b, BT& b_orig, BT& c)
{
    BT temp = c;
    std::fill(c.begin(), c.end(), -1);
    idx_map<int, int> rmap;
    for (size_t j = 0; j < b_orig.size(); ++j)
    {
        if (b_orig[j] == -1)
            continue;
        rmap[b_orig[j]] = b[j];
    }
    for (auto rs : rmap)
    {
        if (size_t(rs.second) >= c.size())
            c.resize(rs.second + 1, -1);
        c[rs.second] = temp[rs.first];
    }
    while (!c.empty() && c.back() == -1)
        c.pop_back();
}

template <class BT>
void relabel_nested(BT& b, BT& b_orig, std::reference_wrapper<BT>& c)
{
    relabel_nested(b, b_orig, c.get());
}

template <class BX, class BY>
void nested_partition_align_labels(BX&& xv, BY&& yv)
{
    int L = std::min(xv.size(), yv.size());
    for (int l = 0; l < L; ++l)
    {
        auto& x = xv[l];
        auto& y = yv[l];

        size_t N = std::max(x.size(), y.size());
        x.resize(N, -1);
        y.resize(N, -1);

        std::vector<int32_t> x_old(x.begin(), x.end());
        partition_align_labels(x, y);

        if (size_t(l + 1) == xv.size())
            continue;

        relabel_nested(x, x_old, xv[l + 1]);
    }
}

template <class BT, class RNG>
void partition_shuffle_labels(BT& x, RNG& rng)
{
     idx_map<int32_t, int32_t> rmap;
     for (auto r : x)
     {
         if (r == -1)
             continue;
         rmap[r] = r;
     }
     std::vector<int32_t> rset;
     for (auto& r : rmap)
         rset.push_back(r.first);
     std::shuffle(rset.begin(), rset.end(), rng);
     size_t pos = 0;
     for (auto& r : rmap)
         r.second = rset[pos++];
     for (auto& r : x)
     {
         if (r == -1)
             continue;
         r = rmap[r];
     }
}

template <class BT, class RNG>
void nested_partition_shuffle_labels(BT& xv, RNG& rng)
{
    for (size_t l = 0; l < xv.size(); ++l)
    {
        auto& x = xv[l];
        std::vector<int32_t> x_old = x;
        partition_shuffle_labels(x, rng);
        if (l < xv.size() - 1)
            relabel_nested(x, x_old, xv[l+1]);
    }
}

template <class BT>
void partition_order_labels(BT& x)
{
    idx_map<int32_t, int32_t> nr;
    for (auto r : x)
    {
        if (r == -1)
            continue;
        nr[r]++;
    }
    std::vector<int32_t> rs;
    for (auto& rnr : nr)
        rs.emplace_back(rnr.first);
    std::sort(rs.begin(), rs.end(),
              [&](auto a, auto b) { return nr[a] > nr[b]; });
    idx_map<int32_t, int32_t> rmap;
    for (size_t i = 0; i < rs.size(); ++i)
        rmap[rs[i]] = i;
    for (auto& r : x)
    {
        if (r == -1)
            continue;
        r = rmap[r];
    }
}

template <class BT>
void nested_partition_order_labels(BT& xv)
{
    for (size_t l = 0; l < xv.size(); ++l)
    {
        auto& x = xv[l];
        std::vector<int32_t> x_old = x;
        partition_order_labels(x);
        if (l < xv.size() - 1)
            relabel_nested(x, x_old, xv[l+1]);
    }

    for (auto& x : xv)
    {
        while (!x.empty() && x.back() == -1)
            x.pop_back();
    }
}

template <class BT, class BS>
double partition_overlap_center(BT& c, BS& bs)
{
    idx_map<int32_t, int32_t> nr;

    typename std::remove_reference_t<decltype(bs)>::index_range idx;

    double p = 0;
    size_t delta = 1;
    while (delta > 0)
    {
        #pragma omp parallel for schedule(runtime)
        for (size_t m = 0; m < bs.shape()[0]; ++m)
            partition_align_labels(bs[boost::indices[m][idx]], c);

        delta = 0;
        p = 0;
        nr.clear();

        #pragma omp parallel for schedule(runtime) firstprivate(nr)     \
            reduction(+:delta,p)
        for (size_t i = 0; i < c.size(); ++i)
        {
            for (size_t m = 0; m < bs.shape()[0]; ++m)
                nr[bs[m][i]]++;
            auto iter = std::max_element(nr.begin(), nr.end(),
                                         [&](auto& a, auto& b)
                                         { return a.second < b.second; });
            if (c[i] != iter->first)
                delta++;
            c[i] = iter->first;
            p += iter->second / double(bs.shape()[0]);
            nr.clear();
        }
    }
    p /= c.size();
    return 1 - p;
}

template <class BT, class BS>
double nested_partition_overlap_center(BT& c, BS& bs)
{
    idx_map<int32_t, int32_t> nr;

    double p = 0;
    size_t N = 0;
    size_t delta = 1;
    while (delta > 0)
    {
        #pragma omp parallel for schedule(runtime)
        for (size_t m = 0; m < bs.size(); ++m)
            nested_partition_align_labels(bs[m], c);

        delta = 0;
        p = 0;
        N = 0;
        for (size_t l = 0; l < c.size(); ++l)
        {
            auto& cl = c[l];
            nr.clear();

            size_t Nl = 0;
            double pl = 0;
            #pragma omp parallel for schedule(runtime) firstprivate(nr)        \
                reduction(+:delta, pl, Nl) if (cl.size() > get_openmp_min_thresh())
            for (size_t i = 0; i < cl.size(); ++i)
            {
                size_t M = 0;
                for (size_t m = 0; m < bs.size(); ++m)
                {
                    auto& bml = bs[m][l];
                    if (i >= bml.size())
                        continue;
                    auto r = bml[i];
                    if (r == -1)
                        continue;
                    nr[r]++;
                    M++;
                }

                if (!nr.empty())
                {
                    auto iter = std::max_element(nr.begin(), nr.end(),
                                                 [&](auto& a, auto& b)
                                                 { return a.second < b.second; });
                    if (cl[i] != iter->first)
                        delta++;
                    cl[i] = iter->first;
                    pl += iter->second / double(M);
                    Nl++;
                }
                else
                {
                    delta += (cl[i] != -1);
                    cl[i] = -1;
                }
                nr.clear();
            }
            p += (Nl - 1) * (pl / Nl);
            N += Nl;
        }
    }

    for (size_t l = 0; l < c.size(); ++l)
    {
        auto& cl = c[l];
        while (!cl.empty() && cl.back() == -1)
            cl.pop_back();
    }

    return 1 - p / (N - c.size());
}

class PartitionModeState
{
public:
    PartitionModeState() {}

    typedef std::vector<int32_t> b_t;
    typedef std::vector<std::reference_wrapper<b_t>> bv_t;

    template <class BT>
    void check_size(BT& b)
    {
        size_t n = std::max(_nr.size(), b.size());
        b.resize(n, -1);
        _nr.resize(n);
        while (!b.empty() && b.back() == -1 && _nr.back().empty())
        {
            b.pop_back();
            _nr.pop_back();
        }
        auto iter = std::max_element(b.begin(), b.end());
        if (*iter >= int(_count.size()))
            _count.resize(*iter + 1);
    }

    size_t add_partition(bv_t& bv, bool relabel)
    {
        if (_coupled_state == nullptr && bv.size() > 1)
        {
            auto* s = this;
            for (size_t i = 0; i < bv.size() - 1; ++i)
            {
                s->_coupled_state = std::make_shared<PartitionModeState>();
                s = s->_coupled_state.get();
            }
        }
        clean_labels(bv, 0);
        return add_partition(bv, 0, relabel);
    }

    void clean_labels(bv_t& bv, size_t pos)
    {
        if (bv.size() - pos == 1)
            return;
        idx_set<int> rs;
        for (auto r : bv[pos].get())
        {
            if (r == -1)
                continue;
            rs.insert(r);
        }
        b_t& c = bv[pos + 1];
        for (size_t r = 0; r < c.size(); ++r)
        {
            if (rs.find(r) == rs.end())
                c[r] = -1;
        }
        clean_labels(bv, pos + 1);
    }

    size_t add_partition(bv_t& bv, size_t pos, bool relabel)
    {
        auto& b = bv[pos].get();

        if (relabel && pos == 0)
            relabel_partition(bv, 0);

        check_size(b);

        for (size_t i = 0; i < b.size(); ++i)
        {
            auto r = b[i];
            if (r == -1)
                continue;
            _nr[i][r]++;
            _count[r]++;
            if (_count[r] == 1)
            {
                _B++;
                _free_idxs.erase(r);
            }
            if (r > _rmax)
                _rmax = r;
        }
        size_t j;
        if (_free_pos.empty())
        {
            j = _max_pos++;
        }
        else
        {
            j = _free_pos.back();
            _free_pos.pop_back();
        }
        _bs.insert(std::make_pair(j, std::ref(b)));

        if (_coupled_state != nullptr)
            _coupled_pos[j] = _coupled_state->add_partition(bv, pos + 1, false);

        return j;
    }

    void remove_partition(size_t j)
    {
        assert(_bs.find(j) != _bs.end());
        b_t& b = _bs.find(j)->second.get();
        for (size_t i = 0; i < b.size(); ++i)
        {
            auto r = b[i];
            if (r == -1)
                continue;
            auto iter = _nr[i].find(r);
            iter->second--;
            if (iter->second == 0)
                _nr[i].erase(iter);
            _count[r]--;
            if (_count[r] == 0)
            {
                _B--;
                _free_idxs.insert(r);
            }
        }
        _bs.erase(j);
        _free_pos.push_back(j);
        if (_coupled_state != nullptr)
            _coupled_state->remove_partition(_coupled_pos[j]);
    }

    template <class RNG>
    double replace_partitions(RNG& rng)
    {
        std::vector<size_t> pos;
        for (auto ib : _bs)
            pos.push_back(ib.first);

        std::shuffle(pos.begin(), pos.end(), rng);

        double ddS = 0;
        for (auto j : pos)
        {
            auto bv = get_nested_partition(j);
            double dS = virtual_remove_partition(bv);
            remove_partition(j);
            dS += virtual_add_partition(bv);
            add_partition(bv, dS < 0);
            if (dS < 0)
                ddS += dS;
        }
        return ddS;
    }

    bool has_partition(size_t j, b_t& b)
    {
        if (_bs.find(j) == _bs.end())
            return false;
        return (b == get_partition(j));
    }

    void rebuild_nr()
    {
        _B = 0;
        _rmax = 0;
        for (auto& x : _nr)
            x.clear();
        std::fill(_count.begin(), _count.end(), 0);
        for (auto jb : _bs)
        {
            auto& b = get_partition(jb.first);
            check_size(b);

            for (size_t i = 0; i < b.size(); ++i)
            {
                auto r = b[i];
                if (r == -1)
                    continue;
                _nr[i][r]++;
                _count[r]++;
                if (_count[r] == 1)
                {
                    _B++;
                    _free_idxs.erase(r);
                }
                if (r > _rmax)
                    _rmax = r;
            }
        }
        for (int r = 0; r < _rmax; ++r)
        {
            if (_count[r] == 0)
                _free_idxs.insert(r);
        }
        if (_coupled_state != nullptr)
            _coupled_state->rebuild_nr();
    }

    void relabel()
    {
        std::vector<int32_t> labels(_count.size()), map(_count.size());
        std::iota(labels.begin(), labels.end(), 0);
        std::sort(labels.begin(), labels.end(),
                  [&](auto r, auto s) { return _count[r] > _count[s]; });

        for (size_t r = 0; r < _count.size(); ++r)
            map[labels[r]] = r;

        for (auto& jb : _bs)
        {
            auto& b = get_partition(jb.first);
            check_size(b);
            b_t b_orig = b;
            for (auto& r : b)
            {
                if (r == -1)
                    continue;
                r = map[r];
            }

            if (_coupled_state != nullptr)
            {
                auto& c = _coupled_state->get_partition(_coupled_pos[jb.first]);
                relabel_nested(b, b_orig, c);
            }
        }
        rebuild_nr();
        if (_coupled_state != nullptr)
            _coupled_state->relabel();
    }

    template <class BV>
    void relabel_partition(BV& bv, size_t pos)
    {
        b_t& b = bv[pos];

        check_size(b);

        b_t b_orig = b;

        idx_map<int32_t, size_t> rpos;
        for (size_t i = 0; i < b.size(); ++i)
        {
            auto r = b[i];
            if (r == -1)
                continue;
            auto iter = rpos.find(r);
            if (iter == rpos.end())
                b[i] = rpos[r] = rpos.size();
            else
                b[i] = iter->second;
        }

        if (_bs.empty())
        {
            if (_coupled_state != nullptr)
            {
                relabel_nested(b, b_orig, bv[pos + 1]);
                _coupled_state->relabel_partition(bv, pos + 1);
            }
            return;
        }

        adj_list<> g;
        typename vprop_map_t<int32_t>::type label(get(vertex_index_t(), g));
        typename vprop_map_t<bool>::type partition(get(vertex_index_t(), g));
        typename eprop_map_t<double>::type mrs(get(edge_index_t(), g));

        get_contingency_graph<true>(g, partition, label, mrs, b, _nr);

        typedef typename graph_traits<adj_list<>>::vertex_descriptor vertex_t;
        typename vprop_map_t<vertex_t>::type match(get(vertex_index_t(), g));

        auto u = undirected_adaptor<adj_list<>>(g);
        maximum_bipartite_weighted_matching(u, partition, mrs, match);

        idx_map<int32_t, size_t> b_vertices;
        for (auto v : vertices_range(g))
        {
            if (v >= rpos.size())
                break;
            b_vertices[label[v]] = v;
        }

        auto ipos = _free_idxs.begin();
        rpos.clear();
        for (size_t i = 0; i < b.size(); ++i)
        {
            auto r = b[i];
            if (r == -1)
                continue;
            auto v = match[b_vertices[r]];
            if (v != graph_traits<adj_list<>>::null_vertex())
            {
                b[i] = label[v];
            }
            else
            {
                auto iter = rpos.find(r);
                if (iter == rpos.end())
                {
                    if (ipos == _free_idxs.end())
                        ipos = _free_idxs.insert(++_rmax).first;
                    rpos[r] = b[i] = *(ipos++);
                }
                else
                {
                    b[i] = iter->second;
                }
            }
        }

        if (_coupled_state != nullptr)
        {
            relabel_nested(b, b_orig, bv[pos + 1]);
            _coupled_state->relabel_partition(bv, pos + 1);
        }
    }

    void align_mode(PartitionModeState& mode)
    {
        adj_list<> g;
        typename vprop_map_t<int32_t>::type label(get(vertex_index_t(), g));
        typename vprop_map_t<bool>::type partition(get(vertex_index_t(), g));
        typename eprop_map_t<double>::type mrs(get(edge_index_t(), g));

        get_contingency_graph<true>(g, partition, label, mrs, _nr, mode._nr);

        typedef typename graph_traits<adj_list<>>::vertex_descriptor vertex_t;
        typename vprop_map_t<vertex_t>::type match(get(vertex_index_t(), g));

        auto u = undirected_adaptor<adj_list<>>(g);
        maximum_bipartite_weighted_matching(u, partition, mrs, match);

        idx_map<int32_t, size_t> vertices;
        for (auto v : vertices_range(g))
        {
            if (partition[v] == 1)
                break;
            vertices[label[v]] = v;
        }

        auto ipos = _free_idxs.begin();
        idx_map<int32_t, size_t> rpos;
        for (size_t r = 0; r < _count.size(); ++r)
        {
            if (_count[r] == 0)
                continue;
            auto v = match[vertices[r]];
            if (v != graph_traits<adj_list<>>::null_vertex())
            {
                rpos[r] = label[v];
            }
            else
            {
                auto iter = rpos.find(r);
                if (iter == rpos.end())
                {
                    if (ipos == _free_idxs.end())
                        ipos = _free_idxs.insert(++_rmax).first;
                    rpos[r] = *(ipos++);
                }
                else
                {
                    rpos[r] = iter->second;
                }
            }
        }

        std::vector<size_t> js;
        for (auto& bi : _bs)
            js.push_back(bi.first);
        for (auto j : js)
        {
            auto bv = get_nested_partition(j);
            remove_partition(j);

            auto& b = bv[0].get();
            b_t b_orig = b;
            for (size_t i = 0; i < b.size(); ++i)
            {
                auto r = b[i];
                if (r == -1)
                    continue;
                b[i] = rpos[r];
            }

            if (_coupled_state != nullptr)
                relabel_nested(b, b_orig, bv[1]);

            add_partition(bv, false);
        }

        if (_coupled_state != nullptr && mode._coupled_state != nullptr)
            _coupled_state->align_mode(*mode._coupled_state);
    }

    b_t& get_partition(size_t i)
    {
        return _bs.find(i)->second.get();
    }

    auto& get_partitions()
    {
        return _bs;
    }

    bv_t get_nested_partition(size_t i)
    {
        bv_t bv;
        bv.push_back(std::ref(get_partition(i)));
        if (_coupled_state != nullptr)
        {
            auto bv_l =_coupled_state->get_nested_partition(_coupled_pos[i]);
            bv.insert(bv.end(), bv_l.begin(), bv_l.end());
        }
        return bv;
    }

    template <class Graph, class VM>
    void get_marginal(Graph& g, VM bm)
    {
        for (auto v : vertices_range(g))
        {
            if (v >= _nr.size())
                break;
            auto& h = bm[v];
            for (auto rn : _nr[v])
            {
                if (rn.first >= h.size())
                    h.resize(rn.first + 1);
                h[rn.first] = rn.second;
            }
        }
    }

    template <class Graph, class VM>
    void get_map(Graph& g, VM b)
    {
        for (auto v : vertices_range(g))
        {
            if (v >= _nr.size())
                break;
            size_t cmax = 0;
            int r = -1;
            for (auto rn : _nr[v])
            {
                if (rn.second > cmax)
                {
                    cmax = rn.second;
                    r = rn.first;
                }
            }
            b[v] = r;
        }
    }

    std::vector<int> get_map_b()
    {
        std::vector<int> b;
        for (auto& nrv : _nr)
        {
            size_t cmax = 0;
            int r = 0;
            for (auto& rn : nrv)
            {
                if (rn.second > cmax)
                {
                    cmax = rn.second;
                    r = rn.first;
                }
            }
            b.push_back(r);
        }
        return b;
    }

    double entropy()
    {
        double L = 0;
        size_t N = 0;
        for (auto& x : _nr)
        {
            size_t n = 0;
            for (auto& rn : x)
            {
                L += lgamma(rn.second + 1);
                n += rn.second;
            }
            if (n > 0)
            {
                N++;
                L += lgamma(_B) - lgamma(n + _B);
            }
        }
        L -= safelog(N);
        if (_coupled_state != nullptr)
            L -= _coupled_state->entropy();
        return -L;
    }

    double posterior_entropy(bool MLE)
    {
        if (_bs.size() == 0)
            return 0;
        double S = 0;
        for (auto& x : _nr)
        {
            if (_nr.empty())
                continue;
            double Si = 0;
            size_t n = 0;
            for (auto& rn : x)
            {
                Si -= xlogx(MLE ? rn.second : rn.second + 1);
                n += rn.second;
            }
            if (n == 0)
                continue;
            auto B = MLE ? 0 : _B;
            Si /= (n + B);
            Si += log(n + B);
            S += Si;
        }
        if (_coupled_state != nullptr)
            S += _coupled_state->posterior_entropy(MLE);
        return S;
    }

    double posterior_cdev(bool MLE)
    {
        if (_bs.size() == 0)
            return 0;
        double ce = 0;
        size_t N = 0;
        for (auto& x : _nr)
        {
            if (x.empty())
                continue;
            size_t n = 0;
            size_t nmax = 0;
            for (auto& rn : x)
            {
                nmax = std::max(rn.second, nmax);
                n += rn.second;
            }
            if (n == 0)
                continue;
            if (MLE)
                ce += nmax / double(n);
            else
                ce += (nmax + 1) / double(n + _B);
            N++;
        }
        return 1. - ce / N;
    }

    double posterior_lprob(bv_t& b, bool MLE)
    {
        return posterior_lprob(b, 0, MLE);
    }

    double posterior_lprob(bv_t& bv, size_t pos, bool MLE)
    {
        if (_bs.size() == 0)
            return 0;

        b_t& b = bv[pos];
        double L = 0;
        for (size_t i = 0; i < _nr.size(); ++i)
        {
            auto& x = _nr[i];
            if (x.empty())
            {
                if (b[i] == -1)
                    continue;
                L = -numeric_limits<double>::infinity();
                break;
            }
            size_t n = 0;
            for (auto& rn : x)
                n += rn.second;
            size_t nr = 0;
            auto iter = x.find(b[i]);
            if (iter != x.end())
                nr = iter->second;
            if (MLE)
            {
                if (nr == 0)
                {
                    L = -numeric_limits<double>::infinity();
                    break;
                }
                L += log(nr) - log(n);
            }
            else
            {
                if (nr == 0 && (size_t(b[i]) >= _count.size() ||
                                _count[b[i]] == 0))
                {
                    L = -numeric_limits<double>::infinity();
                    break;
                }
                L += log1p(nr) - log(n + _B);
            }
        }
        if (std::isinf(L))
            return L;
        if (_coupled_state != nullptr)
            L += _coupled_state->posterior_lprob(bv, pos+1, MLE);
        return L;
    }

    template <bool add, class BV>
    double virtual_change_partition(const BV& x, size_t pos, bool relabel=false)
    {
        std::vector<b_t> bv;
        for (auto& b : x)
            bv.emplace_back(b);

        if (add && bv.size() - pos > 1 && _coupled_state == nullptr)
            _coupled_state = std::make_shared<PartitionModeState>();

        if (add && pos == 0 && relabel)
            relabel_partition(bv, 0);

        auto& b = bv[pos];

        check_size(b);

        double dL = 0;
        std::vector<int> delta_nr(_count.size());
        for (size_t i = 0; i < b.size(); ++i)
        {
            auto r = b[i];
            if (r == -1)
                continue;
            auto& x = _nr[i];

            int nir = 0;
            auto iter = x.find(r);
            if (iter != x.end())
                nir = iter->second;
            assert(nir >= 0);

            dL -= lgamma_fast(nir + 1);
            if (size_t(r) >= delta_nr.size())
                delta_nr.resize(r + 1);
            if constexpr (add)
            {
                dL += lgamma_fast(nir + 2);
                delta_nr[r]++;
            }
            else
            {
                assert(nir > 0);
                dL += lgamma_fast(nir);
                delta_nr[r]--;
            }
        }

        int dB = 0;
        for (size_t r = 0; r < delta_nr.size(); ++r)
        {
            if constexpr (add)
            {
                if (_count[r] == 0 && delta_nr[r] > 0)
                    dB++;
            }
            else
            {
                if (_count[r] > 0 && _count[r] + delta_nr[r] == 0)
                    dB--;
            }
        }

        int dN = 0;
        size_t N = 0;
        for (size_t i = 0; i < _nr.size(); ++i)
        {
            int n = 0;
            for (auto rn : _nr[i])
                n += rn.second;

            assert(n >= 0);
            if (n > 0)
            {
                dL -= lgamma_fast(_B) - lgamma_fast(n + _B);
                N++;
            }
            if (b[i] != -1)
            {
                if constexpr (add)
                {
                    if (n == 0)
                        dN++;
                    n++;
                }
                else
                {
                    n--;
                    if (n == 0)
                        dN--;
                }
            }
            if (n > 0)
                dL += lgamma_fast(_B + dB) - lgamma_fast(n + _B + dB);
            assert(n >= 0);
        }

        dL -= safelog_fast(N + dN);
        dL += safelog_fast(N);

        if (_coupled_state != nullptr)
            dL -= _coupled_state->virtual_change_partition<add>(bv, pos + 1);

        return -dL;
    }

    double virtual_remove_partition(bv_t& x)
    {
        return virtual_change_partition<false>(x, 0, false);
    }

    double virtual_add_partition(bv_t& x, bool relabel = true)
    {
        return virtual_change_partition<true>(x, 0, relabel);
    }

    size_t get_N()
    {
        size_t N = 0;
        for (auto& x : _nr)
        {
            if (!x.empty())
                N++;
        }
        return N;
    }

    size_t get_B()
    {
        return _B;
    }

    size_t get_M()
    {
        return _bs.size();
    }

    std::shared_ptr<PartitionModeState> get_coupled_state()
    {
        return _coupled_state;
    }

    template <class RNG>
    b_t sample_partition(bool MLE, RNG& rng)
    {
        b_t b;
        std::vector<int> rs_base;
        std::vector<double> probs_base;
        if (not MLE)
        {
            for (size_t r = 0; r < _count.size(); ++r)
            {
                if (_count[r] == 0)
                    continue;
                rs_base.push_back(r);
                probs_base.push_back(1);
            }
        }

        for (auto& nrv : _nr)
        {
            if (nrv.empty())
            {
                b.push_back(-1);
                continue;
            }
            auto rs = rs_base;
            auto probs = probs_base;
            for (auto rn : nrv)
            {
                rs.push_back(rn.first);
                probs.push_back(rn.second);
            }
            Sampler<int> sample(rs, probs);
            b.push_back(sample.sample(rng));
        }
        return b;
    }

    template <class RNG>
    std::vector<b_t> sample_nested_partition(bool MLE, bool fix_empty, RNG& rng)
    {
        std::vector<b_t> bv;
        bv.emplace_back(sample_partition(MLE, rng));
        if (fix_empty)
        {
            for (auto& r : bv.back())
            {
                if (r == -1)
                    r = 0;
            }
        }
        if (_coupled_state != nullptr)
        {
            auto nbv = _coupled_state->sample_nested_partition(MLE, fix_empty, rng);
            bv.insert(bv.end(), nbv.begin(), nbv.end());
        }
        return bv;
    }

//private:

    idx_map<size_t, std::reference_wrapper<b_t>> _bs;
    std::vector<gt_hash_map<size_t, size_t>> _nr;
    std::vector<size_t> _count;
    size_t _B = 0;
    int _rmax = 0;
    std::set<size_t> _free_idxs;
    std::vector<size_t> _free_pos;
    size_t _max_pos = 0;

    typedef char _entropy_args_t;
    std::shared_ptr<PartitionModeState> _coupled_state;
    idx_map<size_t, size_t> _coupled_pos;

};

} // graph_tool namespace

#endif //GRAPH_PARTITION_MODE_HH
