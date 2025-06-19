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

#ifndef GRAPH_KNN_HH
#define GRAPH_KNN_HH

#include <tuple>
#include <random>
#include <list>
#include <shared_mutex>
#include <boost/functional/hash.hpp>

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "parallel_rng.hh"
#include "idx_map.hh"

#include "random.hh"

#include "hash_map_wrap.hh"

#include "graph_contract_edges.hh"
#include "shared_heap.hh"

#include "../clustering/graph_clustering.hh"


namespace graph_tool
{
using namespace std;
using namespace boost;

template <bool directed, bool parallel, bool keep_iter, class Dist>
class DistCache
{
public:
    DistCache(Dist&& d, size_t N)
        : _cache(N), _d(d), _mutex(parallel ? N : 0)

    {}

    double operator()(size_t u, size_t v, [[maybe_unused]] size_t iter = 0)
    {
        if (!directed && (u > v))
            std::swap(u, v);

        if constexpr (keep_iter)
        {
            if (iter < _last_iter)
                _offset += _last_iter + 2;
            _last_iter = iter;
            iter += _offset;
        }

        auto& cache = _cache[v];

        double d;
        if constexpr (parallel)
        {
            auto& mutex = _mutex[v];
            {
                std::shared_lock lock(mutex);
                auto iter = cache.find(u);
                if (iter != cache.end())
                {
                    if constexpr (keep_iter)
                        return get<0>(iter->second);
                    else
                        return iter->second;
                }
            }

            d = _d(u, v);

            {
                std::unique_lock lock(mutex);
                if constexpr (keep_iter)
                    cache[u] = {d, iter};
                else
                    cache[u] = d;
                _miss_count++;
            }
        }
        else
        {
            auto iter = cache.find(u);
            if (iter != cache.end())
            {
                if constexpr (keep_iter)
                    return get<0>(iter->second);
                else
                    return iter->second;
            }
            d = _d(u, v);
            if constexpr (keep_iter)
                cache[u] = {d, iter};
            else
                cache[u] = d;
            _miss_count++;
        }

        return d;
    }

    size_t _miss_count = 0;
    std::conditional_t
        <keep_iter,
         std::vector<gt_hash_map<size_t, std::tuple<double, size_t>>>,
         std::vector<gt_hash_map<size_t, double>>> _cache;

private:
    Dist& _d;
    std::vector<std::shared_mutex> _mutex;
    size_t _last_iter = 0;
    size_t _offset = 0;
};

template <bool directed, bool parallel, bool keep_iter, class Dist>
DistCache<directed, parallel, keep_iter, Dist> make_dist_cache(Dist&& d, size_t N)
{
    return DistCache<directed, parallel, keep_iter, Dist>(d, N);
}

template <bool parallel, class Graph, class Dist, class Weight, class Hint, class RNG>
size_t gen_knn(Graph& g, Dist&& d, size_t k, double r, size_t max_rk,
               double epsilon, bool c_stop, size_t max_iter, Weight eweight,
               Hint& hint, bool verbose, RNG& rng_)
{
    parallel_rng<rng_t> prng(rng_);

    auto cmp =
        [] (auto& x, auto& y)
        {
            return get<1>(x) < get<1>(y);
        };

    std::vector<std::vector<std::tuple<size_t, double>>>
        B(num_vertices(g));

    std::vector<size_t> vs, vs_;
    for (auto v : vertices_range(g))
        vs.push_back(v);
    vs_ = vs;

    idx_set<size_t> seen;

    size_t n_tot = 0;

    if (verbose)
        cout << "random init" << endl;

    #pragma omp parallel if (parallel) firstprivate(vs_) firstprivate(seen) reduction(+:n_tot)
    parallel_loop_no_spawn
        (vs,
         [&](auto, auto v)
         {
             auto& rng = prng.get(rng_);
             seen.clear();
             auto& Bv = B[v];
             for (auto u : random_permutation_range(vs_, rng))
             {
                 if (u == v)
                     continue;
                 double l = d(u, v, 0);
                 ++n_tot;
                 Bv.emplace_back(u, l);
                 std::push_heap(Bv.begin(), Bv.end(), cmp);
                 seen.insert(u);
                 if (Bv.size() == k)
                     break;
             }

             auto update =
                 [&](auto u, auto w)
                 {
                     if (w == u || w == v || seen.find(w) != seen.end())
                         return;
                     double l = d(w, v, 0);
                     ++n_tot;
                     if (l < get<1>(Bv.front()))
                     {
                         std::pop_heap(Bv.begin(), Bv.end(), cmp);
                         Bv.back() = {w, l};
                         std::push_heap(Bv.begin(), Bv.end(), cmp);
                     }
                     seen.insert(w);
                 };

             for (auto u : all_neighbors_range(v, g))
                 update(v, u);

             for (auto u : all_neighbors_range(v, hint))
             {
                 update(v, u);
                 for (auto w : all_neighbors_range(u, hint))
                     update(u, w);
             }
         });

    auto build_vertex = [&](auto v)
        {
            for (auto& [u, l] : B[v])
            {
                auto e = add_edge(u, v, g).first;
                eweight[e] = l;
            }
        };

    std::vector<std::vector<size_t>> out_neighbors(num_vertices(g));

    std::bernoulli_distribution rsample(r);

    undirected_adaptor g_u(g);
    UnityPropertyMap<size_t, typename graph_traits<Graph>::edge_descriptor> dummy;

    size_t iter = 1;
    double delta = epsilon + 1;
    double clust = 0;
    while (delta > epsilon)
    {
        if (verbose)
            cout << "build graph" << endl;

        for (size_t v : vs)
            clear_vertex(v, g);
        for (auto v : vs)
            build_vertex(v);

        if (c_stop)
        {
            auto nclust = get<0>(get_global_clustering(g_u, dummy));
            if (iter > 1 && nclust <= clust)
            {
                if (verbose)
                    cout << " " << nclust << endl;
                break;
            }
            clust = nclust;
        }

        #pragma omp parallel if (parallel)
        parallel_loop_no_spawn
            (vs,
             [&](auto, auto v)
             {
                 auto& rng = prng.get(rng_);
                 auto& us = out_neighbors[v];
                 us.clear();
                 for (auto u : out_neighbors_range(v, g))
                     us.push_back(u);
                 if (max_rk < us.size())
                 {
                     size_t i = 0;
                     for ([[maybe_unused]] auto u : random_permutation_range(us, rng))
                     {
                         if (++i == max_rk)
                             break;
                     }
                     us.erase(us.begin() + max_rk, us.end());
                 }
             });

        if (verbose)
            cout << "update neighbors" << endl;

        size_t c = 0;
        size_t n = 0;

        #pragma omp parallel if (parallel) firstprivate(seen) reduction(+:c,n,n_tot)
        parallel_loop_no_spawn
            (vs,
             [&](auto, auto v)
             {
                 auto& rng = prng.get(rng_);

                 auto& Bv = B[v];

                 seen.clear();
                 for (auto& [v, l] : Bv)
                     seen.insert(v);

                 auto update =
                     [&](size_t u, size_t w)
                     {
                         if (u == w || w == v)
                             return;

                         auto it = seen.find(w);
                         if (it != seen.end())
                             return;

                         if (!rsample(rng))
                             return;

                         double l = d(w, v, iter);
                         ++n_tot;

                         if (l < get<1>(Bv.front()))
                         {
                             std::pop_heap(Bv.begin(), Bv.end(), cmp);
                             Bv.back() = {w, l};
                             std::push_heap(Bv.begin(), Bv.end(), cmp);
                             c++;
                         }

                         seen.insert(w);
                         n++;
                     };

                 for (auto u : in_neighbors_range(v, g))
                 {
                     for (auto w : in_neighbors_range(u, g))
                         update(u, w);
                     for (auto w : out_neighbors[u])
                         update(u, w);
                 }

                 for (auto u : out_neighbors[v])
                 {
                     update(v, u);
                     for (auto w : in_neighbors_range(u, g))
                         update(u, w);
                     for (auto w : out_neighbors[u])
                         update(u, w);
                 }
             });

        delta = c / double(vs.size() * k);

        if (verbose)
        {
            cout << iter << " " << delta << " " << c;
            if (c_stop)
                cout << " " << clust;
            cout << " " << n << " " << n_tot << endl;
        }

        iter++;

        if (max_iter > 0 && iter == max_iter + 1)
            break;
    }

    for (size_t v : vs)
        clear_vertex(v, g);
    for (auto v : vs)
        build_vertex(v);
    return n_tot;
}

template <bool parallel, class Graph, class Dist, class Weight>
size_t gen_knn_exact(Graph& g, Dist&& d, size_t k, Weight eweight)
{
    std::vector<size_t> vs;
    for (auto v : vertices_range(g))
        vs.push_back(v);
    std::vector<std::vector<std::tuple<size_t, double>>> us(num_vertices(g));
    size_t n_tot = 0;

    #pragma omp parallel if (parallel) reduction(+:n_tot)
    parallel_loop_no_spawn
        (vs,
         [&](auto, auto v)
         {
             auto& ns = us[v];
             for (size_t u : vertices_range(g))
             {
                 if (u == v)
                     continue;
                 ns.emplace_back(u, d(u, v));
                 ++n_tot;
             }
             if (ns.size() <= k)
                 return;
             nth_element(ns.begin(),
                         ns.begin() + k,
                         ns.end(),
                         [](auto& x, auto& y)
                         {
                             return get<1>(x) < get<1>(y);
                         });
             ns.resize(k);
             ns.shrink_to_fit();
         });

    for (auto v : vs)
    {
        for (auto& [u, w] : us[v])
        {
            auto e = add_edge(u, v, g).first;
            eweight[e] = w;
        }
    }
    return n_tot;
}

template <bool parallel, class Graph, class Dist, class Weight>
size_t gen_k_nearest_exact(Graph& g, Dist&& d, size_t k, bool directed,
                           Weight eweight)
{
    std::vector<std::tuple<std::tuple<size_t, size_t>, double>> pairs;

    auto heap = make_shared_heap(pairs, k,
                                 [](auto& x, auto& y)
                                 {
                                     return get<1>(x) < get<1>(y);
                                 });

    std::vector<size_t> vs;
    for (auto v : vertices_range(g))
        vs.push_back(v);

    size_t n_tot = 0;

    #pragma omp parallel if (parallel) firstprivate(heap) reduction(+:n_tot)
    parallel_loop_no_spawn
        (vs,
         [&](auto, auto v)
         {
             for (auto u : vs)
             {
                 if (u == v)
                     continue;
                 if (!directed && u > v)
                     continue;
                 auto l = d(u, v);
                 heap.push({{u, v}, l});
                 ++n_tot;
             }
         });

    heap.merge();

    for (auto& [uv, l] : pairs)
    {
        auto& [u, v] = uv;
        auto e = add_edge(u, v, g).first;
        eweight[e] = l;
    }

    return n_tot;
}

template <class DescriptorProperty>
class MaskFilter
{
public:
    typedef typename boost::property_traits<DescriptorProperty>::value_type value_t;
    MaskFilter(){}
    MaskFilter(DescriptorProperty& filtered_property)
        : _filtered_property(&filtered_property) {}

    template <class Descriptor>
    inline bool operator() (Descriptor&& d) const
    {
        return get(*_filtered_property, d);
    }

    DescriptorProperty& get_filter() { return *_filtered_property; }
    constexpr bool is_inverted() { return false; }

private:
    DescriptorProperty* _filtered_property;
};


template <bool parallel, class Graph, class Dist, class Weight, class Hint, class RNG>
std::tuple<size_t, size_t>
gen_k_nearest(Graph& g, Dist&& d, size_t m, double r, size_t max_rk,
              double epsilon, bool c_stop, size_t max_iter, Weight eweight,
              Hint& hint, bool directed, bool verbose, RNG& rng)
{
    size_t N = num_vertices(g);
    std::vector<bool> select(N, true);
    typename eprop_map_t<bool>::type::unchecked_t eselect(get(edge_index_t(), g));

    auto u = make_filt_graph(g, MaskFilter<decltype(eselect)>(eselect),
                             [&](auto v) { return select[v]; });

    auto uhint = make_filt_graph(hint, keep_all(),
                                 [&](auto v) { return select[v]; });

    size_t n_tot = 0;
    size_t iter = 0;
    while (N > 1)
    {
        iter++;
        if (verbose)
            cout << "m = " << m <<  " nearest iteration: " << iter << endl;

        if (N * N <= 4 * m)
        {
            if (verbose)
                cout << "Running exact m nearest with N = " << N << endl;
            n_tot += gen_k_nearest_exact<parallel>(u, d, m, directed, eweight);
            break;
        }

        size_t nk = ceil((4. * m)/N);

        eselect.reserve(N * nk);

        if (verbose)
            cout << "Running KNN with N = " << N << " and k = " << nk << endl;
        n_tot += gen_knn<parallel>(u, d, nk, r, max_rk, epsilon, c_stop,
                                   max_iter, eweight, uhint, verbose, rng);

        typedef typename graph_traits<Graph>::edge_descriptor edge_t;
        std::vector<std::tuple<edge_t, double>> medges;

        // 2m shortest directed pairs
        auto heap = make_shared_heap(medges, 2 * m,
                                     [](auto& x, auto& y)
                                     {
                                         return get<1>(x) < get<1>(y);
                                     });
        if (verbose)
            cout << "Keeping 2m = " << 2 * m << " of "
                 << nk * N << " closest directed pairs..." << endl;

        #pragma omp parallel if (parallel) firstprivate(heap)
        parallel_edge_loop_no_spawn(u,
                                    [&](auto& e)
                                    { heap.push({e, eweight[e]}); });
        heap.merge();

        if (verbose)
            cout << "heap size: " << medges.size()
                 << ", top: " << get<1>(medges.front()) << endl;

        if (verbose)
            cout << "Selecting nodes..." << endl;

        typename eprop_map_t<bool>::type ekeep(get(edge_index_t(), g));
        for (auto e : edges_range(u))
            ekeep[e] = false;

        parallel_loop(medges,
                      [&](size_t, auto& el)
                      {
                          auto e = get<0>(el);
                          ekeep[e] = true;
                          auto s = source(e, u);
                          auto t = target(e, u);
                          if (directed)
                              return;
                          auto ne = edge(t, s, u);
                          if (ne.second)
                              ekeep[ne.first] = true;
                      }, 0);

        N = 0;
        std::vector<bool> nselect(select);
        #pragma omp parallel if (parallel) reduction(+:N)
        parallel_vertex_loop_no_spawn
            (u,
             [&](auto v)
             {
                 nselect[v] = true;
                 for (auto e : in_edges_range(v, u))
                 {
                     if (!ekeep[e])
                     {
                         nselect[v] = false;
                         break;
                     }
                 }
                 if (nselect[v])
                     N++;
             });

        select = nselect;
        if (N > 1)
        {
            #pragma omp parallel if (parallel)
            parallel_edge_loop_no_spawn(u, [&](auto& e){ eselect[e] = false; });
        }
    };

    if (verbose)
        cout << "Removing parallel edges..." << endl;

    if (!directed)
    {
        undirected_adaptor g_u(g);
        contract_parallel_edges(g_u, dummy_property_map());
    }
    else
    {
        contract_parallel_edges(g, dummy_property_map());
    }

    if (verbose)
        cout << "Selecting best m = " << m << " out of "
             << num_edges(g) << " edges..." << endl;

    std::vector<std::tuple<std::tuple<size_t, size_t>, double>> pairs;

    auto heap = make_shared_heap(pairs, m,
                                 [](auto& x, auto& y)
                                 {
                                     return get<1>(x) < get<1>(y);
                                 });

    #pragma omp parallel if (parallel) firstprivate(heap)
    parallel_edge_loop_no_spawn
        (g,
         [&](auto& e)
         {
             size_t u = source(e, g);
             size_t v = target(e, g);
             if (!directed && u > v)
                 std::swap(u, v);
             auto l = eweight[e];
             heap.push({{u, v}, l});
         });
    heap.merge();

    if (verbose)
        cout << "E = " << pairs.size()
             << ", top: " << get<1>(pairs.front()) << endl;

    if (verbose)
        cout << "Building graph..." << endl;

    for (auto v : vertices_range(g))
        clear_vertex(v, g);

    for (auto& [uv, l] : pairs)
    {
        auto& [u, v] = uv;
        auto e = add_edge(u, v, g).first;
        eweight[e] = l;
    }

    return {n_tot, iter};
}

} // graph_tool namespace

#endif // GRAPH_KNN_HH
