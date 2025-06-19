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

#ifndef CACHE_HH
#define CACHE_HH

#include "config.h"
#include "../../openmp.hh"

#include <vector>
#include <cmath>

namespace graph_tool
{
using namespace std;

// Repeated computation of x*log(x) and log(x) actually adds up to a lot of
// time. A significant speedup can be made by caching pre-computed values.

extern vector<vector<double>> __safelog_cache;
extern vector<vector<double>> __xlogx_cache;
extern vector<vector<double>> __lgamma_cache;

constexpr size_t __max_size = (500 * (1 << 20)) / sizeof(double);

template <class T>
size_t get_size(T n)
{
    size_t k = 1;
    while (k < size_t(n))
        k <<= 1;
    return k;
}

template <bool Init=true, class T, class F, class Cache>
[[gnu::pure]]
inline double get_cached(T x, F&& f, Cache& tcache)
{
    auto t = get_thread_num();
    auto& cache = tcache[t];
    if (size_t(x) >= cache.size())
    {
        if (Init && size_t(x) < __max_size)
        {
            size_t old_size = cache.size();
            if (size_t(x) >= old_size)
            {
                cache.resize(get_size(x + 1));
                for (size_t y = old_size; y < cache.size(); ++y)
                    cache[y] = f(y);
            }
        }
        else
        {
            return f(x);
        }
    }
    return cache[x];
}

template <class T>
[[gnu::const]]
inline double safelog(T x)
{
    if (x == 0)
        return 0;
    return log(x);
}

template <bool Init=true, class T>
[[gnu::pure]]
inline double safelog_fast(T x)
{
    return get_cached<Init>(x, [](T x) { return safelog(x); }, __safelog_cache);
}

template <class T>
[[gnu::const]]
inline double xlogx(T x)
{
    return x * safelog(x);
}

template <bool Init=true, class T>
[[gnu::pure]]
inline double xlogx_fast(T x)
{
    return get_cached<Init>(x, [](T x) { return xlogx(x); }, __xlogx_cache);
}

template <bool Init=true, class T>
[[gnu::pure]]
inline double lgamma_fast(T x)
{
    return get_cached<Init>(x, [](T x) { return lgamma(x); }, __lgamma_cache);
}

void init_cache();

} // graph_tool namespace

#endif //CACHE_HH
