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

#ifndef INT_PART_HH
#define INT_PART_HH

#include "config.h"

#include <boost/multi_array.hpp>

namespace graph_tool
{
using namespace boost;

void init_q_cache(size_t n_max);
void clear_q_cache();
double q_rec(int n, int k);
double q_rec_memo(int n, int k);
double log_q_approx(size_t n, size_t k);
double log_q_approx_big(size_t n, size_t k);
double log_q_approx_small(size_t n, size_t k);

extern boost::multi_array<double, 2> __q_cache;

template <class T>
[[gnu::const]] [[gnu::hot]]
double log_q(T n, T k)
{
    if (n <= 0 || k < 1)
        return 0;
    if (k > n)
        k = n;
    if (size_t(n) < __q_cache.shape()[0])
        return __q_cache[n][k];
    return log_q_approx(n, k);
}

}
#endif // INT_PART_HH
