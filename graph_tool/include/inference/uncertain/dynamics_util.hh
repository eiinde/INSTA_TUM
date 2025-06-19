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

#ifndef DYNAMICS_UTIL_HH
#define DYNAMICS_UTIL_HH

#include <cmath>

template <class T>
[[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
inline double norm_lpdf(T x, T m, T s, T log_s)
{
    T a = (x - m) / s;
#ifndef __clang__
    constexpr
#endif
        double l2pi = std::log(2 * M_PI);
    return - ((a * a) + l2pi) / 2 - log_s;
}

template <class T>
[[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
inline double norm_lpdf(T x, T m, T s)
{
    return norm_lpdf(x, m, s, log(s));
}

template <class T>
[[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
inline double laplace_lpdf(T x, T lambda)
{
    return -lambda * abs(x) + log(lambda) - log(2);
}

template <class T>
[[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
inline double qlaplace_lprob(T x, T lambda, T delta, bool nonzero)
{
    if (delta == 0)
        return laplace_lpdf(x, lambda);

    if (nonzero)
    {
        return -lambda * abs(x) + 2 * lambda * delta + log1p(-exp(-2*lambda*delta)) - log(2);
    }
    else
    {
        if (x == 0)
            return log1p(-exp(-lambda * delta));
        return -lambda * abs(x) + lambda * delta + log1p(-exp(-2 * lambda * delta)) - log(2);
    }
}

template <class T>
[[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
inline double exponential_lpdf(T x, T lambda)
{
    return -lambda * x + log(lambda);
}


#endif // DYNAMICS_UTIL_HH
