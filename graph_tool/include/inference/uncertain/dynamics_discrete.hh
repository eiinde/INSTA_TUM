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

#ifndef DYNAMICS_DISCRETE_HH
#define DYNAMICS_DISCRETE_HH

#include "config.h"

#include <vector>

#define BOOST_PYTHON_MAX_ARITY 30

#include "graph_python_interface.hh"
#include "dynamics_base.hh"

#include "../../support/util.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

template <class Spec, bool keep_k, bool tshift, int xmin>
class DiscreteStateBase
    : public NSumStateBase<Spec, true, keep_k, tshift>
{
public:
    template <class... Args>
    DiscreteStateBase(Args&&... s)
        : NSumStateBase<Spec, true, keep_k, tshift>(s...)
    {}

    template <class Val>
    constexpr Val transform_input(size_t, size_t, Val x) { return x; }

    template <class Val>
    Val transform_theta(Val x, Val) { return x; }

    typedef NSumStateBase<Spec, true, keep_k, tshift> base_t;
};

class SIState: public DiscreteStateBase<SIState, false, true, 0>
{
public:
    using typename DiscreteStateBase<SIState, false, true, 0>::base_t;

    enum State { S, I, R, E };

    template <class... Args>
    SIState(python::dict params, Args&&... s)
        : DiscreteStateBase<SIState, false, true, 0>(s...),
          _exposed(python::extract<bool>(params["exposed"])),
          _E(_exposed ? State::E : State::I)
    {
        set_params(params);
    };

    typedef typename vprop_map_t<std::vector<uint8_t>>::type amap_t;

    virtual void set_params(python::dict) {}

    int transform_input(size_t, size_t, int x) { return x == State::I; }

    [[gnu::pure]]
    double log_P(double theta, double m, int s, int ns)
    {
        double lr = theta;
        double xl = log1p(-exp(m)) + log1p(-exp(lr));
        double a = min(xl, lr);
        double b = max(xl, lr);
        double lp = b + log1p(exp(a-b));
        return (s == State::S) * ((ns == _E) * lp + (ns != _E) * log1p(-exp(lp)));
    }

private:
    bool _exposed;
    int _E;
};

template <class T>
[[gnu::const]] [[gnu::flatten]] [[gnu::always_inline]]
inline double l2cosh(T x) // log(exp(x) + exp(-x))
{
    auto y = std::abs(x);
    return y + std::log1p(std::exp(-2*y));
    //return log_sum_exp(x, -x);
}

template <class T>
[[gnu::const]] [[gnu::flatten]] [[gnu::always_inline]]
inline double l1p2cosh(T x) // log(1 + exp(x) + exp(-x))
{
    auto y = std::abs(x);
    return y + std::log1p(std::exp(-y) + std::exp(-2*y));
    //return log_sum_exp(0, x, -x);
}

class IsingBaseState
{
public:
    typedef vprop_map_t<int>::type::unchecked_t theta_t;

    using double_int_t = typename std::conditional<sizeof(double) == sizeof(uint32_t), uint32_t,
                         typename std::conditional<sizeof(double) == sizeof(uint64_t), uint64_t,
                                                   void>::type>::type;

    IsingBaseState(python::dict params)
    {
        set_params(params);
    };

    virtual void set_params(python::dict params)
    {
        _has_zero = python::extract<bool>(params["has_zero"]);
    }

    [[gnu::flatten]] [[gnu::always_inline]]
    double get_l2cosh(double x)
    {
        if (_has_zero)
            return l1p2cosh(x);
        else
            return l2cosh(x);
    }

    [[gnu::flatten]] [[gnu::always_inline]]
    double log_P(double theta, double m, int s)
    {
        double x = theta + m;
        return s * x - get_l2cosh(x);
    }

private:
    bool _has_zero;
};

class PseudoIsingState
    : public DiscreteStateBase<PseudoIsingState, false, false, -1>,
      public IsingBaseState
{
public:
    template <class... Args>
    PseudoIsingState(python::dict params, Args&&... s)
        : DiscreteStateBase<PseudoIsingState, false, false, -1>(s...),
          IsingBaseState(params) {};

    using typename DiscreteStateBase<PseudoIsingState, false, false, -1>::base_t;
};

class IsingGlauberState
    : public DiscreteStateBase<IsingGlauberState, false, true, -1>,
      public IsingBaseState
{
public:
    using typename DiscreteStateBase<IsingGlauberState, false, true, -1>::base_t;

    template <class... Args>
    IsingGlauberState(python::dict params, Args&&... s)
        : DiscreteStateBase<IsingGlauberState, false, true, -1>(s...),
          IsingBaseState(params) {};

    [[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
    double log_P(double theta, double m, int, int ns)
    {
        return IsingBaseState::log_P(theta, m, ns);
    }
};

} // graph_tool namespace

#endif //DYNAMICS_DISCRETE_HH
