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

#ifndef DYNAMICS_CONTINUOUS_HH
#define DYNAMICS_CONTINUOUS_HH

#include "config.h"

#include <vector>

#define BOOST_PYTHON_MAX_ARITY 30
#include "graph_python_interface.hh"

#include "dynamics_base.hh"
#include "dynamics_util.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

template <class Base, bool keep_k, bool tshift>
class ContinuousStateBase
    : public NSumStateBase<Base, false, keep_k, tshift>
{
public:
    typedef NSumStateBase<Base, false, keep_k, tshift> base_t;

    template <class... Args>
    ContinuousStateBase(Args&&... s)
        : NSumStateBase<Base, false, keep_k, tshift>(s...)
    {}

    template <class Val>
    constexpr Val transform_input(size_t, size_t, Val x) { return x; }

    template <class Val>
    Val transform_theta(Val x, Val) { return x; }
};


template <class T>
[[gnu::flatten]] [[gnu::always_inline]]
inline double l2sinha(T x) // log((exp(x) - exp(-x))/x)
{
    x = abs(x);
    if (x < 1e-8)
        return log(2);
    return x + log1p(-exp(-2*x)) - log(x);
}

class CIsingBaseState
{
public:
    double log_P(double theta, double m, double s)
    {
        double x = theta + m;
        return s * x - l2sinha(x);
    }
};

class PseudoCIsingState
    : public ContinuousStateBase<PseudoCIsingState, false, false>,
      public CIsingBaseState
{
public:
    using typename ContinuousStateBase<PseudoCIsingState, false, false>::base_t;

    template <class... Args>
    PseudoCIsingState(python::dict, Args&&... s)
        : ContinuousStateBase<PseudoCIsingState, false, false>(s...) {}
};

class CIsingGlauberState
    : public ContinuousStateBase<CIsingGlauberState, false, true>,
      public CIsingBaseState
{
public:
    using typename ContinuousStateBase<CIsingGlauberState, false, true>::base_t;

    template <class... Args>
    CIsingGlauberState(python::dict, Args&&... s)
        : ContinuousStateBase<CIsingGlauberState, false, true>(s...) {}

    double log_P(double theta, double m, double, double sn)
    {
        return CIsingBaseState::log_P(theta, m, sn);
    }
};

class PseudoNormalState
    : public ContinuousStateBase<PseudoNormalState, true, false>
{
public:
    using typename ContinuousStateBase<PseudoNormalState, true, false>::base_t;

    template <class... Args>
    PseudoNormalState(python::dict params, Args&&... s)
        : ContinuousStateBase<PseudoNormalState, true, false>(s...)
    {
        set_params(params);
    };

    virtual void set_params(python::dict p)
    {
        _positive = python::extract<bool>(p["positive"]);
        _pslack = python::extract<double>(p["pslack"]);
    }

    template <class Val>
    Val transform_theta(Val theta, Val k)
    {
        return (_positive && k > 0) ?
            std::min(-std::log(k)/2 - _pslack, theta) :
            theta;
    }

    [[gnu::flatten]] [[gnu::always_inline]]
    double log_P(double theta, double m, double, double x)
    {
        double lsigma = theta;
        double sigma = exp(lsigma);
        double a = exp(2*lsigma);
        return norm_lpdf(x, -a * m, sigma, lsigma);
    }

    bool _positive = true;
    double _pslack = 0;
};

class NormalGlauberState
    : public ContinuousStateBase<NormalGlauberState, false, true>
{
public:
    using typename ContinuousStateBase<NormalGlauberState, false, true>::base_t;

    template <class... Args>
    NormalGlauberState(python::dict, Args&&... s)
        : ContinuousStateBase<NormalGlauberState, false, true>(s...)
    {};

    [[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
    double log_P(double theta, double m, double, double nx)
    {
        double lsigma = theta;
        double sigma = exp(lsigma);
        double a = exp(2*lsigma);
        return norm_lpdf(nx, -a * m, sigma, lsigma);
    }
};


class LinearNormalState
    : public ContinuousStateBase<LinearNormalState, false, true>
{
public:
    using typename ContinuousStateBase<LinearNormalState, false, true>::base_t;

    template <class... Args>
    LinearNormalState(python::dict, Args&&... s)
        : ContinuousStateBase<LinearNormalState, false, true>(s...)
    {};

    [[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
    double log_P(double theta, double m, double x, double nx)
    {
        double sigma = exp(theta);
        double a = x + m;
        return norm_lpdf(nx, a, sigma, theta);
    }
};

class LVState
    : public ContinuousStateBase<LVState, false, true>
{
public:
    using typename ContinuousStateBase<LVState, false, true>::base_t;

    template <class... Args>
    LVState(python::dict p, Args&&... s)
        : ContinuousStateBase<LVState, false, true>(s...)
    {
        set_params(p);
    };

    virtual void set_params(python::dict p)
    {
        _sigma = python::extract<double>(p["sigma"]);
        _lsigma = log(_sigma);
    }

    [[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
    double log_P(double theta, double m, double x, double nx)
    {
        double diff = x * (theta + m);
        double sigma = _sigma * sqrt(x);
        return norm_lpdf(nx, x + diff, sigma, _lsigma + log(x)/2);
    }

    double _sigma = 1;
    double _lsigma = 0;
};

} // graph_tool namespace

#endif //DYNAMICS_CONTINUOUS_HH
