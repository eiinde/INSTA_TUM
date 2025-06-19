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

#ifndef NESTED_FOR_LOOP_HH
#define NESTED_FOR_LOOP_HH

namespace boost
{
namespace mpl
{
// The following is a implementation of a nested for_each loop, which runs a
// given Action functor for each combination of its arguments, given by the type
// ranges, as such:
//
//     struct foo
//     {
//         template<class T1, class T2, class T3>
//         void operator()(T1, T2, T3) const
//         {
//             ...
//         }
//     };
//
//     ...
//
//     typedef mpl::vector<int,float,long> r1;
//     typedef mpl::vector<string,double> r2;
//     typedef mpl::vector<size_t,char> r3;
//
//     any x = float(2);
//     any y = string("foo");
//     any z = size_t(42);
//
//     bool found = nested_for_each<r1,r2,r3>(foo(), x, y, z);
//
// The code above will run iterate through all combinations of foo::operator(T1,
// T2, T3) and call the one that corresponds to the actual types stored in x, y,
// and z. If the types are not found during iteration, we have found == true,
// otherwise found == false. This provides a more general compile-time to
// run-time bridge than the simpler mpl::for_each().


// this is a functor wrapper that will perform an any_cast<> in each in an array
// of arguments according to the called types. If the cast is successful, the
// function will be called with those types, and true will be returned.

// recursion-free variadic version of for_each
template <class...>
struct for_each_variadic;

template <class F, class... Ts>
struct for_each_variadic<F,std::tuple<Ts...>>
{
    bool operator()(F f)
    {
        auto call = [&](auto&& arg) -> bool {return f(std::forward<decltype(arg)>(arg));};
        return (call(typename std::add_pointer<Ts>::type()) || ...);
    }
};

// convert mpl sequence to std::tuple
template <class T, class R>
struct to_tuple_imp;

template <class... Ts, class X>
struct to_tuple_imp<std::tuple<Ts...>, X>
{
    typedef std::tuple<Ts..., X> type;
};

template <class Seq>
struct to_tuple
{
    typedef typename mpl::fold<Seq, std::tuple<>,
                               to_tuple_imp<mpl::_1, mpl::_2>>::type type;
};

template <class Seq>
using to_tuple_t = typename to_tuple<Seq>::type;

// nested type loops via variadic templates

template <class...>
struct inner_loop {};

template <class Action, class... Ts>
struct inner_loop<Action, std::tuple<Ts...>>
{
    inner_loop(Action a): _a(a) {}

    template <class T>
    [[gnu::always_inline]]
    bool operator()(T*) const
    { return _a(typename std::add_pointer<Ts>::type()...,
                typename std::add_pointer<T>::type()); }  // innermost loop
    Action _a;
};

template <class Action, class... Ts, class TR1, class... TRS>
struct inner_loop<Action, std::tuple<Ts...>, TR1, TRS...>
{
    inner_loop(Action a): _a(a) {}

    template <class T>
    [[gnu::always_inline]]
    bool operator()(T*) const
    {
        typedef inner_loop<Action, std::tuple<Ts..., T>, TRS...> inner_loop_t;
        typedef typename to_tuple<TR1>::type tr_tuple;
        return for_each_variadic<inner_loop_t, tr_tuple>()(inner_loop_t(_a));
    }
    Action _a;
};

// final function

template <class TR1, class... TRS, class Action>
void nested_for_each(Action a)
{
    typedef typename to_tuple<TR1>::type tr_tuple;

    // wrap action into a bool-returning function
    auto ab = [=](auto*... args) -> bool { a(args...); return false; };

    typedef inner_loop<decltype(ab), std::tuple<>, TRS...> inner_loop_t;
    for_each_variadic<inner_loop_t, tr_tuple>()(inner_loop_t(ab));
}


} // mpl namespace
} // boost namespace

#endif //NESTED_FOR_LOOP_HH
