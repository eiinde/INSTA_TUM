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

#ifndef MODULE_REGISTRY_HH
#define MODULE_REGISTRY_HH

#include <functional>
#include <vector>
#include <tuple>
#include <algorithm>
#include <limits>
#include <typeindex>
#include <unordered_map>
#include <any>
#include <boost/python/class.hpp>
#include <boost/python/exception_translator.hpp>

#include "demangle.hh"

#ifndef __MOD__
#error "__MOD__ needs to be defined"
#endif

#define REGISTER_MOD static __MOD__::RegisterMod __reg

namespace __MOD__
{

typedef std::vector<std::tuple<int,std::function<void()>>> reg_t;

reg_t& mod_reg()
#ifdef DEF_REGISTRY
{
    static reg_t* reg = new reg_t();
    return *reg;
}
#else
;
#endif

typedef std::unordered_map<std::type_index, std::any> creg_t;

creg_t& class_reg()
#ifdef DEF_REGISTRY
{
    static creg_t* creg = new creg_t();
    return *creg;
}
#else
;
#endif

class ClassNotFound: public std::exception
{
public:
    ClassNotFound(const std::type_info& tid)
    {
        _msg = std::string("class not found: ") + name_demangle(tid.name());
    }

    const char* what() const noexcept
    {
        return _msg.c_str();
    }

private:
    std::string _msg;
};

template <class... Args, class... As>
boost::python::class_<Args...>& get_class(As&&... as)
{
    typedef boost::python::class_<Args...> class_t;
    auto& creg = class_reg();
    auto idx = std::type_index(typeid(class_t));
    auto iter = creg.find(idx);
    if (iter != creg.end())
        return std::any_cast<class_t&>(iter->second);
    if constexpr (sizeof...(as) == 0)
    {
        throw ClassNotFound(typeid(class_t));
    }
    else
    {
        auto& a = creg[idx];
        a.emplace<class_t>(as...);
        return std::any_cast<class_t&>(a);
    }
}

class RegisterMod
{
public:
    RegisterMod(std::function<void()> f, int p = 0)
    {
        mod_reg().emplace_back(p, f);
    }
};

class EvokeRegistry
{
public:
    EvokeRegistry()
    {
        boost::python::register_exception_translator<ClassNotFound>
            ([](const auto& e)
             {
                 PyObject* error = PyExc_RuntimeError;
                 PyErr_SetString(error, e.what());
             });
        reg_t& reg = mod_reg();
        std::sort(reg.begin(), reg.end(),
                  [](const auto& a, const auto& b)
                  { return std::get<0>(a) < std::get<0>(b); });
        for (auto& [p, f] : reg)
            f();
        delete &reg;
        delete &class_reg();
    }
};

} // namespace __MOD__

#endif // MODULE_REGISTRY_HH
