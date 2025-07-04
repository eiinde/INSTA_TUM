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

#ifndef STR_REPR_HH
#define STR_REPR_HH

#include <clocale>
#include <sstream>
#include <boost/lexical_cast.hpp>

//
// Data type string representation
// ===============================
//
// String representation of individual data types. Among other things, we have
// to take care specifically that no information is lost with floating point
// I/O.
//

namespace boost
{

//
// "chars" should be printed as numbers, since they can be non-printable
//

template <>
std::string lexical_cast<std::string,uint8_t>(const uint8_t& val)
{
    return lexical_cast<std::string>(int(val));
}

template <>
uint8_t lexical_cast<uint8_t,std::string>(const std::string& val)
{
    return uint8_t(lexical_cast<int>(val));
}

// float, double and long double should be printed with enough precision
// preserve internal representation. (we also need to make sure the
// representation is locale-independent).

template <class Val>
std::string print_float(Val val)
{
    std::ostringstream s;
    s.imbue(std::locale("C"));
    s << std::setprecision(std::numeric_limits<Val>::max_digits10);
    s << val;
    return s.str();
}

static int scan_float_dispatch(const char* str, float& val)
{
    return sscanf(str, "%a", &val);
}

static int scan_float_dispatch(const char* str, double& val)
{
    return sscanf(str, "%la", &val);
}

static int scan_float_dispatch(const char* str, long double& val)
{
    return sscanf(str, "%La", &val);
}

template <class Val>
static int scan_float(const char* str, Val& val)
{
    char* locale = setlocale(LC_NUMERIC, NULL);
    setlocale(LC_NUMERIC, "C");
    int retval = scan_float_dispatch(str, val);
    setlocale(LC_NUMERIC, locale);
    return retval;
}


template <>
std::string lexical_cast<std::string,float>(const float& val)
{
    return print_float(val);
}

template <>
float lexical_cast<float,std::string>(const std::string& val)
{
    float ret;
    int nc = scan_float(val.c_str(), ret);
    if (nc != 1)
        throw bad_lexical_cast();
    return ret;
}

template <>
std::string lexical_cast<std::string,double>(const double& val)
{
    return print_float(val);
}

template <>
double lexical_cast<double,std::string>(const std::string& val)
{
    double ret;
    int nc = scan_float(val.c_str(), ret);
    if (nc != 1)
        throw bad_lexical_cast();
    return ret;
}

template <>
std::string lexical_cast<std::string,long double>(const long double& val)
{
    return print_float(val);
}

template <>
long double lexical_cast<long double,std::string>(const std::string& val)
{
    long double ret;
    int nc = scan_float(val.c_str(), ret);
    if (nc != 1)
        throw bad_lexical_cast();
    return ret;
}
} // namespace boost

//
// stream i/o of std::vector<>
//

namespace std
{

// string vectors need special attention, since separators must be properly
// escaped.
template <>
ostream& operator<<(ostream& out, const std::vector<std::string>& vec)
{
    for (size_t i = 0; i < vec.size(); ++i)
    {
        std::string s = vec[i];
        // escape separators
        boost::replace_all(s, "\\", "\\\\");
        boost::replace_all(s, ", ", ",\\ ");

        out << s;
        if (i < vec.size() - 1)
            out << ", ";
    }
    return out;
}

template <>
istream& operator>>(istream& in, std::vector<std::string>& vec)
{
    using namespace boost;
    using namespace boost::algorithm;
    using namespace boost::xpressive;

    vec.clear();
    std::string data;
    while (in.good())
    {
        std::string line;
        getline(in, line);
        data += line;
    }

    if (data == "")
        return in; // empty string is OK

    sregex re = sregex::compile(", ");
    sregex_token_iterator iter(data.begin(), data.end(), re, -1), end;
    for (; iter != end; ++iter)
    {
        vec.push_back(*iter);
        // un-escape separators
        boost::replace_all(vec.back(), ",\\ ", ", ");
        boost::replace_all(vec.back(), "\\\\", "\\");
    }
    return in;
}

} // std namespace

#endif // STR_REPR_HH
