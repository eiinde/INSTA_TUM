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

#ifndef GRAPH_HISTOGRAM_HH
#define GRAPH_HISTOGRAM_HH

#include "config.h"

#include <vector>

#include "../blockmodel/graph_blockmodel_util.hh"
#include "../support/graph_state.hh"

#include "../../idx_map.hh"

#include <boost/container/static_vector.hpp>
#include <boost/math/special_functions/zeta.hpp>

template <class T, size_t D>
struct empty_key<boost::container::static_vector<T,D>>
{
    static boost::container::static_vector<T,D> get()
    {
        boost::container::static_vector<T,D> x(D);
        for (size_t i = 0; i < D; ++i)
            x[i] = empty_key<T>::get();
        return x;
    }
};

template <class T, size_t D>
struct deleted_key<boost::container::static_vector<T,D>>
{
    static boost::container::static_vector<T,D> get()
    {
        boost::container::static_vector<T,D> x(D);
        for (size_t i = 0; i < D; ++i)
            x[i] = deleted_key<T>::get();
        return x;
    }
};


namespace std
{
template <class Value, size_t D>
struct hash<boost::container::static_vector<Value, D>>
{
    size_t operator()(const boost::container::static_vector<Value, D>& v) const
    {
        size_t seed = 0;
        for (const auto& x : v)
            std::_hash_combine(seed, x);
        return seed;
    }
};

}

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef mpl::vector2<multi_array_ref<double,2>,
                     multi_array_ref<int64_t,2>> x_ts;

typedef mpl::vector1<multi_array_ref<uint64_t,1>> w_ts;

#define HIST_STATE_params                                                     \
    ((__class__,&, mpl::vector<python::object>, 1))                           \
    ((x_r,, x_ts, 1))                                                         \
    ((w_r,, w_ts, 1))                                                         \
    ((obins,, python::list, 0))                                               \
    ((obounded,, python::list, 0))                                            \
    ((odiscrete,, python::list, 0))                                           \
    ((ocategorical,, python::list, 0))                                        \
    ((alpha,, double, 0))                                                     \
    ((pcount,, double, 0))                                                    \
    ((conditional,, size_t, 0))

GEN_STATE_BASE(HistStateBase, HIST_STATE_params)

template <size_t n>
struct HVa
{
    template <class T>
    using type = std::array<T, n>;
};

template <class T>
using HVec = std::vector<T>;

template <class Value>
bool is_valid_val(Value x)
{
    if constexpr (!std::numeric_limits<Value>::has_quiet_NaN)
    {
        if (x == numeric_limits<Value>::max())
            return false;
    }
    else
    {
        if (std::isnan(x))
            return false;
    }
    return true;
}

template <class Value>
void set_invalid(Value& x)
{
    if constexpr (!std::numeric_limits<Value>::has_quiet_NaN)
        x = numeric_limits<Value>::max();
    else
        x = numeric_limits<Value>::quiet_NaN();
}

template <template <class T> class VT>
class HistD
{
public:
    template <class... Ts>
    class HistState
        : public HistStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(HistStateBase<Ts...>, HIST_STATE_params)
        GET_PARAMS_TYPEDEF(Ts, HIST_STATE_params)

        typedef typename x_r_t::element value_t;

        typedef multi_array<value_t,2> x_t;
        typedef std::vector<uint64_t> w_t;

        typedef std::vector<value_t> bins_t;

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
        HistState(ATs&&... args)
            : HistStateBase<Ts...>(std::forward<ATs>(args)...),
              _x(_x_r),
              _w(_w_r.begin(), _w_r.end()),
              _N(0),
              _D(_x.shape()[1]),
              _mgroups(_D),
              _mpos(_D)
        {
            for (size_t j = 0; j < _D; ++j)
                _bins.push_back(&python::extract<bins_t&>(_obins[j])());

            for (size_t j = 0; j < _D; ++j)
                _bounded.push_back({python::extract<bool>(_obounded[j][0])(),
                                    python::extract<bool>(_obounded[j][1])()});

            for (size_t j = 0; j < _D; ++j)
                _discrete.push_back(python::extract<bool>(_odiscrete[j])());

            for (size_t j = 0; j < _D; ++j)
                _categorical.push_back(python::extract<bool>(_ocategorical[j])());

            for (size_t i = 0; i < _x.shape()[0]; ++i)
            {
                if (!is_valid(i))
                    continue;
                update_hist<true, false>(i);
            }
        }

        x_t _x;
        w_t _w;

        size_t _N;
        size_t _D;

        std::vector<bins_t*> _bins;
        std::vector<std::pair<value_t, value_t>> _bounds;
        std::vector<std::pair<bool, bool>> _bounded;
        std::vector<bool> _discrete;
        std::vector<bool> _categorical;

        typedef VT<value_t> group_t;
        typedef is_instance<group_t, std::vector> is_vec;
        typedef std::conditional_t
            <is_vec{},
             group_t,
             boost::container::static_vector
                 <value_t,
                  std::tuple_size<std::conditional_t
                                    <is_vec{},
                                     std::array<value_t,1>,
                                     group_t>>::value>>
            cgroup_t;

        gt_hash_map<group_t, size_t> _hist;
        gt_hash_map<cgroup_t, size_t> _chist;

        typedef idx_set<size_t, true> mgroup_t;
        std::vector<gt_hash_map<value_t, mgroup_t>> _mgroups;
        std::vector<std::vector<size_t>> _mpos;

        group_t _r_temp;

        bool is_valid(size_t i)
        {
            bool valid = true;
            for (size_t j = 0; j < _D; ++j)
            {
                if (!is_valid_val(_x[i][j]))
                {
                    valid = false;
                    break;
                }
            }
            return valid;
        }

        mgroup_t _dummy_set;
        mgroup_t& get_mgroup(size_t j, value_t x, bool dummy=false)
        {
            auto& m = _mgroups[j];
            auto iter = m.find(x);
            if (iter == m.end())
            {
                if (dummy)
                    return _dummy_set;
                iter = m.insert({x, idx_set<size_t, true>(_mpos[j])}).first;
            }
            return iter->second;
        }

        void reset_mgroups()
        {
            for (auto& mgroup : _mgroups)
                mgroup.clear();
            for (auto& mpos : _mpos)
                mpos.clear();

            for (size_t i = 0; i < _x.shape()[0]; ++i)
            {
                if (!is_valid(i))
                    continue;

                auto r = get_bin(_x[i]);
                for (size_t j = 0; j < _D; ++j)
                {
                    auto& vs = get_mgroup(j, r[j]);
                    vs.insert(i);
                }
            }
        }

        template <class V>
        group_t get_bin(V&& x)
        {
            group_t r = group_t();
            for (size_t j = 0; j < _D; ++j)
            {
                if (!_categorical[j])
                {
                    auto& bins = *_bins[j];

                    assert(x[j] >= *bins.begin());
                    assert(x[j] < *bins.rbegin());

                    auto iter = std::upper_bound(bins.begin(), bins.end(), x[j]) - 1;

                    if constexpr (is_instance<VT<value_t>, std::vector>{})
                        r.push_back(*iter);
                    else
                        r[j] = *iter;
                }
                else
                {
                    if constexpr (is_instance<VT<value_t>, std::vector>{})
                        r.push_back(size_t(x[j]));
                    else
                        r[j] = size_t(x[j]);
                }
            }
            return r;
        }

        size_t get_hist(const group_t& r)
        {
            auto iter = _hist.find(r);
            if (iter != _hist.end())
                return iter->second;
            return 0;
        }

        size_t get_chist(const cgroup_t& cr)
        {
            auto iter = _chist.find(cr);
            if (iter != _chist.end())
                return iter->second;
            return 0;
        }

        template <class V>
        const group_t& to_group(V&& r)
        {
            auto& nr = _r_temp;
            if constexpr (is_instance<VT<value_t>, std::vector>{})
            {
                nr.clear();
                nr.insert(nr.end(), r.begin(), r.end());
            }
            else
            {
                for (size_t i = 0; i < r.size(); ++i)
                    nr[i] = r[i];
            }
            return nr;
        }

        cgroup_t to_cgroup(const group_t& r)
        {
            cgroup_t cr;
            cr.insert(cr.end(), r.begin() + _conditional, r.end());
            return cr;
        }

        template <class V>
        size_t get_hist(V&& r)
        {
            return get_hist(to_group(r));
        }

        size_t get_hist(size_t i)
        {
            return get_hist(get_bin(_x[i]));
        }

        template <bool Add, bool update_mgroup, bool conditional=true>
        void update_hist(size_t i, const group_t& r, size_t w)
        {
            if constexpr (Add)
            {
                _hist[r] += w;

                if constexpr (update_mgroup)
                {
                    for (size_t j = 0; j < _D; ++j)
                    {
                        auto& vs = get_mgroup(j, r[j]);
                        vs.insert(i);
                    }
                }

                if constexpr (conditional)
                {
                    if (_conditional < _D)
                        _chist[to_cgroup(r)] += w;
                }

                _N += w;
            }
            else
            {
                auto iter = _hist.find(r);
                assert(iter != _hist.end());
                assert(iter->second >= w);
                iter->second -= w;
                if (iter->second == 0)
                    _hist.erase(iter);

                if constexpr (update_mgroup)
                {
                    for (size_t j = 0; j < _D; ++j)
                    {
                        auto& vs = get_mgroup(j, r[j]);
                        vs.erase(i);
                        if (vs.empty())
                            _mgroups[j].erase(r[j]);
                    }
                }

                if constexpr (conditional)
                {
                    if (_conditional < _D)
                    {
                        auto iter = _chist.find(to_cgroup(r));
                        iter->second -= w;
                        if (iter->second == 0)
                            _chist.erase(iter);
                    }
                }

                _N -= w;
            }
        }

        template <bool Add, bool update_mgroups, bool conditional, class V>
        void update_hist(size_t i, V&& r, size_t w)
        {
            update_hist<Add, update_mgroups, conditional>(i, to_group(r), w);
        }

        template <bool Add, bool update_mgroups, bool conditional=true>
        void update_hist(size_t i)
        {
            auto r = get_bin(_x[i]);
            size_t w = _w.empty() ? 1 : _w[i];
            update_hist<Add, update_mgroups, conditional>(i, r, w);
        }

        double get_lw(const group_t& r)
        {
            double lw = 0;
            for (size_t j = 0; j < _conditional; ++j)
            {
                auto x = r[j];
                auto& bin = *_bins[j];
                auto iter = std::lower_bound(bin.begin(),
                                             bin.end(), x);
                assert(*(iter+1) > *iter);
                lw += log(*(iter+1) - *iter);
            }
            return lw;
        }

        double entropy_group(size_t n, double lw)
        {
            double S = 0;

            if (_pcount == 1)
                S -= lgamma_fast(n + 1);
            else
                S -= lgamma(n + _pcount) - lgamma(_pcount);

            S += n * lw;

            return S;
        }

        double entropy_group(const group_t& r, size_t n)
        {
            return entropy_group(n, get_lw(r));
        }

        double get_Mx()
        {
            if (_pcount == 1)
            {
                size_t Mx = 1;
                for (size_t j = 0; j < _conditional; ++j)
                    Mx *= _bins[j]->size() - 1;
                return Mx;
            }
            else
            {
                double Mx = 1;
                for (size_t j = 0; j < _conditional; ++j)
                    Mx *= (_bins[j]->size() - 1) * _pcount;
                return Mx;
            }
        }

        double entropy_cgroup(size_t n, double Mx)
        {
            return lgamma(Mx + n) - lgamma(Mx);
        }

        double entropy_cgroup(size_t n)
        {
            return entropy_group(n, get_Mx());
        }

        double get_M()
        {
            double M = 1;
            for (size_t j = 0; j < _D; ++j)
                M *= (_bins[j]->size() - 1) * _pcount;
            return M;
        }

        double entropy()
        {
            double S = 0;

            S += _D * safelog(_N);

            auto zalpha = boost::math::zeta(_alpha);
            auto lalpha = log(_alpha);
            auto leps = log(std::numeric_limits<double>::epsilon());

            for (size_t j = 0; j < _D; ++j)
            {
                if (_categorical[j])
                    continue;
                size_t Md = _bins[j]->size() - 1;
                auto delta = *(_bins[j]->end()-1) - *_bins[j]->begin();
                assert(delta > 0);
                if (_discrete[j])
                    S += lbinom(delta - 1, value_t(Md - 1)) + _alpha * log(delta) + zalpha;
                else
                    S += (Md + _alpha + 1) * log(delta) + lgamma(Md) - lalpha - _alpha * leps;
            }

            if (_conditional >= _D)
            {
                double M = get_M();
                S += lgamma(_N + M) - lgamma(M);
            }
            else
            {
                auto Mx = get_Mx();
                for (auto& [cr, n] : _chist)
                    S += entropy_cgroup(n, Mx);
            }

            for (auto& nrc : _hist)
            {
                auto& [r, n] = nrc;
                S += entropy_group(r, n);
            }

            return S;
        }

        // =========================================================================
        // State modification
        // =========================================================================

        template <bool add, class VS>
        void update_vs(size_t j, VS& vs)
        {
            if (j < _conditional)
            {
                for (auto& v : vs)
                    update_hist<add, true, false>(v);
            }
            else
            {
                for (auto& v : vs)
                    update_hist<add, true, true>(v);
            }
        }

        void move_edge(size_t j, size_t i, value_t y)
        {
            auto& bins_j = *_bins[j];
            auto x = bins_j[i];
            auto& mvs = get_mgroup(j, x, true);
            std::vector<size_t> vs(mvs.begin(), mvs.end());

            if (i > 0)
            {
                auto xn = bins_j[i-1];
                auto& nvs = get_mgroup(j, xn, true);
                vs.insert(vs.end(), nvs.begin(), nvs.end());
            }

            update_vs<false>(j, vs);

            bins_j[i] = y;

            update_vs<true>(j, vs);
        }

        void remove_edge(size_t j, size_t i)
        {
            auto& bins_j = *_bins[j];
            auto x = bins_j[i];
            auto& mvs = get_mgroup(j, x, true);
            std::vector<size_t> vs(mvs.begin(), mvs.end());

            update_vs<false>(j, vs);

            bins_j.erase(bins_j.begin() + i);

            update_vs<true>(j, vs);
        }

        void add_edge(size_t j, size_t i, value_t y)
        {
            auto& bins_j = *_bins[j];
            auto x = bins_j[i];
            auto& mvs = get_mgroup(j, x, true);
            std::vector<size_t> vs(mvs.begin(), mvs.end());

            update_vs<false>(j, vs);

            bins_j.insert(bins_j.begin() + i + 1, y);

            update_vs<true>(j, vs);
        }

        template <class V>
        void get_rs(V& vs, gt_hash_set<group_t>& rs)
        {
            for (auto v : vs)
                rs.insert(get_bin(_x[v]));
        }

        gt_hash_set<group_t> _rs;
        gt_hash_set<cgroup_t> _crs;

        double virtual_move_edge(size_t j, size_t i, value_t y)
        {
            auto& bins_j = *_bins[j];
            auto x = bins_j[i];

            _rs.clear();
            get_rs(get_mgroup(j, x, true), _rs);
            if (i > 0)
                get_rs(get_mgroup(j, bins_j[i-1], true), _rs);

            auto S_terms =
                [&]()
                {
                    double S = 0;
                    for (auto& r : _rs)
                        S += entropy_group(r, get_hist(r));

                    if (j >= _conditional)
                    {
                        _crs.clear();
                        for (auto& r : _rs)
                            _crs.insert(to_cgroup(r));
                        auto Mx = get_Mx();
                        for (auto& cr : _crs)
                            S += entropy_cgroup(get_chist(cr), Mx);
                    }

                    if (i == 0 || i == bins_j.size() - 1)
                    {
                        size_t Md = bins_j.size() - 1;
                        auto delta = *(bins_j.end()-1) - *bins_j.begin();
                        if (_discrete[j])
                            S += lbinom(delta - 1, value_t(Md - 1)) + _alpha * log(delta);
                        else
                            S += (Md + _alpha + 1) * log(delta);
                    }
                    return S;
                };

            double Sb = S_terms();

            move_edge(j, i, y);

            _rs.clear();
            get_rs(get_mgroup(j, y, true), _rs);
            if (i > 0)
                get_rs(get_mgroup(j, bins_j[i-1], true), _rs);

            double Sa = S_terms();

            move_edge(j, i, x);

            return Sa - Sb;
        }

        template <bool Add>
        double virtual_change_edge(size_t j, size_t i, value_t y)
        {
            auto& bins_j = *_bins[j];

            auto x = bins_j[i];

            if constexpr (!Add)
                y = bins_j[i-1];

            _rs.clear();
            get_rs(get_mgroup(j, x, true), _rs);
            if constexpr (!Add)
                get_rs(get_mgroup(j, y, true), _rs);

            double M = (_conditional >= _D) ? get_M() : 0;
            size_t Md = bins_j.size() - 1;
            auto delta = *(bins_j.end()-1) - *bins_j.begin();

            auto S_terms =
                [&]()
                {
                    double S = 0;
                    for (auto& r : _rs)
                        S += entropy_group(r, get_hist(r));

                    if (_discrete[j])
                        S += lbinom(delta - 1, value_t(Md - 1));
                    else
                        S += (Md + _alpha + 1) * log(delta) + lgamma(Md);

                    if (_conditional < _D)
                    {
                        auto Mx = get_Mx();
                        if (j < _conditional)
                        {
                            for (auto& [cr, n] : _chist)
                                S += entropy_cgroup(n, Mx);
                        }
                        else
                        {
                            _crs.clear();
                            for (auto& r : _rs)
                                _crs.insert(to_cgroup(r));
                            for (auto& cr : _crs)
                                S += entropy_cgroup(get_chist(cr), Mx);
                        }
                    }
                    else
                    {
                        S += lgamma(_N + M) - lgamma(M);
                    }

                    return S;
                };

            double Sb = S_terms();

            if constexpr (Add)
                add_edge(j, i, y);
            else
                remove_edge(j, i);

            _rs.clear();
            if constexpr (Add)
            {
                get_rs(get_mgroup(j, x, true), _rs);
                get_rs(get_mgroup(j, y, true), _rs);
            }
            else
            {
                get_rs(get_mgroup(j, y, true), _rs);
            }

            M /= Md * _pcount;
            Md = bins_j.size() - 1;
            M *= Md * _pcount;

            double Sa = S_terms();

            if constexpr (Add)
                remove_edge(j, i + 1);
            else
                add_edge(j, i - 1, x);

            return Sa - Sb;
        }

        template <class V>
        void check_bounds(size_t i, V&& xn, bool move_edges)
        {
            if (!_bounds.empty())
            {
                for (size_t j = 0; j < _D; ++j)
                {
                    if (_categorical[j])
                        continue;
                    if (_x[i][j] == _bounds[j].first ||
                        _x[i][j] == _bounds[j].second ||
                        xn[j] <= _bounds[j].first ||
                        xn[j] >= _bounds[j].second)
                    {
                        _bounds.clear();
                        break;
                    }
                }
            }

            if (move_edges)
            {
                for (size_t j = 0; j < _D; ++j)
                {
                    if (_categorical[j])
                        continue;
                    auto& bins_j = *_bins[j];
                    if (*bins_j.begin() > xn[j])
                        move_edge(j, 0, xn[j]);
                    if (*bins_j.rbegin() <= xn[j])
                    {
                        if (_discrete[j])
                            move_edge(j, bins_j.size() - 1, xn[j] + 1);
                        else
                            move_edge(j, bins_j.size() - 1,
                                      std::nextafter(xn[j], numeric_limits<value_t>::max()));
                    }
                }
            }

        }

        template <class V>
        void replace_point(size_t i, V&& xn, size_t w = 1, bool move_edges = false)
        {
            check_bounds(i, xn, move_edges);

            update_hist<false, false>(i);
            for (size_t j = 0; j < _D; ++j)
                _x[i][j] = xn[j];
            if (!_w.empty())
                _w[i] = w;
            update_hist<true, false>(i);
        }

        template <class V>
        double virtual_replace_point_dS(size_t i, V&& xn)
        {
            bool outbounds = false;
            for (size_t j = 0; j < _D; ++j)
            {
                if (_categorical[j])
                    continue;
                auto& bins_j = *_bins[j];
                if (xn[j] < *bins_j.begin() || xn[j] >= *bins_j.rbegin())
                {
                    if (j < _conditional)
                        return numeric_limits<double>::infinity();
                    else
                        outbounds = true;
                }
            }

            group_t r = get_bin(_x[i]);
            group_t nr = get_bin(xn);
            size_t w = _w.empty() ? 1 : _w[i];

            if (r == nr && !outbounds)
                return 0.0;

            size_t count_r = get_hist(r);
            size_t count_nr = outbounds ? 0 : get_hist(nr);

            auto lwr = get_lw(r);
            auto lwnr = get_lw(nr);
            double Sb = entropy_group(count_r, lwr) + entropy_group(count_nr, lwnr);
            double Sa = entropy_group(count_r - w, lwr) + entropy_group(count_nr + w, lwnr);

            if (_conditional < _D)
            {
                cgroup_t cr = to_cgroup(r);
                cgroup_t cnr = to_cgroup(nr);
                if (cr != cnr && !outbounds)
                {
                    auto Mx = get_Mx();
                    size_t count_cr = get_chist(cr);
                    size_t count_cnr = outbounds ? 0 : get_chist(cnr);
                    Sb += entropy_cgroup(count_cr, Mx) + entropy_cgroup(count_cnr, Mx);
                    Sa += entropy_cgroup(count_cr - w, Mx) + entropy_cgroup(count_cnr + w, Mx);
                }
            }

            return Sa - Sb;
        }

        template <bool Add, bool move_edges=false, class V=int>
        void modify_point(size_t i, V&& xn=0, size_t w = 1)
        {
            if constexpr (Add)
            {
                if (i >= _x.shape()[0])
                {
                    size_t N = _x.shape()[0];
                    _x.resize(boost::extents[(i + 1) * 2][_D]);
                    for (size_t m = N; m < _x.shape()[0]; ++m)
                    {
                        for (size_t j = 0; j < _D; ++j)
                            set_invalid(_x[m][j]);
                    }
                }

                bool was_empty = _w.empty();
                if (w != 1 || !was_empty)
                {
                    if (i >= _w.size())
                        _w.resize((i + 1) * 2);
                    if (was_empty)
                    {
                        for (size_t m = 0; m < i; ++m)
                            _w[m] = 1;
                    }
                }

                check_bounds(i, xn, move_edges);

                for (size_t j = 0; j < _D; ++j)
                    _x[i][j] = xn[j];
                if (!_w.empty())
                    _w[i] = w;
                update_hist<true, false>(i);
            }
            else
            {
                update_hist<false, false>(i);
                for (size_t j = 0; j < _D; ++j)
                    set_invalid(_x[i][j]);
            }
        }

        template <bool Add, class V=int>
        double virtual_modify_point_dS(size_t i, V&& xn = 0, int w = 1)
        {
            group_t r;
            size_t count_r;
            bool outbounds = false;
            if constexpr (Add)
            {
                for (size_t j = 0; j < _D; ++j)
                {
                    if (_categorical[j])
                        continue;
                    auto& bins_j = *_bins[j];
                    if (xn[j] < *bins_j.begin() || xn[j] >= *bins_j.rbegin())
                    {
                        if (j < _conditional)
                            return numeric_limits<double>::infinity();
                        else
                            outbounds = true;

                    }
                }
                r = get_bin(xn);
                count_r = outbounds ? 0 : get_hist(r);
            }
            else
            {
                r = get_bin(_x[i]);
                count_r = get_hist(r);
            }

            w = _w.empty() ? 1 : (Add ? w : _w[i]);

            int diff = Add ? w : -w;

            double Sb = _D * safelog(_N);
            double Sa = _D * safelog(_N + diff);

            if (_conditional >= _D)
            {
                double M = get_M();
                Sb += lgamma(_N + M);
                Sa += lgamma(_N + M + diff);
            }

            assert(int(count_r) + diff >= 0);
            auto lw = get_lw(r);
            Sb += entropy_group(count_r, lw);
            Sa += entropy_group(count_r + diff, lw);

            if (_conditional < _D)
            {
                auto Mx = get_Mx();
                size_t count_cr = outbounds ? 0 : get_chist(to_cgroup(r));
                Sb += entropy_cgroup(count_cr, Mx);
                Sa += entropy_cgroup(count_cr + diff, Mx);
            }

            return Sa - Sb;
        }

        void trim_points()
        {
            std::vector<std::vector<value_t>> x;
            std::vector<size_t> ws;
            for (size_t m = 0; m < _x.shape()[0]; ++m)
            {
                if (!is_valid(m))
                    continue;
                x.emplace_back();
                for (size_t j = 0; j < _D; ++j)
                    x.back().push_back(_x[m][j]);
                if (!_w.empty())
                    ws.push_back(_w[m]);
            }
            _x.resize(boost::extents[x.size()][_x.shape()[1]]);
            if (!_w.empty())
                _w.resize(x.size());
            for (size_t m = 0; m < _x.shape()[0]; ++m)
            {
                for (size_t j = 0; j < _D; ++j)
                    _x[m][j] = x[m][j];
                if (!_w.empty())
                    _w[m] = ws[m];
                update_hist<true, false>(m);
            }
        }

        void update_bounds()
        {
            if (_bounds.empty())
            {
                _bounds.resize(_D, {std::numeric_limits<value_t>::max(),
                                    std::numeric_limits<double>::lowest()});
                for (size_t i = 0; i < _x.shape()[0]; ++i)
                {
                    if (!is_valid(i))
                        continue;

                    for (size_t j = 0; j < _D; ++j)
                    {
                        _bounds[j].first = std::min(_bounds[j].first, _x[i][j]);
                        _bounds[j].second = std::max(_bounds[j].second, _x[i][j]);
                    }
                }
            }
        }

        void check_bins()
        {
#ifndef NDEBUG
            size_t N = 0;
            for (size_t i = 0; i < _x.shape()[0]; ++i)
            {
                auto x = _x[i];
                if (!is_valid(i))
                    continue;
                for (size_t j = 0; j < _D; ++j)
                {
                    if (_categorical[j])
                        continue;

                    auto& bins = *_bins[j];
                    assert(x[j] >= *bins.begin());
                    assert(x[j] < *bins.rbegin());
                }
                auto w = _w.empty() ? 1 : _w[i];
                N += w;
            }
            assert(N == _N);
#endif
        }

        // sampling and querying

        template <class V>
        double get_lpdf(const V& x, bool mle)
        {
            for (size_t j = 0; j < _D; ++j)
            {
                if (_categorical[j])
                    continue;
                auto& bin = *_bins[j];
                if (x[j] < *bin.begin() || x[j] >= *bin.rbegin())
                    return -numeric_limits<double>::infinity();
            }

            auto r = get_bin(x);

            double lw = 0;
            double M = 1, Mp = 1;
            for (size_t j = 0; j < _conditional; ++j)
            {
                auto& bin = *_bins[j];
                auto iter = std::lower_bound(bin.begin(),
                                             bin.end(), r[j]);
                if (iter == bin.end() || iter == bin.end() - 1)
                    return -numeric_limits<double>::infinity();
                lw += log(*(iter+1) - *iter);
                M *= (bin.size() - 1);
                Mp *= (bin.size() - 1) * _pcount;
            }

            double L = -lw + log(get_hist(r) + _pcount - int(mle));

            if (_conditional >= _D)
            {
                L -= log(_N + Mp - M * int(mle));
            }
            else
            {
                auto n = get_chist(to_cgroup(r));
                if (n == 0)
                    return numeric_limits<double>::quiet_NaN();
                L -= log(n + Mp - M * int(mle));
            }

            return L;
        }

        template <class V>
        double get_cond_mean(V x, size_t mean_j, bool mle)
        {
            for (size_t j = 0; j < _D; ++j)
            {
                if (_categorical[j] || j == mean_j)
                    continue;
                auto& bin = *_bins[j];
                if (x[j] < *bin.begin() || x[j] >= *bin.rbegin())
                    return numeric_limits<double>::quiet_NaN();
            }

            auto& bins = *_bins[mean_j];

            double a = 0;
            size_t N = 0;
            for (size_t i = 0; i < bins.size() - 1; ++i)
            {
                auto w = bins[i + 1] - bins[i];
                x[mean_j] = bins[i];
                auto r = get_bin(x);
                auto n = get_hist(r) + _pcount - int(mle);
                a += (bins[i] + w/2.) * n;
                N += n;
            }
            return a / N;
        }

        template <class RNG>
        multi_array<value_t, 2> sample(size_t n, multi_array_ref<value_t, 1> cx,
                                       RNG& rng)
        {
            multi_array<value_t, 2> x(extents[n][_conditional]);

            std::vector<value_t> y;
            y.push_back(*_bins[0]->begin());
            y.insert(y.end(), cx.begin(), cx.end());

            auto cr = to_cgroup(get_bin(y));

            std::vector<group_t> nrs;
            std::vector<double> counts;

            for (auto& [r, count] : _hist)
            {
                if (to_cgroup(r) != cr)
                    continue;
                nrs.emplace_back(r);
                counts.push_back(count);
            };

            Sampler<group_t> idx_sampler(nrs, counts);

            for (size_t i = 0; i < n; ++i)
            {
                auto& r = idx_sampler.sample(rng);
                for (size_t j = 0; j < _conditional; ++j)
                {
                    auto& bin = *_bins[j];
                    auto iter = std::lower_bound(bin.begin(),
                                                 bin.end(), r[j]);
                    if (_discrete[j])
                    {
                        std::uniform_int_distribution<int64_t> d(*iter, *(iter+1)-1);
                        x[i][j] = d(rng);
                    }
                    else
                    {
                        std::uniform_real_distribution<double> d(*iter, *(iter+1));
                        x[i][j] = d(rng);
                    }
                }
            }

            return x;
        }

        x_t& get_x()
        {
            return _x;
        }

        w_t& get_w()
        {
            return _w;
        }

        bool empty()
        {
            return _N == 0;
        }
    };
};
} // graph_tool namespace

#endif //GRAPH_HISTOGRAM_HH
