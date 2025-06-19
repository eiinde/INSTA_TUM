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

#ifndef GRAPH_AVG_CORR_HH
#define GRAPH_AVG_CORR_HH

#include "graph_correlations.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

// retrieves the generalized correlation
template <class GetDegreePair>
struct get_avg_correlation
{
    get_avg_correlation(python::object& avg, python::object& dev,
                        const vector<long double>& bins,
                        python::object& ret_bins)
        : _avg(avg), _dev(dev), _bins(bins), _ret_bins(ret_bins) {}

    template <class Graph, class DegreeSelector1, class DegreeSelector2,
              class WeightMap>
    void operator()(Graph& g, DegreeSelector1 deg1, DegreeSelector2 deg2,
                    WeightMap weight) const
    {
        GILRelease gil_release;

        GetDegreePair put_point;

        typedef typename DegreeSelector1::value_type type1;
        typedef typename DegreeSelector2::value_type type2;

        typedef typename graph_tool::detail::
            select_float_and_larger::apply<type2,double>::type
            avg_type;
        typedef type1 val_type;

        typedef typename property_traits<WeightMap>::value_type count_type;
        typedef Histogram<type1,count_type,1> count_t;
        typedef Histogram<val_type,avg_type,1> sum_t;

        std::array<vector<val_type>,1> bins;
        bins[0].resize(_bins.size());
        clean_bins(_bins, bins[0]);

        sum_t sum(bins);
        sum_t sum2(bins);
        count_t count(bins);

        SharedHistogram<sum_t> s_sum(sum);
        SharedHistogram<sum_t> s_sum2(sum2);
        SharedHistogram<count_t> s_count(count);

        #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh()) \
            firstprivate(s_sum, s_sum2, s_count)
        parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 put_point(v, deg1, deg2, g, weight, s_sum, s_sum2, s_count);
             });
        s_sum.gather();
        s_sum2.gather();
        s_count.gather();

        auto& asum = sum.get_array();
        auto& asum2 = sum2.get_array();
        auto& acount = count.get_array();
        for (size_t i = 0; i < asum.size(); ++i)
        {
            asum[i] /= acount[i];
            asum2[i] = sqrt(abs(asum2[i] / acount[i] -
                                asum[i] * asum[i])) / sqrt(acount[i]);
        }

        bins = sum.get_bins();

        gil_release.restore();
        python::list ret_bins;
        ret_bins.append(wrap_vector_owned(bins[0]));
        _ret_bins = ret_bins;
        _avg = wrap_multi_array_owned(sum.get_array());
        _dev = wrap_multi_array_owned(sum2.get_array());
    }
    python::object& _avg;
    python::object& _dev;
    const vector<long double>& _bins;
    python::object& _ret_bins;
};

} // graph_tool namespace

#endif
