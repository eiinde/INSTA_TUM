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

#ifndef SEGMENT_SAMPLER_HH
#define SEGMENT_SAMPLER_HH

#include "../../../generation/sampler.hh"

namespace graph_tool
{

class SegmentSampler
{
public:
    SegmentSampler(std::vector<double>& xs, std::vector<double>& ws)
        : _xs(xs), _ws(ws)
    {
        double M = *std::max_element(_ws.begin(), _ws.end());
        for (auto& w : _ws)
        {
            w -= M;
            _ews.push_back(exp(w));
        }

        _sampler = std::piecewise_linear_distribution<>(_xs.begin(), _xs.end(),
                                                        _ews.begin());

        _lZ = -numeric_limits<double>::infinity();
        for (size_t i = 0; i < _xs.size() - 1; ++i)
            _lZ = log_sum_exp(_lZ, lZi(i));
    }

    template <class RNG>
    double sample(RNG& rng)
    {
        if (_xs.size() == 1)
            return _xs[0];
        return _sampler(rng);
    }

    double lprob(double x)
    {
        if (x < _xs.front() || x >= _xs.back())
            return -numeric_limits<double>::infinity();

        if (_xs.size() == 1)
            return 0;

        auto iter = std::upper_bound(_xs.begin(), _xs.end(), x);
        --iter;
        size_t i = iter - _xs.begin();

        assert(i < _ws.size() - 1);

        if (_ws[i + 1] == _ws[i] || x == _xs[i])
            return _ws[i] - _lZ;

        double a = log(x - _xs[i]) - log(_xs[i + 1] - _xs[i]);

        return log_sum_exp(_ws[i + 1] + a,
                           _ws[i] + log1p(-exp(a))) - _lZ;
    }

    double lZi(size_t i)
    {
        if (_xs.size() == 1)
            return i == 0 ? 0 : -numeric_limits<double>::infinity();
        double dx = _xs[i + 1] - _xs[i];
        double lZi = log_sum_exp(_ws[i], _ws[i + 1]) - log(2) + log(dx);
        return lZi;
    }

    double lprob_int(double a, double b)
    {
        if (_xs.size() == 1)
            return (a < _xs[0] && _xs[0] < b) ? 0 : -numeric_limits<double>::infinity();

        double lp = -numeric_limits<double>::infinity();

        for (size_t i = 0; i < _xs.size() - 1; ++i)
        {
            if (a >= _xs[i + 1] || b < _xs[i])
                continue;
            double a_i = (_xs[i] < a && a < _xs[i + 1]) ? a : _xs[i];
            double b_i = (_xs[i] < b && b < _xs[i + 1]) ? b : _xs[i + 1];

            double w_a, w_b;
            double ldx = log(_xs[i + 1] - _xs[i]);
            if (_ws[i] < _ws[i+1])
            {
                double dw = _ws[i+1] + log1p(-exp(_ws[i] - _ws[i+1]));
                w_a = log_sum_exp(_ws[i], dw + log(a_i - _xs[i]) - ldx);
                w_b = log_sum_exp(_ws[i], dw + log(b_i - _xs[i]) - ldx);
            }
            else
            {
                double dw = _ws[i] + log1p(-exp(_ws[i+1] - _ws[i]));
                w_a = log_sum_exp(_ws[i+1], dw + log(_xs[i+1] - a_i) - ldx);
                w_b = log_sum_exp(_ws[i+1], dw + log(_xs[i+1] - b_i) - ldx);
            }

            double lp_i = log(b_i - a_i) + log_sum_exp(w_a, w_b) - log(2);

            lp = log_sum_exp(lp, lp_i);

            assert(!std::isnan(lp));
        }

        assert(!std::isnan(lp - _lZ));
        return lp - _lZ;
    }

private:
    std::vector<double> _xs;
    std::vector<double> _ws;
    std::vector<double> _ews;
    double _lZ;

    std::piecewise_linear_distribution<> _sampler;
};

}
#endif // SEGMENT_SAMPLER_HH
