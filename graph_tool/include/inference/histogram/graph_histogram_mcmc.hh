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

#ifndef GRAPH_HISTOGRAM_MCMC_HH
#define GRAPH_HISTOGRAM_MCMC_HH

#include "config.h"

#include <vector>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include <boost/mpl/vector.hpp>

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_HIST_STATE_params(State)                                          \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((verbose,, int, 0))                                                       \
    ((niter,, size_t, 0))

enum class hmove_t { move = 0, add, remove, null };

ostream& operator<<(ostream& s, hmove_t v);

template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCHistStateBase, MCMC_HIST_STATE_params(State))

    template <class... Ts>
    class MCMCHistState
        : public MCMCHistStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MCMCHistStateBase<Ts...>,
                         MCMC_HIST_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_HIST_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCHistState(ATs&&... as)
           : MCMCHistStateBase<Ts...>(as...)
        {
            _state.reset_mgroups();
            _state.update_bounds();

            for (size_t j = 0; j < _state._D; ++j)
            {
                if (_state._categorical[j])
                    continue;
                _dims.push_back(j);
            }
        }

        std::vector<size_t> _dims;

        constexpr static hmove_t _null_move = hmove_t::null;

        typedef typename State::value_t value_t;

        size_t node_state(size_t)
        {
            return 0;
        }

        constexpr bool skip_node(size_t)
        {
            return false;
        }

        size_t _i;
        size_t _j;
        value_t _x;

        constexpr static double _epsilon = 1e-8;
        constexpr static int64_t _max_int =
                      std::conditional<std::is_same_v<value_t, double>,
                                       std::integral_constant<int64_t, 2L << 51>,
                                       std::integral_constant<int64_t, std::numeric_limits<int64_t>::max()>>::type::value;
        constexpr static uint64_t _max_uint =
                      std::conditional<std::is_same_v<value_t, double>,
                                       std::integral_constant<uint64_t, 2uL << 51>,
                                       std::integral_constant<uint64_t, std::numeric_limits<uint64_t>::max()>>::type::value;

        template <class RNG>
        hmove_t move_proposal(size_t, RNG& rng)
        {
            _j = uniform_sample(_dims, rng);

            std::uniform_int_distribution<size_t>
                random_i(0, _state._bins[_j]->size()-1);
            _i = random_i(rng);

            size_t m;
            if (_i == _state._bins[_j]->size()-1)
            {
                m = 0;
            }
            else if (_i == 0)
            {
                std::uniform_int_distribution<size_t> random(0, 1);
                m = random(rng);
            }
            else
            {
                std::uniform_int_distribution<size_t> random(0, 2);
                m = random(rng);
            }

            hmove_t move = hmove_t::null;

            switch (m)
            {
            case 0:
                move = hmove_t::move;
                {
                    if (_i == 0)
                    {
                        if (_state._bounded[_j].first)
                        {
                            move = hmove_t::null;
                        }
                        else
                        {
                            auto& bin = *_state._bins[_j];
                            if (_state._discrete[_j])
                            {
                                auto right = std::min(_state._bounds[_j].first, bin[1] - 1);
                                auto w = right - bin[0] + 1;
                                std::geometric_distribution<uint64_t> d(1./(2 * w));
                                int64_t step = std::min(d(rng), _max_uint);
                                _x = std::min(int64_t(right) - step - 1, _max_int);
                                _x = std::max(int64_t(_x), -_max_int);

                                assert(_x <= right);
                            }
                            else
                            {
                                double right = std::min(_state._bounds[_j].first, bin[1]);
                                auto w = std::max(right - bin[0], _epsilon);

                                std::exponential_distribution<double> d(1./(2 * w));
                                _x = right - d(rng);

                                assert(_x <= right);
                            }

                        }
                    }
                    else if (_i == _state._bins[_j]->size()-1)
                    {
                        if (_state._bounded[_j].second)
                        {
                            move = hmove_t::null;
                        }
                        else
                        {
                            auto& bin = *_state._bins[_j];
                            if (_state._discrete[_j])
                            {
                                auto left = std::max(_state._bounds[_j].second,
                                                     bin[bin.size()-2]);
                                auto w = bin[bin.size()-1] - left + 1;
                                std::geometric_distribution<uint64_t> d(1./(2 * w));
                                int64_t step = std::min(d(rng), _max_uint);
                                _x = std::min(int64_t(left) + step + 1, _max_int);
                                _x = std::max(int64_t(_x), int64_t(left));
                                assert(_x > left);
                            }
                            else
                            {
                                double left = std::max(_state._bounds[_j].second ,
                                                       bin[bin.size()-2]);
                                auto w = bin[bin.size()-1] - left;
                                w = std::max(double(w), _epsilon);
                                std::exponential_distribution<double> d(1./(2 * w));
                                _x = left + d(rng);
                                if (_x == left)
                                    move = hmove_t::null;
                                // if (_x <= _state._bounds[_j].second ||
                                //     _x <= *(_state._bins[_j]->end() - 2))
                                //     move = hmove_t::null;
                                assert(_x > left || move == hmove_t::null);
                            }
                        }
                    }
                    else
                    {
                        if (_state._discrete[_j])
                        {
                            std::uniform_int_distribution<int64_t>
                                random_x((*_state._bins[_j])[_i-1]+1,
                                         (*_state._bins[_j])[_i+1]-1);
                            _x = random_x(rng);
                        }
                        else
                        {
                            std::uniform_real_distribution<double>
                                random_x((*_state._bins[_j])[_i-1],
                                         (*_state._bins[_j])[_i+1]);
                            _x = random_x(rng);
                            if (_x <= (*_state._bins[_j])[_i-1] ||
                                _x >= (*_state._bins[_j])[_i+1])
                                move = hmove_t::null;
                        }
                    }
                }
                break;
            case 1:
                move = hmove_t::add;
                {
                    if (_state._discrete[_j])
                    {
                        auto a = (*_state._bins[_j])[_i] + 1;
                        auto b = (*_state._bins[_j])[_i+1] - 1;
                        if (b < a)
                        {
                            move = hmove_t::null;
                        }
                        else
                        {
                            std::uniform_int_distribution<int64_t> random_x(a, b);
                            _x = random_x(rng);
                        }
                    }
                    else
                    {
                        std::uniform_real_distribution<double>
                            random_x((*_state._bins[_j])[_i],
                                     (*_state._bins[_j])[_i+1]);
                        _x = random_x(rng);
                        if (_x <= (*_state._bins[_j])[_i] ||
                            _x >= (*_state._bins[_j])[_i+1])
                            move = hmove_t::null;
                    }
                }
                break;
            case 2:
                move = hmove_t::remove;
                break;
            default:
                break;
            }

            return move;
        }

        std::tuple<double, double>
        virtual_move_dS(size_t, hmove_t move)
        {
            double dS = 0;
            double pf = 0;
            double pb = 0;

            switch (move)
            {
            case hmove_t::move:
                dS = _state.virtual_move_edge(_j, _i, _x);

                if (_i == 0)
                {
                    auto& bin = *_state._bins[_j];

                    if (_state._discrete[_j])
                    {
                        auto right = std::min(_state._bounds[_j].first, bin[1] - 1);
                        auto w = right - bin[0] + 1;
                        auto nw = right - _x + 1;

                        auto delta_f = right - _x;
                        auto delta_b = right - bin[0];

                        double p = 1./(2 * w);
                        pf = log1p(-p) * delta_f + log(p);
                        p = 1./(2 * nw);
                        pb = log1p(-p) * delta_b + log(p);
                    }
                    else
                    {
                        double right = std::min(_state._bounds[_j].first, bin[1]);
                        auto w = std::max(right - bin[0], _epsilon);
                        auto nw = std::max(right - _x, _epsilon);

                        double lf = 1./(2 * w);
                        double lb = 1./(2 * nw);
                        double delta_f = right - _x;
                        double delta_b = right - bin[0];
                        pf = -lf * delta_f - log(lf);
                        pb = -lb * delta_b - log(lb);
                    }
                }
                else if (_i == _state._bins[_j]->size()-1)
                {
                    auto& bin = *_state._bins[_j];

                    if (_state._discrete[_j])
                    {
                        auto left = std::max(_state._bounds[_j].second,
                                             bin[bin.size()-2]);
                        auto w = bin[bin.size()-1] - left + 1;
                        auto nw = _x - left + 1;

                        auto delta_f = _x - left;
                        auto delta_b = bin[bin.size()-1] - left;

                        double p = 1./(2 * w);
                        pf = log1p(-p) * delta_f + log(p);
                        p = 1./(2 * nw);
                        pb = log1p(-p) * delta_b + log(p);
                    }
                    else
                    {
                        double left = std::max(_state._bounds[_j].second ,
                                               bin[bin.size()-2]);
                        auto w = std::max(bin[bin.size()-1] - left, _epsilon);
                        auto nw = std::max(_x - left, _epsilon);

                        double lf = 1./(2 * w);
                        double lb = 1./(2 * nw);
                        double delta_f = _x - left;
                        double delta_b = bin[bin.size()-1] - left;
                        pf = -lf * delta_f - log(lf);
                        pb = -lb * delta_b - log(lb);
                    }
                }

                break;
            case hmove_t::add:
                dS = _state.template virtual_change_edge<true>(_j, _i, _x);
                pf = -safelog_fast(_state._bins[_j]->size() - 2);
                pb = -safelog_fast(_state._bins[_j]->size() - 1);
                break;
            case hmove_t::remove:
                dS = _state.template virtual_change_edge<false>(_j, _i, 0.);
                pf = -safelog_fast(_state._bins[_j]->size() - 2);
                pb = -safelog_fast(_state._bins[_j]->size() - 3);
                break;
            default:
                break;
            }

            if (_verbose)
                cout << "propose : " << int(move)
                     << " " << _j << " " << _i << " "
                     << dS << " " << pb - pf << endl;

            return std::make_tuple(dS, pb - pf);
        }

        void perform_move(size_t, hmove_t move)
        {
            if (_verbose)
                cout << "perform : " << int(move) << " " << _j << " "
                     << _i << " " << _x << endl;

            switch (move)
            {
            case hmove_t::move:
                _state.move_edge(_j, _i, _x);
                break;
            case hmove_t::add:
                _state.add_edge(_j, _i, _x);
                break;
            case hmove_t::remove:
                _state.remove_edge(_j, _i);
                break;
            default:
                break;
            }
        }

        bool is_deterministic()
        {
            return false;
        }

        bool is_sequential()
        {
            return false;
        }

        std::array<size_t,1> _vlist = {0};
        auto& get_vlist()
        {
            return _vlist;
        }

        double get_beta()
        {
            return _beta;
        }

        size_t get_niter()
        {
            return _niter;
        }

        void step(size_t, hmove_t)
        {
        }

        template <class RNG>
        void init_iter(RNG&)
        {
        }
    };
};


} // graph_tool namespace

#endif //GRAPH_HISTOGRAM_MCMC_HH
