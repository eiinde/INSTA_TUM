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

#ifndef GRAPH_BLOCKMODEL_WEIGHTS_HH
#define GRAPH_BLOCKMODEL_WEIGHTS_HH

namespace graph_tool
{

// Weighted entropy terms
// ======================

enum weight_type
{
    NONE,
    COUNT,
    REAL_EXPONENTIAL,
    REAL_NORMAL,
    DISCRETE_GEOMETRIC,
    DISCRETE_POISSON,
    DISCRETE_BINOMIAL
};

// exponential
template <class DT>
double positive_w_log_P(DT N, double x, double alpha, double beta,
                        double epsilon)
{
    if (N == 0)
        return 0.;
    if (std::isnan(alpha) && std::isnan(beta))
    {
        if (x < epsilon || N == 1)
            return 0.;
        else
            return lgamma(N) - (N - 1) * log(x);
    }
    return lgamma(N + alpha) - lgamma(alpha) + alpha * log(beta) -
        (alpha + N) * log(beta + x);
}

// normal
template <class DT>
double signed_w_log_P(DT N, double x, double x2, double m0, double k0, double v0,
                      double nu0, double epsilon)
{
    if (N == 0)
        return 0.;
    if (std::isnan(m0) && std::isnan(k0))
    {
        auto smu1 = x * (x / N);

        if (N < 2 || smu1 >= x2 || (x2 - smu1) < std::pow(epsilon, 2))
            return 0.;
        else
            return (lgamma((N - 1) / 2.) + log(N) / 2.
                    - ((int(N) - 3) / 2.) * log(x2 - smu1)
                    - ((N - 1) / 2.) * log(M_PI));
    }
    auto v = x2 - x * (x / N);
    auto k_n = k0 + N;
    auto nu_n = nu0 + N;
    auto v_n = (v0 * nu0 + v + ((N * k0)/(k0 + N)) * pow(m0 - x/N, 2)) / nu_n;
    return lgamma(nu_n / 2.) - lgamma(nu0 / 2.) + (log(k0) - log(k_n)) / 2. +
        (nu0 / 2.) * log(nu0 * v0) - (nu_n / 2.) * log(nu_n * v_n) - (N / 2.) *
        log(M_PI);
}

// discrete: geometric
template <class DT>
double geometric_w_log_P(DT N, double x, double alpha, double beta)
{
    if (N == 0)
        return 0.;
    if (std::isnan(alpha) && std::isnan(beta))
        return -lbinom((N - 1) + x, x);
    return lbeta(N + alpha, x + beta) - lbeta(alpha, beta);
}

// discrete: binomial
template <class DT>
double binomial_w_log_P(DT N, double x, int n, double alpha, double beta)
{
    if (N == 0)
        return 0.;
    if (std::isnan(alpha) && std::isnan(beta))
        return -lbinom(N * n, x);
    return lbeta(x + alpha, N * n - x + beta) - lbeta(alpha, beta);
}

// discrete: Poisson
template <class DT>
double poisson_w_log_P(DT N, double x, double alpha, double beta)
{
    if (N == 0)
        return 0.;
    if (std::isnan(alpha) && std::isnan(beta))
        return lgamma(x+1) - x * log(N);
    return lgamma(x + alpha) - (x + alpha) * log(N + beta) - lgamma(alpha) +
        alpha * log(beta);
}


template <class State>
std::tuple<double,double> rec_entropy(State& state, const entropy_args_t& ea)
{
    double S = 0, S_dl = 0;
    for (size_t i = 0; i < state._rec_types.size(); ++i)
    {
        auto& wp = state._wparams[i];
        switch (state._rec_types[i])
        {
        case weight_type::COUNT:
            break;
        case weight_type::REAL_EXPONENTIAL:
            for (auto me : edges_range(state._bg))
            {
                auto ers = state._brec[0][me];
                auto xrs = state._brec[i][me];
                S += -positive_w_log_P(ers, xrs, wp[0], wp[1],
                                       state._epsilon[i]);
            }
            if (ea.recs_dl && std::isnan(wp[0]) && std::isnan(wp[1]))
                S_dl += -positive_w_log_P(state._B_E, state._recsum[i], wp[0], wp[1],
                                          state._epsilon[i]);
            break;
        case weight_type::DISCRETE_GEOMETRIC:
            for (auto me : edges_range(state._bg))
            {
                auto ers = state._brec[0][me];
                auto xrs = state._brec[i][me];
                S += -geometric_w_log_P(ers, xrs, wp[0], wp[1]);
            }
            if (ea.recs_dl && std::isnan(wp[0]) && std::isnan(wp[1]))
                S += -geometric_w_log_P(state._B_E, state._recsum[i], wp[0], wp[1]);
            break;
        case weight_type::DISCRETE_POISSON:
            for (auto me : edges_range(state._bg))
            {
                auto ers = state._brec[0][me];
                auto xrs = state._brec[i][me];
                S += -poisson_w_log_P(ers, xrs, wp[0], wp[1]);
            }
            for (auto e : edges_range(state._g))
                S += lgamma(state._rec[i][e] + 1);
            if (ea.recs_dl && std::isnan(wp[0]) && std::isnan(wp[1]))
                S += -geometric_w_log_P(state._B_E, state._recsum[i], wp[0],
                                        wp[1]);
            break;
        case weight_type::DISCRETE_BINOMIAL:
            for (auto me : edges_range(state._bg))
            {
                auto ers = state._brec[0][me];
                auto xrs = state._brec[i][me];
                S += -binomial_w_log_P(ers, xrs, wp[0], wp[1], wp[2]);
            }
            for (auto e : edges_range(state._g))
                S -= lbinom(wp[0], state._rec[i][e]);
            if (ea.recs_dl && std::isnan(wp[1]) && std::isnan(wp[2]))
                S += -geometric_w_log_P(state._B_E, state._recsum[i], wp[1],
                                        wp[2]);
            break;
        case weight_type::REAL_NORMAL:
            for (auto me : edges_range(state._bg))
            {
                auto ers = state._brec[0][me];
                auto xrs = state._brec[i][me];
                auto x2rs = state._bdrec[i][me];
                S += -signed_w_log_P(ers, xrs, x2rs, wp[0], wp[1],
                                     wp[2], wp[3], state._epsilon[i]);
            }
            if (std::isnan(wp[0]) && std::isnan(wp[1]))
            {
                if (ea.recs_dl)
                    S_dl += -signed_w_log_P(state._B_E, state._recsum[i],
                                            state._recx2[i], wp[0], wp[1],
                                            wp[2], wp[3], state._epsilon[i]);
                S += -positive_w_log_P(state._B_E_D, state._recdx[i], wp[2],
                                       wp[3], state._epsilon[i]);
            }
            break;
        }
    }
    return std::make_tuple(S, S_dl);
}

template <class State, class MEntries>
std::tuple<double, double> rec_entries_dS(State& state, MEntries& m_entries,
                                          const entropy_args_t& ea,
                                          std::vector<double>& dBdx, int& dL)
{
    int ddL = 0;
    double dS = 0, dS_dl = 0;
    auto positive_entries_op =
        [&](size_t i, auto&& w_log_P, auto&& w_log_prior)
        {
            int dB_E = 0;
            wentries_op(m_entries, state._emat,
                        [&](auto, auto, auto& me, auto delta, auto& edelta)
                        {
                            double ers = 0;
                            double xrs = 0;
                            if (me != state._emat.get_null_edge())
                            {
                                ers = state._brec[0][me];
                                xrs = state._brec[i][me];
                            }
                            assert(get<0>(edelta).size() > i);
                            auto d = get<0>(edelta)[0];
                            auto dx = get<0>(edelta)[i];
                            dS -= -w_log_P(ers, xrs);
                            dS += -w_log_P(ers + d, xrs + dx);

                            if (ea.recs_dl)
                            {
                                size_t ers = 0;
                                if (me != state._emat.get_null_edge())
                                    ers = state._mrs[me];
                                if (ers == 0 && delta > 0)
                                    dB_E++;
                                if (ers > 0 && ers + delta == 0)
                                    dB_E--;
                            }
                        });
            if (dB_E != 0 && ea.recs_dl && std::isnan(state._wparams[i][0])
                && std::isnan(state._wparams[i][1]))
            {
                dS_dl -= -w_log_prior(state._B_E);
                dS_dl += -w_log_prior(state._B_E + dB_E);
            }
        };

    for (size_t i = 0; i < state._rec_types.size(); ++i)
    {
        auto& wp = state._wparams[i];
        switch (state._rec_types[i])
        {
        case weight_type::COUNT:
            break;
        case weight_type::REAL_EXPONENTIAL:
            positive_entries_op(i,
                                [&](auto N, auto x)
                                { return positive_w_log_P(N, x, wp[0],
                                                          wp[1],
                                                          state._epsilon[i]);
                                },
                                [&](size_t B_E)
                                { return positive_w_log_P(B_E,
                                                          state._recsum[i],
                                                          wp[0], wp[1],
                                                          state._epsilon[i]);
                                });
            break;
        case weight_type::DISCRETE_GEOMETRIC:
            positive_entries_op(i,
                                [&](auto N, auto x)
                                { return geometric_w_log_P(N, x, wp[0],
                                                           wp[1]);
                                },
                                [&](size_t B_E)
                                { return geometric_w_log_P(B_E,
                                                           state._recsum[i],
                                                           wp[0],
                                                           wp[1]);
                                });
            break;
        case weight_type::DISCRETE_POISSON:
            positive_entries_op(i,
                                [&](auto N, auto x)
                                { return poisson_w_log_P(N, x, wp[0],
                                                         wp[1]);
                                },
                                [&](size_t B_E)
                                { return geometric_w_log_P(B_E,
                                                           state._recsum[i],
                                                           wp[0],
                                                           wp[1]);
                                });
            break;
        case weight_type::DISCRETE_BINOMIAL:
            positive_entries_op(i,
                                [&](auto N, auto x)
                                { return binomial_w_log_P(N, x, wp[0],
                                                          wp[1], wp[2]);
                                },
                                [&](size_t B_E)
                                { return geometric_w_log_P(B_E,
                                                           state._recsum[i],
                                                           wp[1],
                                                           wp[2]);
                                });
            break;
        case weight_type::REAL_NORMAL:
            {
                int dB_E = 0;
                int dB_E_D = 0;
                double dBx2 = 0;
                state._dBdx[i] = 0;
                wentries_op(m_entries, state._emat,
                            [&](auto, auto, auto& me, auto, auto& edelta)
                            {
                                double ers = 0;
                                double xrs = 0, x2rs = 0;
                                if (me != state._emat.get_null_edge())
                                {
                                    ers = state._brec[0][me];
                                    xrs = state._brec[i][me];
                                    x2rs = state._bdrec[i][me];
                                }
                                auto d = get<0>(edelta)[0];
                                auto dx = get<0>(edelta)[i];
                                auto dx2 = get<1>(edelta)[i];
                                dS -= -signed_w_log_P(ers, xrs, x2rs,
                                                      wp[0], wp[1],
                                                      wp[2], wp[3],
                                                      state._epsilon[i]);
                                dS += -signed_w_log_P(ers + d,
                                                      xrs + dx,
                                                      x2rs + dx2,
                                                      wp[0], wp[1],
                                                      wp[2], wp[3],
                                                      state._epsilon[i]);

                                if (std::isnan(wp[0]) && std::isnan(wp[1]))
                                {
                                    auto n_ers = ers + get<0>(edelta)[0];
                                    if (ers == 0 && n_ers > 0)
                                        dB_E++;
                                    if (ers > 0 && n_ers == 0)
                                        dB_E--;
                                    if (n_ers > 1)
                                    {
                                        if (ers < 2)
                                            dB_E_D++;
                                        state._dBdx[i] += (x2rs + dx2 -
                                                           std::pow(xrs + dx, 2) / n_ers);
                                    }
                                    if (ers > 1)
                                    {
                                        if (n_ers < 2)
                                            dB_E_D--;
                                        state._dBdx[i] -= (x2rs - std::pow(xrs, 2) / ers);
                                    }
                                    dBx2 += (std::pow(xrs + dx, 2) -
                                             std::pow(xrs, 2));
                                }
                            });

                if (std::isnan(wp[0]) && std::isnan(wp[1]))
                {
                    if (ea.recs_dl && (dB_E != 0 || dBx2 != 0))
                    {
                        dS_dl -= -signed_w_log_P(state._B_E, state._recsum[i],
                                                 state._recx2[i], wp[0], wp[1],
                                                 wp[2], wp[3],
                                                 state._epsilon[i]);
                        dS_dl += -signed_w_log_P(state._B_E + dB_E,
                                                 state._recsum[i],
                                                 state._recx2[i] + dBx2, wp[0],
                                                 wp[1], wp[2], wp[3],
                                                 state._epsilon[i]);
                    }

                    if (dB_E_D != 0 || state._dBdx[i] != 0)
                    {
                        dS -= -positive_w_log_P(state._B_E_D, state._recdx[i],
                                                wp[2], wp[3],
                                                state._epsilon[i]);
                        dS += -positive_w_log_P(state._B_E_D + dB_E_D,
                                                state._recdx[i] + state._dBdx[i],
                                                wp[2], wp[3],
                                                state._epsilon[i]);
                    }

                    if (ddL == 0)
                    {
                        if (state._B_E_D == 0 && dB_E_D > 0)
                            ddL++;
                        if (state._B_E_D > 0 && state._B_E_D + dB_E_D == 0)
                            ddL--;
                        dL += ddL;
                    }

                    if (state._Lrecdx[0] >= 0)
                    {
                        size_t N_B_E_D = state._B_E_D + dB_E_D;

                        dS_dl -= -safelog_fast(state._B_E_D);
                        dS_dl += -safelog_fast(N_B_E_D);

                        dBdx[i] += state._recdx[i] * dB_E_D +
                            state._dBdx[i] * N_B_E_D;

                        if (state._coupled_state == nullptr)
                        {
                            size_t L = state._Lrecdx[0];
                            dS_dl -= -positive_w_log_P(L, state._Lrecdx[i+1],
                                                       wp[2], wp[3],
                                                       state._epsilon[i]);
                            dS_dl += -positive_w_log_P(L + dL,
                                                       state._Lrecdx[i+1] + dBdx[i],
                                                       wp[2], wp[3],
                                                       state._epsilon[i]);
                        }
                    }
                }
            }
            break;
        }
    }
    return std::make_tuple(dS, dS_dl);
}

template <class State, class Edge, class MEntries>
void recs_propagate_insert(State& state, size_t r, size_t s, Edge& e, int d,
                           std::vector<double> dx, MEntries& m_entries)
{
    assert(dx.size() == state._rec.size());
    auto dx2 = dx;
    if (e == state._emat.get_null_edge())
    {
        dx[0] = (d > 0) ? 1 : 0;
        for (size_t i = 0; i < state._rec_types.size(); ++i)
            dx2[i] = std::pow(dx[i], 2);
    }
    else
    {
        for (size_t i = 0; i < state._rec_types.size(); ++i)
        {
            auto x = state._rec[i][e];
            dx2[i] = (std::pow(x + dx[i], 2) -
                      std::pow(x, 2));
        }

        int ers = state._eweight[e];
        if (ers == 0 && d > 0)
        {
            dx[0] = 1;
        }
        else
        {
            if (ers > 0 && ers + d == 0)
                dx[0] = -1;
            else
                dx[0] = 0;
        }
    }
    m_entries.template insert_delta<true>(r, s, d, dx, dx2);
}


template <bool Add, bool Remove, class State, class MEntries, class EOps>
void recs_apply_delta(State& state, MEntries& m_entries, EOps&& eops)
{
    auto skip =
        [&](auto delta, auto& edelta)
        {
            if (delta != 0)
                return false;
            if (get<0>(edelta).empty())
                return true;
            for (size_t i = 0; i < state._rec_types.size(); ++i)
            {
                if (get<0>(edelta)[i] != 0)
                    return false;
                if (state._rec_types[i] == weight_type::REAL_NORMAL &&
                    get<1>(edelta)[i] != 0)
                    return false;
            }
            return true;
        };

    if (state._coupled_state != nullptr)
    {
        m_entries._p_entries.clear();
        wentries_op(m_entries, state._emat,
                    [&](auto r, auto s, auto& me, auto delta, auto& edelta)
                    {
                        if (skip(delta, edelta))
                            return;
                        m_entries._p_entries.emplace_back(r, s, me, delta,
                                                          get<0>(edelta));
                    });

        if (!m_entries._p_entries.empty())
        {
            state._coupled_state->propagate_delta(m_entries.get_move().first,
                                                  m_entries.get_move().second,
                                                  m_entries._p_entries);
        }
    }

    auto end_op =
        [&](auto& me, auto& edelta)
        {
            for (size_t i = 0; i < state._rec_types.size(); ++i)
            {
                state._brec[i][me] += get<0>(edelta)[i];
                if (state._rec_types[i] == weight_type::REAL_NORMAL)
                    state._bdrec[i][me] += get<1>(edelta)[i];
            }
        };

    auto mid_op_BE =
        [&](auto& me, auto& edelta)
        {
            auto mrs = state._brec[0][me];
            if (Add && mrs == 0 && (mrs + get<0>(edelta)[0]) > 0)
            {
                state._B_E++;
                if (state._coupled_state != nullptr)
                    state._coupled_state->add_edge_rec(me);
            }

            if (Remove && mrs > 0 && (mrs + get<0>(edelta)[0]) == 0)
            {
                state._B_E--;
                if (state._coupled_state != nullptr)
                    state._coupled_state->remove_edge_rec(me);
            }
        };

    if (state._rt != weight_type::REAL_NORMAL)
    {
        eops([](auto&&... args) { wentries_op(args...);},
             mid_op_BE, end_op, skip);
    }
    else
    {
        auto mid_op =
            [&](auto& me, auto& edelta)
            {
                auto& mrs = state._brec[0][me];
                mid_op_BE(me, edelta);

                auto n_mrs = mrs + get<0>(edelta)[0];

                if (n_mrs > 1)
                {
                    if (Add && mrs < 2)
                    {
                        if (state._B_E_D == 0 && state._Lrecdx[0] >= 0)
                            state._Lrecdx[0] += 1;
                        state._B_E_D++;
                    }

                    for (size_t i = 0; i < state._rec_types.size(); ++i)
                    {
                        if (state._rec_types[i] != weight_type::REAL_NORMAL)
                            continue;
                        auto dx = (state._bdrec[i][me] + get<1>(edelta)[i]
                                   - (std::pow((state._brec[i][me] +
                                                get<0>(edelta)[i]), 2) / n_mrs));
                        state._recdx[i] += dx;
                    }
                }

                if (mrs > 1)
                {
                    if (Remove && n_mrs < 2)
                    {
                        state._B_E_D--;
                        if (state._B_E_D == 0 && state._Lrecdx[0] >= 0)
                            state._Lrecdx[0] -= 1;
                    }

                    for (size_t i = 0; i < state._rec_types.size(); ++i)
                    {
                        if (state._rec_types[i] != weight_type::REAL_NORMAL)
                            continue;
                        auto dx = (state._bdrec[i][me] -
                                   std::pow(state._brec[i][me], 2) / mrs);
                        state._recdx[i] -= dx;
                    }
                }

                for (size_t i = 0; i < state._rec_types.size(); ++i)
                {
                    if (state._rec_types[i] == weight_type::REAL_NORMAL)
                    {
                        state._recx2[i] -= std::pow(state._brec[i][me], 2);
                        state._recx2[i] += std::pow(state._brec[i][me] +
                                                    get<0>(edelta)[i], 2);
                    }
                }

            };

        auto coupled_end_op =
            [&](auto& me, auto& edelta)
            {
                end_op(me, edelta);
                if (state._coupled_state != nullptr)
                    state._coupled_state->update_edge_rec(me, get<0>(edelta));
            };

        if (state._Lrecdx[0] >= 0)
        {
            for (size_t i = 0; i < state._rec_types.size(); ++i)
                state._Lrecdx[i+1] -= state._recdx[i] * state._B_E_D;
        }

        eops([](auto&&... args) { wentries_op(args...);},
             mid_op, coupled_end_op, skip);

        if (state._Lrecdx[0] >= 0)
        {
            for (size_t i = 0; i < state._rec_types.size(); ++i)
                state._Lrecdx[i+1] += state._recdx[i] * state._B_E_D;
        }
    }

    if (state._coupled_state != nullptr)
    {
        std::vector<double> dummy(state._rec_types.size(), 0);
        m_entries._p_entries.clear();
        wentries_op(m_entries, state._emat,
                    [&](auto r, auto s, auto& me, auto, auto&)
                    {
                        m_entries._p_entries.emplace_back(r, s, me, 0, dummy);
                    });

        if (!m_entries._p_entries.empty())
        {
            state._coupled_state->propagate_delta(m_entries.get_move().first,
                                                  m_entries.get_move().second,
                                                  m_entries._p_entries);
        }
    }

}

} //namespace graph_tool

#endif // GRAPH_BLOCKMODEL_WEIGHTS_HH
