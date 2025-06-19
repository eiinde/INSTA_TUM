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

#ifndef DYNAMICS_HH
#define DYNAMICS_HH

#include "config.h"

#include <vector>
#include <map>
#include <algorithm>
#include <mutex>
#include <shared_mutex>

#include "idx_map.hh"

#include "../../support/graph_state.hh"
#include "../../support/fibonacci_search.hh"
#include "../uncertain_util.hh"
#include "../../../generation/graph_knn.hh"
#include "segment_sampler.hh"

#include "dynamics_util.hh"

#include <boost/math/tools/minima.hpp>
#include <boost/math/special_functions/relative_difference.hpp>

namespace graph_tool
{
using namespace boost;
using namespace std;

struct dentropy_args_t:
        public uentropy_args_t
{
    dentropy_args_t(const uentropy_args_t& ea)
        : uentropy_args_t(ea) {}

    double alpha = 1;
    double delta = 1e-8;
    bool xdist = true;
    bool tdist = true;
    bool xdist_uniform = false;
    bool tdist_uniform = false;
    double xl1 = 0;
    double tl1 = 0;
    bool normal = false;
    double mu = 0;
    double sigma = 1;
};

typedef eprop_map_t<double>::type xmap_t;
typedef vprop_map_t<double>::type tmap_t;
typedef vprop_map_t<double>::type smap_t;

#define DYNAMICS_STATE_params                                                  \
    ((g, &, never_filtered_never_reversed, 1))                                 \
    ((x,, xmap_t, 0))                                                          \
    ((params,, python::dict, 0))                                               \
    ((theta,, tmap_t, 0))                                                      \
    ((xmin_bound,, double, 0))                                                 \
    ((xmax_bound,, double, 0))                                                 \
    ((tmin_bound,, double, 0))                                                 \
    ((tmax_bound,, double, 0))                                                 \
    ((nmax_extend,, double, 0))                                                \
    ((disable_xdist,, bool, 0))                                                \
    ((disable_tdist,, bool, 0))                                                \
    ((self_loops,, bool, 0))                                                   \
    ((max_m,, int, 0))

class DStateBase
{
public:

    virtual double get_edge_dS(size_t u, size_t v, double x, double nx) = 0;
    virtual double get_edges_dS(std::vector<size_t>& us, size_t v,
                                std::vector<double>& x,
                                std::vector<double>& nx) = 0;
    virtual double get_edges_dS(const std::array<size_t,2>& us, size_t v,
                                const std::array<double,2>& x,
                                const std::array<double,2>& nx) = 0;

    virtual double get_node_dS(size_t v, double dt) = 0;
    virtual double get_node_prob(size_t v) = 0;
    virtual double get_node_prob(size_t v, size_t n, size_t t, double s) = 0;

    virtual void update_edge(size_t u, size_t v, double x, double nx) = 0;
    virtual void update_edges(std::vector<size_t>& us, size_t v,
                              std::vector<double>& x,
                              std::vector<double>& nx) = 0;

    virtual double node_TE(size_t u, size_t v) = 0;
    virtual double node_MI(size_t u, size_t v) = 0;
    virtual double node_cov(size_t u, size_t v, bool, bool) = 0;

    virtual void set_params(boost::python::dict) {};
};

template <class BlockState>
struct Dynamics
{
    GEN_STATE_BASE(DynamicsStateBase, DYNAMICS_STATE_params)

    template <class... Ts>
    class DynamicsState
        : public DynamicsStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(DynamicsStateBase<Ts...>,
                         DYNAMICS_STATE_params)
        GET_PARAMS_TYPEDEF(Ts, DYNAMICS_STATE_params)

        typedef typename property_traits<x_t>::value_type xval_t;
        typedef typename property_traits<theta_t>::value_type tval_t;

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
        DynamicsState(BlockState& block_state, ATs&&... args)
            : DynamicsStateBase<Ts...>(std::forward<ATs>(args)...),
              _block_state(block_state),
              _dummy_hint(num_vertices(_u)),
              _xc(_x.get_checked()),
              _e_mutex(num_vertices(_u)),
              _v_mutex(num_vertices(_u))
        {
            _u_edges.resize(num_vertices(_u));
            for (auto v : vertices_range(_u))
            {
                for (auto e : out_edges_range(v, _u))
                {
                    if (!graph_tool::is_directed(_u) && v > target(e, _u))
                        continue;

                    get_u_edge<true>(v, target(e, _u)) = e;
                    if (_self_loops || v != target(e, _u))
                    {
                        if (!_disable_xdist)
                            hist_add(_x[e], _xhist, _xvals);
                        _M++;
                    }
                    _E += _eweight[e];
                }
            }

            if (!_disable_tdist)
            {
                for (auto v : vertices_range(_u))
                    hist_add(_theta[v], _thist, _tvals);
            }
        }

        ~DynamicsState()
        {
            for (auto& es : _u_edges)
                for (auto& [v, e] : es)
                    delete e;
        }

        DynamicsState(const DynamicsState&) = delete;

        typedef BlockState block_state_t;
        BlockState& _block_state;
        typename BlockState::g_t& _u = _block_state._g;
        adj_list<> _dummy_hint;
        typename BlockState::eweight_t& _eweight = _block_state._eweight;
        GraphInterface::edge_t _null_edge;

        std::vector<gt_hash_map<size_t, GraphInterface::edge_t*>> _u_edges;

        size_t _E = 0;
        size_t _M = 0;

        DStateBase* _dstate;

        void set_dstate(DStateBase& dstate)
        {
            _dstate = &dstate;
        }

        typename x_t::checked_t _xc;

        std::vector<double> _xvals;
        std::vector<double> _tvals;

        gt_hash_map<double, size_t> _xhist;
        gt_hash_map<double, size_t> _thist;

        std::vector<std::shared_mutex> _e_mutex;
        std::vector<std::mutex> _v_mutex;

        template <bool insert, class Graph, class Elist>
        auto& _get_edge(size_t u, size_t v, Graph& g, Elist& edges)
        {
            if (!graph_tool::is_directed(g) && u > v)
                std::swap(u, v);
            auto& qe = edges[u];

            GraphInterface::edge_t* e;

            if constexpr (insert)
            {
                std::unique_lock lock(_e_mutex[u]);
                auto& ep = qe[v];
                if (ep == nullptr)
                    ep = new GraphInterface::edge_t();
                e = ep;
            }
            else
            {
                std::shared_lock lock(_e_mutex[u]);
                auto iter = qe.find(v);
                if (iter != qe.end())
                    e = iter->second;
                else
                    e = &_null_edge;
            }

            return *e;
        }

        template <class Graph, class Elist>
        void _erase_edge(size_t u, size_t v, Graph& g, Elist& edges)
        {
            if (!graph_tool::is_directed(g) && u > v)
                std::swap(u, v);
            auto& qe = edges[u];

            std::unique_lock lock(_e_mutex[u]);
            auto iter = qe.find(v);
            delete iter->second;
            qe.erase(iter);
        }

        template <bool insert=false>
        auto& get_u_edge(size_t u, size_t v)
        {
            return _get_edge<insert>(u, v, _u, _u_edges);
        }

        void erase_u_edge(size_t u, size_t v)
        {
            return _erase_edge(u, v, _u, _u_edges);
        }

        std::tuple<size_t, xval_t> edge_state(size_t u, size_t v)
        {
            auto&& e = get_u_edge(u, v);
            if (e == _null_edge)
                return {0, 0};
            return {_eweight[e], _xc[e]};
        }

        double get_node_prob(size_t u)
        {
            return _dstate->get_node_prob(u);
        }

        template <class Hist>
        size_t get_count(Hist& h, double r)
        {
            auto iter = h.find(r);
            if (iter == h.end())
                return 0;
            return iter->second;
        }

        template <class Hist, class Vals>
        void hist_add(double r, Hist& h, Vals& vals, size_t delta=1)
        {
            auto& c = h[r];
            if (c == 0)
                vals.insert(std::upper_bound(vals.begin(), vals.end(), r), r);
            c += delta;
        }

        template <class Hist, class Vals>
        void hist_remove(double r, Hist& h, Vals& vals, size_t delta=1)
        {
            auto& c = h[r];
            c -= delta;
            if (c == 0)
            {
                h.erase(r);
                vals.erase(std::lower_bound(vals.begin(), vals.end(), r));
            }
        }

        template <class Hist>
        double hist_entropy(size_t N, Hist& h, bool uniform, double l1,
                            double delta, bool nonzero)
        {
            auto W = h.size();
            double S = 0;
            if (N == 0)
                return S;
            for (auto& [x, c] : h)
                S -= qlaplace_lprob(x, l1, delta, nonzero);
            if (uniform)
            {
                S += N * safelog_fast(W);
            }
            else
            {
                S += safelog_fast(N);
                S += lbinom_fast(N - 1, W - 1);
                S += lgamma_fast(N + 1);
                for (auto& [r, n] : h)
                    S -= lgamma_fast(n + 1);
            }
            return S;
        }

        template <class Hist>
        double hist_move_dS(double r, double s, size_t N, Hist& h, bool uniform,
                            double l1, double delta, bool nonzero, size_t dn=1)
        {
            if (r == s)
                return 0;

            double Sa = 0;
            double Sb = 0;

            size_t nr = get_count(h, r);
            size_t ns = get_count(h, s);
            size_t W = h.size();
            int dW = 0;

            assert(nr > 0);
            assert(nr >= dn);

            if (nr == dn)
            {
                Sa -= -qlaplace_lprob(r, l1, delta, nonzero);
                dW--;
            }

            if (ns == 0)
            {
                Sa += -qlaplace_lprob(s, l1, delta, nonzero);
                dW++;
            }

            auto get_S =
                [&](size_t nr, size_t ns, size_t W)
                {
                    double S = 0;
                    if (uniform)
                    {
                        S += N * safelog_fast(W);
                    }
                    else
                    {
                        S += safelog_fast(N);
                        S += lbinom_fast(N - 1, W - 1);
                        S -= lgamma_fast(nr + 1) + lgamma_fast(ns + 1);
                    }
                    return S;
                };

            Sb += get_S(nr, ns, W);
            Sa += get_S(nr - dn, ns + dn, W + dW);

            assert(!isinf(Sa-Sb));

            return Sa - Sb;
        }

        template <bool Add, class Hist>
        double hist_modify_dS(double r, size_t N, Hist& h, bool uniform,
                              double l1, double delta, bool nonzero,
                              size_t dn=1)
        {
            double Sa = 0;
            double Sb = 0;

            size_t nr = get_count(h, r);
            size_t W = h.size();
            int dW = 0;
            int dN = 0;
            if constexpr (Add)
            {
                if (nr == 0)
                {
                    Sa += -qlaplace_lprob(r, l1, delta, nonzero);
                    dW++;
                }
                dN += dn;
            }
            else
            {
                dN -= dn;
                if (nr == dn)
                {
                    Sa -= -qlaplace_lprob(r, l1, delta, nonzero);
                    dW--;
                }
            }

            auto get_S =
                [&](size_t N, size_t nr, size_t W)
                {
                    double S = 0;
                    if (N == 0)
                        return S;
                    if (uniform)
                    {
                        S += N * safelog_fast(W);
                    }
                    else
                    {
                        S += safelog_fast(N);
                        S += lbinom_fast(N - 1, W - 1);
                        S += lgamma_fast(N + 1);
                        S -= lgamma_fast(nr + 1);
                    }
                    return S;
                };

            Sb += get_S(N,      nr,      W);
            Sa += get_S(N + dN, nr + dN, W + dW);

            assert(!isinf(Sa-Sb));
            return Sa - Sb;
        }

        double entropy(const dentropy_args_t& ea)
        {
            double S = 0;

            if (ea.latent_edges)
            {
                #pragma omp parallel reduction(+:S)
                parallel_vertex_loop_no_spawn
                    (_u,
                     [&](auto v)
                     {
                         S -= _dstate->get_node_prob(v);
                     });
                S *= ea.alpha;
            }

            if (ea.density)
                S += -(_E * log(ea.aE)) + lgamma_fast(_E + 1) - ea.aE;

            #pragma omp parallel reduction(+:S)
            parallel_edge_loop_no_spawn
                (_u,
                 [&](auto e)
                 {
                     if (source(e, _u) == target(e, _u) && !_self_loops)
                         return;
                     S += edge_x_S(_x[e], ea);
                 });

            size_t N = num_vertices(_u);
            size_t T = graph_tool::is_directed(_u) ?
                N * (_self_loops ? N : N-1) :
                (N * (_self_loops ? N + 1 : N - 1)) / 2;
            S += (T - _E) * edge_x_S(0, ea);

            if (!_disable_xdist && ea.xdist)
                S += hist_entropy(_M, _xhist, ea.xdist_uniform, ea.xl1, ea.delta, true);

            if (!_disable_tdist && ea.tdist)
                S += hist_entropy(num_vertices(_u), _thist, ea.tdist_uniform, ea.tl1, 0., false);

            #pragma omp parallel reduction(+:S)
            parallel_vertex_loop_no_spawn
                (_u,
                 [&](auto v)
                 {
                    S += node_x_S(_theta[v], ea);
                 });

// #ifndef NDEBUG
//             for (auto v : vertices_range(_u))
//                 _dstate->check_m(*this, v);
// #endif
            return S;
        }

        double dstate_edge_dS(size_t u, size_t v, double x, double nx,
                              const dentropy_args_t& ea)
        {
            double dS = 0;
            dS += _dstate->get_edge_dS(u, v, x, nx);
            if (u != v && !graph_tool::is_directed(_u) && !isinf(dS))
                dS += _dstate->get_edge_dS(v, u, x, nx);
            return dS * ea.alpha;
        }

        double dstate_edges_dS(std::vector<size_t>& us, size_t v,
                               std::vector<double>& x,
                               std::vector<double>& nx,
                               const dentropy_args_t& ea)
        {
            double dS = _dstate->get_edges_dS(us, v, x, nx);
            return dS * ea.alpha;
        }

        double dstate_node_dS(size_t v, double dx, const dentropy_args_t& ea)
        {
            double dS = _dstate->get_node_dS(v, dx);
            return dS * ea.alpha;
        }

        std::shared_mutex _sbm_mutex;
        std::shared_mutex _x_mutex;
        std::shared_mutex _t_mutex;

        template <class F, class Mutex>
        void do_lock(F&& f, Mutex& mutex, bool lock)
        {
            if (lock)
            {
                std::shared_lock lock(mutex);
                f();
            }
            else
            {
                f();
            }
        }

        double edge_x_S(xval_t x, const dentropy_args_t& ea)
        {
            if ((ea.sbm && x == 0) || ea.xdist)
                return 0.;
            double S = 0;
            if (ea.normal)
                S -= norm_lpdf(x, ea.mu, ea.sigma);
            else if (ea.xl1 > 0 && ea.delta == 0)
                S -= laplace_lpdf(x, ea.xl1);
            else if (ea.xl1 > 0)
                S -= qlaplace_lprob(x, ea.xl1, ea.delta/2, ea.sbm);
            if (std::isnan(S))
                cout << x << " " << ea.xl1 << " " << ea.delta << " " << ea.sbm << " " << S << endl;
            return S;
        }

        double node_x_S(xval_t x, const dentropy_args_t& ea)
        {
            double S = 0;
            if (!ea.tdist && ea.tl1 > 0)
                S -= laplace_lpdf(x, ea.tl1);
            if (std::isnan(S))
                cout << x << " " << ea.tl1 << " " << ea.tdist << " " << S << endl;
            return S;
        }

        double remove_edge_dS(size_t u, size_t v, int dm,
                              const dentropy_args_t& ea, bool dstate = true,
                              bool lock = true)
        {
            if (dm == 0)
                return 0;

            auto& e = get_u_edge(u, v);
            auto x = _xc[e];

            assert(x != 0);

            double dS = 0;

            if (ea.sbm)
            {
                do_lock([&]()
                        {
                            dS += _block_state.modify_edge_dS(u, v, e, -dm, ea);
                        }, _sbm_mutex, lock);
            }

            if (ea.density)
            {
                dS += log(ea.aE) * dm;
                dS += lgamma_fast(_E + 1 - dm) - lgamma_fast(_E + 1);
            }

            if (ea.latent_edges)
            {
                if (_eweight[e] == dm && (_self_loops || u != v))
                {
                    if (dstate)
                        dS += dstate_edge_dS(u, v, x, 0, ea);

                    dS += edge_x_S(0, ea) - edge_x_S(x, ea);

                    if (ea.xdist && !_disable_xdist)
                    {
                        do_lock([&]()
                                {
                                    dS += hist_modify_dS<false>(x, _M, _xhist,
                                                                ea.xdist_uniform,
                                                                ea.xl1, ea.delta,
                                                                ea.sbm);
                                }, _x_mutex, lock);
                    }
                }
            }

            return dS;
        }

        double add_edge_dS(size_t u, size_t v, int dm, double x,
                           const dentropy_args_t& ea, bool dstate = true,
                           bool lock = true)
        {
            if (dm == 0)
                return 0;

            assert(x != 0);

            auto& e = get_u_edge(u, v);

            auto m = (e == _null_edge) ? 0 : _eweight[e];

            if (m + dm > _max_m)
                return numeric_limits<double>::infinity();

            double dS = 0;

            if (ea.sbm)
            {
                do_lock([&]()
                        {
                            dS += _block_state.modify_edge_dS(u, v, e, dm, ea);
                        }, _sbm_mutex, lock);
            }

            if (ea.density)
            {
                dS -= log(ea.aE) * dm;
                dS += lgamma_fast(_E + 1 + dm) - lgamma_fast(_E + 1);
            }

            assert(!std::isinf(dS) && !std::isnan(dS));

            if (ea.latent_edges)
            {
                if ((e == _null_edge || _eweight[e] == 0) && (_self_loops || u != v))
                {
                    if (dstate)
                        dS += dstate_edge_dS(u, v, 0, x, ea);
                    assert(!std::isinf(dS) && !std::isnan(dS));

                    dS += edge_x_S(x, ea) - edge_x_S(0, ea);

                    assert(!std::isinf(dS) && !std::isnan(dS));
                    if (ea.xdist && !_disable_xdist)
                    {
                        do_lock([&]()
                                {
                                    dS += hist_modify_dS<true>(x, _M, _xhist,
                                                               ea.xdist_uniform,
                                                               ea.xl1, ea.delta,
                                                               ea.sbm);
                                    assert(!isinf(dS));
                                }, _x_mutex, lock);
                    }
                    assert(!std::isinf(dS) && !std::isnan(dS));
                }
            }

            return dS;
        }

        double update_edge_dS(size_t u, size_t v, double nx, const dentropy_args_t& ea,
                              bool dstate = true, bool lock = true)
        {
            assert(nx != 0);

            double dS = 0;
            if (ea.latent_edges)
            {
                auto& e = get_u_edge(u, v);
                auto x = _x[e];

                if (x == nx)
                    return 0;

                if (_self_loops || u != v)
                {
                    if (dstate)
                        dS += dstate_edge_dS(u, v, x, nx, ea);

                    assert(!std::isinf(dS) && !std::isnan(dS));

                    dS += (edge_x_S(nx, ea) - edge_x_S(x, ea));

                    assert(!std::isinf(dS) && !std::isnan(dS));

                    if (ea.xdist && !_disable_xdist)
                    {
                        do_lock([&]()
                                {
                                    dS += hist_move_dS(x, nx, _M, _xhist,
                                                       ea.xdist_uniform,
                                                       ea.xl1, ea.delta, ea.sbm);
                                    assert(!std::isinf(dS) && !std::isnan(dS));
                                }, _x_mutex, lock);
                    }
                }
            }
            return dS;
        }

        template <class F>
        double update_edges_dS(F&& get_es, double x, double nx,
                               const dentropy_args_t& ea)
        {
            gt_hash_map<size_t, std::vector<size_t>> edges;
            std::vector<std::tuple<size_t, size_t>> eds;

            get_es([&](auto u, auto v)
                   {
                       edges[v].push_back(u);
                       if (!graph_tool::is_directed(_u))
                           edges[u].push_back(v);
                       eds.emplace_back(u, v);
                   });

            double dx = nx - x;
            if (dx == 0 || eds.empty())
                return 0.;

            std::vector<std::tuple<size_t, std::vector<size_t>*>> temp;
            for (auto& [v, us] : edges)
                temp.emplace_back(v, &us);

            double dS = 0;
            std::vector<double> xs, nxs;
            #pragma omp parallel for schedule(runtime) reduction(+:dS) firstprivate(xs, nxs)
            for (size_t i = 0; i < temp.size(); ++i)
            {
                auto& [v, us] = temp[i];
                xs.resize(us->size());
                nxs.resize(us->size());
                std::fill(xs.begin(), xs.end(), x);
                std::fill(nxs.begin(), nxs.end(), nx);
                dS += dstate_edges_dS(*us, v, xs, nxs, ea);
            }

            if (x != 0 && nx != 0)
            {
                if (ea.xdist && !_disable_xdist)
                    dS += hist_move_dS(x, nx, _E, _xhist, ea.xdist_uniform,
                                       ea.xl1, ea.delta, ea.sbm, eds.size());
                dS += (edge_x_S(nx, ea) - edge_x_S(x, ea)) * eds.size();
            }
            else if (x == 0)
            {
                assert(nx != 0);
                for (auto& [u, v] : eds)
                {
                    dS += add_edge_dS(u, v, 1, nx, ea, false);
                    add_edge(u, v, 1, nx, [](){}, false);
                }
                for (auto& [u, v] : eds)
                    remove_edge(u, v, 1, [](){}, false);
            }
            else
            {
                std::vector<int> ms;
                for (auto& [u, v] : eds)
                {
                    auto m = get<0>(edge_state(u, v));
                    dS += remove_edge_dS(u, v, m, ea, false);
                    remove_edge(u, v, m, [](){}, false);
                    ms.push_back(m);
                }
                size_t pos = 0;
                for (auto& [u, v] : eds)
                    add_edge(u, v, ms[pos++], x, [](){}, false);
            }

            return dS;
        }

        template <class Unlock = std::function<void(void)>>
        void remove_edge(size_t u, size_t v, int dm, Unlock&& unlock = [](){},
                         bool dstate=true)
        {
            //serial part
            if (dm == 0)
            {
                unlock();
                return;
            }

            auto& e = get_u_edge(u, v);
            auto m = _eweight[e];
            auto x = _x[e];

            assert(e != _null_edge);

            {
                std::unique_lock lock(_sbm_mutex);
                _block_state.template modify_edge<false>(u, v, e, dm);
                if (e == _null_edge)
                    erase_u_edge(u, v);
            }

            #pragma omp atomic
            _E -= dm;

            if ((m == dm) && (_self_loops || u != v))
            {
                if (!_disable_xdist)
                {
                    std::unique_lock lock(_x_mutex);
                    hist_remove(x, _xhist, _xvals);
                }

                #pragma omp atomic
                _M--;

                unlock();

                // parallel part
                if (dstate)
                {
                    _dstate->update_edge(u, v, x, 0);
                    if (u != v && !graph_tool::is_directed(_u))
                        _dstate->update_edge(v, u, x, 0);
                }
            }
            else
            {
                unlock();
            }

        }

        template <class Unlock = std::function<void(void)>>
        void add_edge(size_t u, size_t v, int dm, double nx,
                      Unlock&& unlock = [](){}, bool dstate = true)
        {
            // serial part
            if (dm == 0)
            {
                unlock();
                return;
            }

            assert (nx != 0);

            auto& e = get_u_edge<true>(u, v);

            {
                std::unique_lock lock(_sbm_mutex);
                _block_state.template modify_edge<true>(u, v, e, dm);
            }

            #pragma omp atomic
            _E += dm;

            if (_eweight[e] == dm)
            {
                _xc[e] = nx;
                if (_self_loops || u != v)
                {
                    if (!_disable_xdist)
                    {
                        std::unique_lock lock(_x_mutex);
                        hist_add(nx, _xhist, _xvals);
                    }

                    #pragma omp atomic
                    _M++;

                    unlock();

                    //parallel part
                    if (dstate)
                    {
                        _dstate->update_edge(u, v, 0, nx);
                        if (u != v && !graph_tool::is_directed(_u))
                            _dstate->update_edge(v, u, 0, nx);
                    }
                }
                else
                {
                    unlock();
                }
            }
            else
            {
                unlock();
            }
        }

        template <class Unlock = std::function<void(void)>>
        void update_edge(size_t u, size_t v, double nx, Unlock&& unlock = [](){})
        {
            if (_self_loops || u != v)
            {
                // serial part
                auto& e = get_u_edge(u, v);
                auto x = _x[e];
                if (x == nx)
                {
                    unlock();
                    return;
                }

                if (!_disable_xdist)
                {
                    std::unique_lock lock(_x_mutex);
                    hist_remove(x, _xhist, _xvals);
                    hist_add(nx, _xhist, _xvals);
                }

                assert(nx != 0);
                _x[e] = nx;

                unlock();

                // parallel part
                _dstate->update_edge(u, v, x, nx);
                if (u != v && !graph_tool::is_directed(_u))
                    _dstate->update_edge(v, u, x, nx);
            }
            else
            {
                unlock();
            }
        }

        template <class F>
        void update_edges(F&& get_es, double x, double nx)
        {
            if (nx == x)
                return;

            gt_hash_map<size_t, std::vector<size_t>> edges;
            std::vector<std::tuple<size_t, size_t>> eds;

            get_es([&](auto u, auto v)
                   {
                       edges[v].push_back(u);
                       if (!graph_tool::is_directed(_u))
                           edges[u].push_back(v);
                       eds.emplace_back(u, v);
                   });

            std::vector<std::tuple<size_t, std::vector<size_t>*>> temp;
            for (auto& [v, us] : edges)
                temp.emplace_back(v, &us);

            std::vector<double> xs, nxs;
            #pragma omp parallel for schedule(runtime) firstprivate(xs, nxs)
            for (size_t i = 0; i < temp.size(); ++i)
            {
                auto& [v, us] = temp[i];
                xs.resize(us->size());
                nxs.resize(us->size());
                std::fill(xs.begin(), xs.end(), x);
                std::fill(nxs.begin(), nxs.end(), nx);
                _dstate->update_edges(*us, v, xs, nxs);
            }

            if (x != 0 && nx != 0)
            {
                for (auto& [u, v] : eds)
                {
                    auto e = get_u_edge(u, v);
                    _x[e] = nx;
                }

                if (!_disable_xdist)
                {
                    hist_remove(x, _xhist, _xvals, eds.size());
                    hist_add(nx, _xhist, _xvals, eds.size());
                }
            }
            else if (x == 0)
            {
                assert(nx != 0);
                for (auto& [u, v] : eds)
                    add_edge(u, v, 1, nx, [](){}, false);
            }
            else
            {
                for (auto& [u, v] : eds)
                {
                    auto m = get<0>(edge_state(u, v));
                    remove_edge(u, v, m, [](){}, false);
                }
            }
        }

        double update_node_dS(size_t v, double nt, const dentropy_args_t& ea,
                              bool dstate = true, bool lock = true)
        {
            auto t = _theta[v];
            if (nt == t)
                return 0;

            double dt = nt - t;

            double dS = 0;
            if (dstate)
                dS += dstate_node_dS(v, dt, ea);

            if (ea.tdist && !_disable_tdist)
            {
                do_lock([&]()
                        {
                            dS += hist_move_dS(t, nt, num_vertices(_u), _thist,
                                               ea.tdist_uniform, ea.tl1, 0, false);
                        }, _t_mutex, lock);
            }

            dS += node_x_S(nt, ea) - node_x_S(t, ea);

            return dS;
        }

        template <class VS>
        double update_nodes_dS(VS& vs_, double x, double nx, const dentropy_args_t& ea)
        {
            if (nx == x)
                return 0;
            double dx = nx - x;
            double dS = 0;
            std::vector<size_t> vs(vs_.begin(), vs_.end());
            #pragma omp parallel for schedule(runtime) reduction(+:dS)
            for (size_t i = 0; i < vs.size(); ++i)
                dS += dstate_node_dS(vs[i], dx, ea);
            if (ea.tdist && !_disable_tdist)
                dS += hist_move_dS(x, nx, num_vertices(_u), _thist,
                                   ea.tdist_uniform, ea.tl1, 0, false, vs.size());
            dS += (node_x_S(nx, ea) - node_x_S(x, ea)) * vs.size();
            return dS;
        }

        void update_node(size_t v, double nt)
        {
            auto t = _theta[v];

            if (nt == t)
                return;

            _theta[v] = nt;

            if (!_disable_tdist)
            {
                std::unique_lock lock(_t_mutex);
                hist_remove(t, _thist, _tvals);
                hist_add(nt, _thist, _tvals);
            }
        }

        template <class VS>
        void update_nodes(VS& vs, double x, double nx)
        {
            if (x == nx)
                return;

            for (auto v : vs)
                _theta[v] = nx;

            if (!_disable_tdist)
            {
                hist_remove(x, _thist, _tvals, vs.size());
                hist_add(nx, _thist, _tvals, vs.size());
            }
        }

        double node_TE(size_t u, size_t v)
        {
            return _dstate->node_TE(u, v);
        }

        double node_MI(size_t u, size_t v)
        {
            return _dstate->node_MI(u, v);
        }

        double node_cov(size_t u, size_t v, bool toffset, bool pearson)
        {
            return _dstate->node_cov(u, v, toffset, pearson);
        }

        void shift_zero(xval_t& x)
        {
            if (x == 0)
            {
                if (_xmax_bound > 0)
                    x = std::nextafter(x, _xmax_bound);
                else
                    x = std::nextafter(x, _xmin_bound);
            }
        }

        template <bool keep_iter, class Graph, class WMap, class IMap, class RNG>
        std::tuple<size_t, size_t, size_t>
        get_candidate_edges(Graph& g, size_t k, double r, size_t max_rk,
                            double epsilon, bool c_stop, size_t max_iter, WMap w,
                            IMap ei, const dentropy_args_t& ea, bool exact,
                            bool knn, bool keep_all, bool gradient, double h,
                            size_t f_max_iter, double tol, bool allow_edges,
                            bool include_edges, bool use_hint, size_t nrandom,
                            bool verbose, RNG& rng_)
        {
            bool directed = graph_tool::is_directed(_u);

            size_t N = num_vertices(g);
            size_t M = (directed) ? N * (N - 1) : (N * (N - 1)) / 2;
            bool complete = knn ? k >= N - 1 : k >= M;
            if (complete)
                exact = true;

            parallel_rng<rng_t> prng(rng_);

            double xa = std::numeric_limits<double>::quiet_NaN();
            double xb = std::numeric_limits<double>::quiet_NaN();

            if (!_xvals.empty())
            {
                auto iter = std::lower_bound(_xvals.begin(), _xvals.end(), 0);
                if (iter != _xvals.end())
                    xb = *iter;
                if (iter != _xvals.begin())
                    --iter;
                xa = *iter;
            }

            auto d_ =
                [&](size_t u, size_t v, bool edges)
                {
                    auto [m_, x_] = edge_state(u, v);
                    auto m = m_; // workaround clang bug
                    auto x = x_;
                    if (m > 0 && !edges)
                        return std::numeric_limits<double>::infinity();

                    if (complete)
                        return 0.;

                    auto f =
                        [&](auto nx)
                        {
                            shift_zero(nx);
                            // if (abs(nx) < ea.delta/2)
                            //     nx = (nx < 0) ? -ea.delta/2 : ea.delta/2;
                            if (m == 0)
                                return add_edge_dS(u, v, 1, nx, ea, true, false);
                            else
                                return update_edge_dS(u, v, nx, ea, true, false);
                        };

                    if (gradient)
                    {
                        if (_xvals.empty())
                        {
                            auto get_dS =
                                [&](auto x, auto nx)
                                {
                                    return (dstate_edge_dS(u, v, x, nx, ea) +
                                            (edge_x_S(nx, ea) - edge_x_S(x, ea)));
                                };

                            // central difference works better for L1
                            return -abs(get_dS(x - h, x + h) / (2 * h));
                        }
                        else
                        {
                            double dS = f(xa);
                            if (!std::isnan(xb) && xa != xb)
                                dS = std::min(dS, f(xb));
                            return dS;
                        }
                    }

                    if (_xmin_bound != _xmax_bound)
                    {
                        auto [nx, cache] =
                            bisect(f, x, _xmin_bound, _xmax_bound, f_max_iter, tol,
                                   false,
                                   _xvals.empty() ? -std::numeric_limits<double>::infinity() : _xvals.front(),
                                   _xvals.empty() ? std::numeric_limits<double>::infinity() : _xvals.back());

                        double dS = cache[nx];

                        if (!_xvals.empty())
                        {
                            auto& rng = prng.get(rng_);
                            FibonacciSearch<size_t> fb;
                            dS = std::min(dS,
                                          get<1>(fb.search(0, _xvals.size() - 1,
                                                           [&](size_t xi){ return f(_xvals[xi]); },
                                                           0, 0, rng)));
                        }

                        if (m > 0)
                            dS = std::min(dS, remove_edge_dS(u, v, m, ea, true,
                                                             false));

                        return dS;
                    }
                    else
                    {
                        return add_edge_dS(u, v, 1, _xmin_bound, ea, true, false);
                    }
                };

            auto d__ =
                [&](size_t u, size_t v)
                {
                    return d_(u, v, allow_edges);
                };

            auto d = make_dist_cache<is_directed_::apply<g_t>::type::value,
                                     true, keep_iter>
                (d__, num_vertices(_u));

            reversed_graph<Graph> g_(g);
            size_t n_tot = 0;
            size_t n_iter = 1;

            if (k > 0)
            {
                if (!knn)
                {
                    if (exact)
                        n_tot = gen_k_nearest_exact<true>(g_, d, k, directed, w);
                    else if (use_hint)
                        std::tie(n_tot, n_iter) =
                            gen_k_nearest<true>(g_, d, k, r, max_rk, epsilon, c_stop,
                                                max_iter, w, _u, directed, verbose, rng_);
                    else
                        std::tie(n_tot, n_iter) =
                            gen_k_nearest<true>(g_, d, k, r, max_rk, epsilon, c_stop,
                                                max_iter, w, _dummy_hint, directed,
                                                verbose, rng_);
                }
                else
                {
                    if (exact)
                        n_tot = gen_knn_exact<true>(g_, d, k, w);
                    else if (use_hint)
                        n_tot = gen_knn<true>(g_, d, k, r, max_rk, epsilon, c_stop,
                                              max_iter, w, _u, verbose, rng_);
                    else
                        n_tot = gen_knn<true>(g_, d, k, r, max_rk, epsilon, c_stop,
                                              max_iter, w, _dummy_hint, verbose, rng_);
                }
            }

            if (keep_all)
            {
                for (auto v : vertices_range(g))
                    clear_vertex(v, g);
                for (auto v : vertices_range(g))
                {
                    for (const auto& [u, li] : d._cache[v])
                    {
                        double l;
                        if constexpr (keep_iter)
                            l = get<0>(li);
                        else
                            l = li;
                        if (std::isinf(l))
                            continue;
                        auto e = boost::add_edge(u, v, g).first;
                        w[e] = l;
                    }
                }
            }

            if constexpr (keep_iter)
            {
                for (auto e : edges_range(g))
                {
                    auto u = source(e, g);
                    auto v = target(e, g);
                    if (!directed && (u > v))
                        std::swap(u, v);
                    ei[e] = get<1>(d._cache[v][u]);
                }
            }

            if (include_edges)
            {
                for (auto e : edges_range(_u))
                {
                    auto u = source(e, _u);
                    auto v = target(e, _u);
                    if (edge(u, v, g).second)
                        continue;
                    if (!directed && edge(v, u, g).second)
                        continue;
                    auto ne = boost::add_edge(u, v, g).first;
                    w[ne] = d_(u, v, true);
                }
            }

            if (_self_loops)
            {
                for (auto v : vertices_range(_u))
                {
                    if (edge(v, v, g).second)
                        continue;
                    auto ne = boost::add_edge(v, v, g).first;
                    w[ne] = d_(v, v, true);
                }
            }

            std::uniform_int_distribution<size_t> sample(0, num_vertices(_u)-1);
            for (size_t i = 0; i < nrandom; ++i)
            {
                size_t u, v;
                do
                {
                    u = sample(rng_);
                    v = sample(rng_);
                }
                while ((u == v || _self_loops) && edge(v, u, g).second);
                auto ne = boost::add_edge(u, v, g).first;
                w[ne] = d_(u, v, true);
            }

            return {n_tot, n_iter, d._miss_count};
        }

        bool update_bounds(double nx, double& xmin, double& xmax,
                           double xmin_bound, double xmax_bound, double tol)
        {
            auto xmin_ = xmin;
            auto xmax_ = xmax;

            auto close = [&] (auto a, auto b)
                         {
                             return abs(a - b) < std::max(2 * tol, (xmax_ - xmin_) / 10);
                         };

            if (nx < xmin)
                xmin = nx;

            if (nx > xmax)
                xmax = nx;

            if (close(nx, xmin))
            {
                if (xmin < 0)
                    xmin *= 10;
                else
                    xmin /= 10;
                xmin = std::max(xmin, xmin_bound);
            }

            if (close(nx, xmax))
            {
                if (xmax > 0)
                    xmax *= 10;
                else
                    xmax /= 10;
                xmax = std::min(xmax, xmax_bound);
            }

            if (xmin != xmin_ || xmax != xmax_)
                return true;
            return false;
        }

        template <class F>
        std::tuple<double, std::map<xval_t, double>>
        bisect(F&& f, double x, double xmin_bound, double xmax_bound,
               std::uintmax_t maxiter, double tol, bool reversible,
               double xmin = -numeric_limits<double>::infinity(),
               double xmax = numeric_limits<double>::infinity())
        {
            std::map<xval_t, double> xcache;

            auto f_ =
                [&](auto nx)
                {
                    auto iter = xcache.find(nx);
                    if (iter != xcache.end())
                        return iter->second;
                    auto f_x = f(nx);
                    xcache[nx] = f_x;
                    return f_x;
                };

            double nx = numeric_limits<double>::quiet_NaN();

            if (maxiter == 0)
                maxiter = std::numeric_limits<std::uintmax_t>::max();


            if (std::isinf(xmin) || std::isinf(xmax))
            {
                if (reversible)
                {
                    xmin = std::max(-1., xmin_bound);
                    xmax = std::min(+1., xmax_bound);
                }
                else
                {
                    xmin = (x < 0) ? x * 2 : x / 2;
                    xmax = (x > 0) ? x * 2 : x / 2;
                    xmin = std::max(xmin - 1, xmin_bound);
                    xmax = std::min(xmax + 1, xmax_bound);
                }
            }

            size_t it = 0;
            do
            {
                double dS;
                std::tie(nx, dS) =
                    boost::math::tools::brent_find_minima(f_, xmin, xmax,
                                                          ceil(1-log2(tol)),
                                                          maxiter);
                if (f_(xmin) < dS)
                    std::tie(nx, dS) = std::make_tuple(xmin, f_(xmin));
                if (f_(xmax) < dS)
                    std::tie(nx, dS) = std::make_tuple(xmax, f_(xmax));
            }
            while ((boost::math::epsilon_difference(f_(xmin), f_(xmax)) > 3) &&
                   update_bounds(nx, xmin, xmax, xmin_bound, xmax_bound, tol) &&
                   (++it < _nmax_extend));

            nx = std::min_element(xcache.begin(), xcache.end(),
                                  [&](auto& a, auto& b)
                                  { return a.second < b.second; })->first;

            return {nx, xcache};
        }

        template <class F, class... RNG>
        std::tuple<double, std::map<xval_t, double>>
        bisect_fb(F&& f, std::vector<xval_t>& vals, RNG&... rng)
        {
            std::map<xval_t, double> xcache;

            auto f_ =
                [&](auto nx)
                {
                    auto iter = xcache.find(nx);
                    if (iter != xcache.end())
                        return iter->second;
                    auto f_x = f(nx);
                    xcache[nx] = f_x;
                    return f_x;
                };

            FibonacciSearch<size_t> fb;
            auto [i, dS] = fb.search(0, vals.size() - 1,
                                     [&](size_t i){ return f_(vals[i]); },
                                     0, 0, rng...);
            auto nx = vals[i];
            return {nx, xcache};
        }

        template <class... RNG>
        std::tuple<double, std::map<xval_t, double>>
        bisect_x(size_t u, size_t v, size_t maxiter, double tol,
                 const dentropy_args_t& ea, bool reversible, bool fb,
                 RNG&... rng)
        {
            size_t m;
            double x;
            std::tie(m, x) = edge_state(u, v);

            auto f =
                [&](auto nx)
                {
                    shift_zero(nx);
                    return (dstate_edge_dS(u, v, x, nx, ea) +
                            (edge_x_S(nx, ea) - edge_x_S(x, ea)));
                };

            double nx;
            std::map<xval_t, double> xcache;

            if (fb)
            {
                assert(!_xvals.empty());
                std::shared_lock lock(_x_mutex);
                std::tie(nx, xcache) = bisect_fb(f, _xvals, rng...);
                assert(nx != 0);
            }
            else
            {
                if (ea.delta == 0 || true)
                {
                    std::tie(nx, xcache) =
                        bisect(f, x, _xmin_bound, _xmax_bound, maxiter,
                               tol, reversible);
                }
                else
                {
                    if (_xmin_bound < -ea.delta/2)
                    {
                        double xmax = std::min(_xmax_bound, -ea.delta/2);
                        auto [nx_, xcache_] =
                            bisect(f, x, _xmin_bound, xmax, maxiter, tol, reversible);
                        xcache.insert(xcache_.begin(), xcache_.end());
                    }

                    if (_xmax_bound > ea.delta/2)
                    {
                        double xmin = std::max(_xmin_bound, ea.delta/2);
                        auto [nx_, xcache_] =
                            bisect(f, x, xmin, _xmax_bound, maxiter, tol, reversible);
                        xcache.insert(xcache_.begin(), xcache_.end());
                    }

                    assert(!xcache.empty());

                    nx = std::min_element(xcache.begin(), xcache.end(),
                                          [&](auto& a, auto& b)
                                          { return a.second < b.second; })->first;

                    assert(!std::isnan(nx));
                }
            }

            // remove zero
            auto iter = xcache.find(0.);
            if (iter != xcache.end())
            {
                auto dS = iter->second;
                xcache.erase(iter);
                if (nx == 0)
                {
                    shift_zero(nx);
                    xcache[nx] = dS;
                }
            }

            assert(!std::isnan(nx));
            assert (nx != 0);
            return {nx, xcache};
        }

        template <class... RNG>
        std::tuple<double, std::map<xval_t, double>>
        bisect_t(size_t v, size_t maxiter, double tol,
                 const dentropy_args_t& ea, bool reversible, bool fb,
                 RNG&... rng)
        {
            auto x = _theta[v];
            auto f =
                [&](auto nx)
                {
                    return dstate_node_dS(v, nx - x, ea) +
                        (node_x_S(nx, ea) - node_x_S(x, ea));
                };

            if (fb)
            {
                std::shared_lock lock(_t_mutex);
                return bisect_fb(f, _tvals, rng...);
            }
            {
                return bisect(f, x, _tmin_bound, _tmax_bound, maxiter, tol,
                              reversible);

            }
        }

        template <class... RNG>
        std::tuple<double, std::map<xval_t, double>>
        bisect_xl1(double minval, double maxval, size_t maxiter, double tol,
                   const dentropy_args_t& ea, bool reversible)
        {
            dentropy_args_t ea_ = ea;
            double S0 = entropy(ea);

            auto f =
                [&](auto nx)
                {
                    ea_.xl1 = nx;
                    return entropy(ea_) - S0;
                };

            return bisect(f, ea.xl1, minval, maxval, maxiter, tol,
                          reversible);
        }

        std::tuple<double, std::map<xval_t, double>>
        bisect_tl1(double minval, double maxval, size_t maxiter, double tol,
                   const dentropy_args_t& ea, bool reversible)
        {
            dentropy_args_t ea_ = ea;
            double S0 = entropy(ea);

            auto f =
                [&](auto nx)
                {
                    ea_.tl1 = nx;
                    return entropy(ea_) - S0;
                };

            return bisect(f, ea.tl1, minval, maxval, maxiter, tol,
                          reversible);
        }

        double edge_diff(size_t u, size_t v, double h, const dentropy_args_t& ea)
        {
            auto get_dS =
                [&](auto x, auto nx)
                {
                    return (dstate_edge_dS(u, v, x, nx, ea) +
                            (edge_x_S(nx, ea) - edge_x_S(x, ea)));
                };

            auto x = get<1>(edge_state(u, v));
            return get_dS(x - h, x + h) / (2 * h);
        }

        double node_diff(size_t v, double h, const dentropy_args_t& ea)
        {
            auto get_dS =
                [&](auto x, auto nx)
                {
                    return (dstate_node_dS(v, nx - x, ea) +
                            (node_x_S(nx, ea) - node_x_S(x, ea)));
                };

            auto x = _theta[v];
            return get_dS(x - h, x + h) / (2 * h);
        }

        void quantize_x(double delta)
        {
            std::vector<std::tuple<size_t,size_t,size_t>> es;
            std::vector<std::mutex> vmutex(num_vertices(_u));

            auto dispatch =
                [&](size_t u, size_t v, auto&& f)
                {
                    bool dir = graph_tool::is_directed(_u) || u == v;

                    if (dir)
                        vmutex[v].lock();
                    else
                        std::lock(vmutex[u], vmutex[v]);

                    f();

                    if (dir)
                    {
                        vmutex[v].unlock();
                    }
                    else
                    {
                        vmutex[u].unlock();
                        vmutex[v].unlock();
                    }
                };

            parallel_edge_loop
                (_u,
                 [&](auto e)
                 {
                     size_t u = source(e, _u);
                     size_t v = target(e, _u);
                     double nx = round(_x[e] / delta) * delta;
                     if (nx != 0)
                     {
                         dispatch(u, v,
                                  [&]()
                                  {
                                      update_edge(u, v, nx);
                                  });
                     }
                     else
                     {
                         #pragma omp critical
                         es.emplace_back(u, v, get<0>(edge_state(u, v)));
                     }
                 });

            parallel_loop(es,
                          [&](size_t, auto& uvm)
                          {
                              auto& [u_, v_, m_] = uvm;
                              auto u = u_; // workaround clang bug
                              auto v = v_;
                              auto m = m_;
                              dispatch(u, v,
                                       [&]()
                                       {
                                           remove_edge(u, v, m);
                                       });
                          });
        }

        std::tuple<double, std::map<xval_t, double>>
        bisect_delta(double minval, double maxval, size_t maxiter, double tol,
                     const dentropy_args_t& ea, bool reversible)
        {
            dentropy_args_t ea_ = ea;

            auto f =
                [&](auto delta)
                {
                    std::vector<std::tuple<size_t,size_t,size_t,double>> es;
                    for (auto e : edges_range(_u))
                        es.emplace_back(source(e, _u), target(e, _u),
                                        _eweight[e], _x[e]);

                    quantize_x(delta);

                    ea_.delta = delta;
                    double S = entropy(ea_);
                    if (ea_.sbm)
                        S += _block_state.entropy(ea_);

                    for (auto& [s, t, nm, x] : es)
                    {
                        auto m = get<0>(edge_state(s, t));
                        if (m > 0)
                            update_edge(s, t, x);
                        else
                            add_edge(s, t, nm, x);
                    }

                    return S;
                };

            return bisect(f, ea.delta, minval, maxval, maxiter, tol,
                          reversible);
        }

        template <class Cache>
        SegmentSampler get_seg_sampler(Cache&& xcache, double beta)
        {
            std::vector<xval_t> xs;
            std::vector<double> ws;

            for (auto& [x, w] : xcache)
            {
                xs.push_back(x);
                ws.push_back(-w * beta);
            }

            return SegmentSampler(xs, ws);
        }

        template <class Cache, class RNG>
        double
        sample_val(Cache&& xcache, double beta, RNG& rng)
        {
            if (std::isinf(beta))
                return std::min_element(xcache.begin(), xcache.end(),
                                        [&](auto& a, auto& b)
                                        { return a.second < b.second; })->first;

            SegmentSampler seg = get_seg_sampler(xcache, beta);

            auto nx = seg.sample(rng);
            return nx;
        };

        template <class RNG>
        std::tuple<double, std::map<xval_t, double>>
        sample_x(size_t u, size_t v, double beta, size_t maxiter, double tol,
                 const dentropy_args_t& ea, bool fb, RNG& rng)
        {
            if (_xmin_bound == _xmax_bound)
                return {_xmin_bound, std::map<xval_t, double>()};
            auto bisect =
                [&]() { return bisect_x(u, v, maxiter, tol, ea, !std::isinf(beta), fb, rng); };
            auto [nx_, xcache] = bisect();
            xval_t nx = 0;
            while (nx == 0)
                nx = sample_val(xcache, beta, rng); // remove zero
            // if (abs(nx) < ea.delta)
            //     nx = nx < 0 ? -ea.delta : ea.delta;
            assert(!std::isinf(nx) && !std::isnan(nx));
            return {nx, xcache};
        }

        template <class RNG>
        std::tuple<double, std::map<xval_t, double>>
        sample_t(size_t v, double beta, size_t maxiter, double tol,
                 const dentropy_args_t& ea, bool fb, RNG& rng)
        {
            if (_tmin_bound == _tmax_bound)
                return {_tmin_bound, std::map<xval_t, double>()};
            auto bisect =
                [&]() { return bisect_t(v, maxiter, tol, ea, !std::isinf(beta), fb, rng); };
            auto [nx_, xcache] = bisect();
            return {sample_val(xcache, beta, rng), xcache};
        }

        template <class RNG>
        std::tuple<double, std::map<xval_t, double>>
        sample_xl1(double beta, double minval, double maxval, size_t maxiter,
                   double tol, const dentropy_args_t& ea, RNG& rng)
        {
            auto bisect =
                [&]() { return bisect_xl1(minval, maxval, maxiter, tol, ea,
                                          !std::isinf(beta)); };
            auto [nx_, xcache] = bisect();
            return {sample_val(xcache, beta, rng), xcache};
        }

        template <class RNG>
        std::tuple<double, std::map<xval_t, double>>
        sample_tl1(double beta, double minval, double maxval, size_t maxiter,
                   double tol, const dentropy_args_t& ea, RNG& rng)
        {
            auto bisect =
                [&]() { return bisect_tl1(minval, maxval, maxiter, tol, ea,
                                          !std::isinf(beta)); };
            auto [nx_, xcache] = bisect();
            return {sample_val(xcache, beta, rng), xcache};
        }

        template <class RNG>
        std::tuple<double, std::map<xval_t, double>>
        sample_delta(double beta, double minval, double maxval, size_t maxiter,
                     double tol, const dentropy_args_t& ea, RNG& rng)
        {
            auto bisect =
                [&]() { return bisect_delta(minval, maxval, maxiter, tol, ea,
                                            !std::isinf(beta)); };
            auto [nx_, xcache] = bisect();
            return {sample_val(xcache, beta, rng), xcache};
        }

        double sample_val_lprob(xval_t x, std::map<xval_t, double>& xcache, double beta)
        {
            SegmentSampler seg = get_seg_sampler(xcache, beta);
            return seg.lprob(x);
        }

        double sample_x_lprob(xval_t x, std::map<xval_t, double>& xcache, double beta)
        {
            if (_xmin_bound == _xmax_bound)
                return 0;
            return sample_val_lprob(x, xcache, beta);
        }

        double sample_t_lprob(xval_t x, std::map<xval_t, double>& xcache, double beta)
        {
            if (_tmin_bound == _tmax_bound)
                return 0;
            return sample_val_lprob(x, xcache, beta);
        }

        template <class Vals, class Val>
        std::tuple<double, double, double>
        bracket_closest(Vals& vals, Val x,
                        Val skip = numeric_limits<Val>::quiet_NaN(),
                        Val add = numeric_limits<Val>::quiet_NaN())
        {
            auto iter = std::lower_bound(vals.begin(), vals.end(), x);
            auto begin = iter - std::min<size_t>(3, iter - vals.begin());
            auto end = iter + std::min<size_t>(4, vals.end() - iter);

            std::vector<Val> pts;
            for (auto iter = begin; iter != end; ++iter)
            {
                if (*iter == skip)
                    continue;
                pts.push_back(*iter);
            }

            if (!std::isnan(add))
                pts.insert(std::lower_bound(pts.begin(), pts.end(), add), add);

            auto pos = std::lower_bound(pts.begin(), pts.end(), x);
            if (pos == pts.end() || ((pos != pts.begin()) &&
                                     (x - *(pos-1)) < (*pos - x)))
                --pos;

            auto y = *pos;

            Val a = (pos != pts.begin()) ? *(pos - 1) + (y - *(pos - 1))/2 :
                -numeric_limits<Val>::infinity();
            Val b = ((pos + 1) != pts.end()) ? y + (*(pos + 1) - y)/2 :
                numeric_limits<Val>::infinity();
            return {a, y, b};
        }

        template <class Vals, class Val>
        double
        find_closest(Vals& vals, Val x,
                     Val skip = numeric_limits<xval_t>::quiet_NaN(),
                     Val add = numeric_limits<xval_t>::quiet_NaN())
        {
            return get<1>(bracket_closest(vals, x, skip, add));
        }

        template <class Vals, class Val>
        std::tuple<Val, Val>
        get_close_int(Vals& vals, Val x,
                      Val skip = numeric_limits<Val>::quiet_NaN(),
                      Val add = numeric_limits<Val>::quiet_NaN())
        {
            auto ret = bracket_closest(vals, x, skip, add);
            return {get<0>(ret), get<2>(ret)};
        }

        template <class F>
        std::tuple<double, double, std::map<xval_t, double>>
        val_sweep(F&& f_, double x, double xmin_bound, double xmax_bound,
                  double beta, std::uintmax_t maxiter, double tol)
        {
            std::map<xval_t, double> xcache;

            auto f =
                [&](auto x)
                {
                    auto iter = xcache.find(x);
                    if (iter != xcache.end())
                        return iter->second;
                    auto fx = f_(x);
                    xcache[x] = fx;
                    return fx;
                };

            if (maxiter == 0)
                maxiter = std::numeric_limits<std::uintmax_t>::max();

            double xmin;
            double xmax;

            if (!std::isinf(beta))
            {
                xmin = std::max(-1., xmin_bound);
                xmax = std::min(+1., xmax_bound);
            }
            else
            {
                xmin = (x < 0) ? x * 2 : x / 2;
                xmax = (x > 0) ? x * 2 : x / 2;
                xmin = std::max(xmin - 1, xmin_bound);
                xmax = std::min(xmax + 1, xmax_bound);
            }

            size_t it = 0;
            double nx, dS;
            do
            {
                std::tie(nx, dS) =
                    boost::math::tools::brent_find_minima
                    (f, xmin, xmax, ceil(1-log2(tol)), maxiter);
                if (f(xmin) < dS)
                    std::tie(nx, dS) = std::make_tuple(xmin, f(xmin));
                if (f(xmax) < dS)
                    std::tie(nx, dS) = std::make_tuple(xmax, f(xmax));
            }
            while ((boost::math::epsilon_difference(f(xmin), f(xmax)) > 3) &&
                   update_bounds(nx, xmin, xmax, xmin_bound, xmax_bound, tol) &&
                   (++it < _nmax_extend));

            std::tie(nx, dS) =
                *std::min_element(xcache.begin(), xcache.end(),
                                  [&](auto& a, auto& b)
                                  { return a.second < b.second; });

            return {nx, dS, xcache};
        }


        template <bool try_zero, class F, class U, class Hist, class Vals,
                  class RNG>
        double vals_sweep(F&& f_, U& update_items, Hist& hist, Vals& vals,
                          double xmin_bound, double xmax_bound,
                          double beta, double r, std::uintmax_t maxiter,
                          double tol, size_t min_size, RNG& rng)
        {
            std::bernoulli_distribution skip(1-r);

            double S = 0;
            for (size_t xi = 0; xi < vals.size(); ++xi)
            {
                if ((get_count(hist, vals[xi]) < min_size) || skip(rng))
                    continue;

                auto f = [&](auto x)
                         {
                             return f_(xi, vals[xi], x);
                         };

                auto [nx, dS, xcache] =
                    val_sweep(f, vals[xi], xmin_bound, xmax_bound, beta, maxiter, tol);

                if (try_zero && xcache.find(0) == xcache.end())
                {
                    double dS0 = f(0);
                    xcache[0] = dS0;
                    if (dS0 < dS)
                        std::tie(nx, dS) = std::make_tuple(0., dS0);
                }

                if (std::isinf(beta))
                {
                    if (dS >= 0)
                        continue;
                }
                else
                {
                    double lf = 0;
                    double lb = 0;

                    SegmentSampler seg = get_seg_sampler(xcache, beta);
                    nx = seg.sample(rng);

                    dS = f(nx);

                    lf = seg.lprob(nx);
                    lb = seg.lprob(vals[xi]);

                    std::uniform_real_distribution<> u(0, 1);

                    double a = -beta * dS + lb - lf;

                    if (std::isinf(lb) || (a <= 0 && exp(a) <= u(rng)))
                        continue;
                }

                auto x = vals[xi];

                update_items(xi, x, nx);

                S += dS;
            }
            return S;
        }


        template <class RNG>
        std::tuple<double, size_t>
        xvals_sweep(double beta, double r, size_t maxiter, double tol,
                    size_t min_size, const dentropy_args_t& ea, RNG& rng)
        {
            std::vector<std::vector<typename graph_traits<g_t>::edge_descriptor>>
                eds(_xhist.size());

            std::vector<double> vals;
            for (auto& [x, c] : _xhist)
                vals.push_back(x);

            std::shuffle(vals.begin(), vals.end(), rng);

            gt_hash_map<double, size_t> vmap;
            for (auto x : vals)
            {
                auto r = vmap.size();
                vmap[x] = r;
            }

            for (auto e : edges_range(_u))
            {
                auto x = _x[e];
                auto r = vmap[x];
                eds[r].push_back(e);
            }

            auto f =
                [&](size_t xi, auto x, auto nx)
                {
                    auto& es = eds[xi];
                    return update_edges_dS([&](auto&& f)
                                           {
                                               for (auto& e : es)
                                                   f(source(e, _u), target(e, _u));
                                           }, x, nx, ea);
                };

            size_t nmoves = 0;

            auto update_edges =
                [&](size_t xi, double x, double nx)
                {
                    auto& es = eds[xi];
                    this->update_edges([&](auto&& f)
                                       {
                                           for (auto& e : es)
                                               f(source(e, _u), target(e, _u));
                                       }, x, nx);
                    nmoves += es.size();
                };

            return {vals_sweep<true>(f, update_edges, _xhist, vals, _xmin_bound,
                                     _xmax_bound, beta, r, maxiter, tol, min_size,
                                     rng), nmoves};
        }

        template <class RNG>
        std::tuple<double, size_t>
        tvals_sweep(double beta, double r, size_t maxiter, double tol,
                    size_t min_size, const dentropy_args_t& ea, RNG& rng)
        {
            std::vector<double> vals;
            for (auto& [x, c] : _thist)
                vals.push_back(x);

            std::shuffle(vals.begin(), vals.end(), rng);

            gt_hash_map<double, size_t> vmap;
            for (auto x : vals)
            {
                auto r = vmap.size();
                vmap[x] = r;
            }

            gt_hash_map<size_t, std::vector<size_t>> vertices;
            for (auto v : vertices_range(_u))
            {
                auto x = _theta[v];
                vertices[vmap[x]].push_back(v);
            }

            auto f =
                [&](size_t xi, auto x, auto nx)
                {
                    return update_nodes_dS(vertices[xi], x, nx, ea);
                };

            size_t nmoves = 0;

            auto update_vertices =
                [&](size_t xi, double x, double nx)
                {
                    auto& vs = vertices[xi];
                    update_nodes(vs, x, nx);
                    nmoves += vs.size();
                };

            return {vals_sweep<false>(f, update_vertices, _thist, vals, _tmin_bound,
                                      _tmax_bound, beta, r, maxiter, tol, min_size,
                                      rng), nmoves};
        }

        std::vector<double>& get_xvals()
        {
            return _xvals;
        }

        std::vector<double>& get_tvals()
        {
            return _tvals;
        }

        void set_params(boost::python::dict p)
        {
            _dstate->set_params(p);
        }
    };
};


} // graph_tool namespace

#endif //DYNAMICS_HH
