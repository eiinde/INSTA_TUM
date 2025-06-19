#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2024 Tiago de Paula Peixoto <tiago@skewed.de>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from .. import Vector_size_t, Vector_double

import numpy
from . util import *
from . base_states import *
from . nested_blockmodel import NestedBlockState

def mcmc_equilibrate(state, wait=1000, nbreaks=2, max_niter=numpy.inf,
                     force_niter=None, epsilon=0, gibbs=False, multiflip=True,
                     mcmc_args={}, entropy_args={}, history=False,
                     callback=None, verbose=False):
    r"""Equilibrate a MCMC with a given starting state.

    Parameters
    ----------
    state : Any state class (e.g. :class:`~graph_tool.inference.BlockState`)
        Initial state. This state will be modified during the algorithm.
    wait : ``int`` (optional, default: ``1000``)
        Number of iterations to wait for a record-breaking event.
    nbreaks : ``int`` (optional, default: ``2``)
        Number of iteration intervals (of size ``wait``) without record-breaking
        events necessary to stop the algorithm.
    max_niter : ``int`` (optional, default: ``numpy.inf``)
        Maximum number of iterations.
    force_niter : ``int`` (optional, default: ``None``)
        If given, will force the algorithm to run this exact number of
        iterations.
    epsilon : ``float`` (optional, default: ``0``)
        Relative changes in entropy smaller than epsilon will not be considered
        as record-breaking.
    gibbs : ``bool`` (optional, default: ``False``)
        If ``True``, each step will call ``state.gibbs_sweep`` instead of
        ``state.mcmc_sweep``.
    multiflip : ``bool`` (optional, default: ``True``)
        If ``True`` and ``state`` is an instance of
        :class:`~graph_tool.inference.MultiflipMCMCState`, each step will call
        ``state.multiflip_mcmc_sweep`` instead of ``state.mcmc_sweep``.
    mcmc_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to ``state.mcmc_sweep`` (or ``state.gibbs_sweep``).
    history : ``bool`` (optional, default: ``False``)
        If ``True``, a list of tuples of the form ``(nattempts, nmoves,
        entropy)`` will be kept and returned, where ``entropy`` is the current
        entropy and ``nmoves`` is the number of vertices moved.
    callback : ``function`` (optional, default: ``None``)
        If given, this function will be called after each iteration. The
        function must accept the current state as an argument, and its return
        value must be either `None` or a (possibly empty) list of values that
        will be append to the history, if ``history == True``.
    verbose : ``bool`` or ``tuple`` (optional, default: ``False``)
        If ``True``, progress information will be shown. Optionally, this
        accepts arguments of the type ``tuple`` of the form ``(level, prefix)``
        where ``level`` is a positive integer that specifies the level of
        detail, and ``prefix`` is a string that is prepended to the all output
        messages.

    Notes
    -----

    The MCMC equilibration is attempted by keeping track of the maximum and
    minimum values, and waiting sufficiently long without a record-breaking
    event.

    This function calls ``state.mcmc_sweep`` (or ``state.gibbs_sweep``) at each
    iteration (e.g. :meth:`graph_tool.inference.BlockState.mcmc_sweep` and
    :meth:`graph_tool.inference.BlockState.gibbs_sweep`), and keeps track of
    the value of ``state.entropy(**args)`` with ``args`` corresponding to
    ``mcmc_args["entropy_args"]``.

    Returns
    -------

    history : list of tuples of the form ``(nattempts, nmoves, entropy)``
        Summary of the MCMC run. This is returned only if ``history == True``.
    entropy : ``float``
        Current entropy value after run. This is returned only if ``history ==
        False``.
    nattempts : ``int``
        Number of node move attempts.
    nmoves : ``int``
        Number of node moves.

    References
    ----------

    .. [peixoto-efficient-2014] Tiago P. Peixoto, "Efficient Monte Carlo and
       greedy heuristic for the inference of stochastic block models", Phys.
       Rev. E 89, 012804 (2014), :doi:`10.1103/PhysRevE.89.012804`,
       :arxiv:`1310.4378`

    """

    count = 0
    break_count = 0
    niter = 0
    total_nmoves = 0
    total_nattempts = 0
    S = state.entropy(**mcmc_args.get("entropy_args", {}))
    min_S = max_S = S
    m_eps = 1e-6
    hist = []
    while count < wait:
        if gibbs:
            delta, nattempts, nmoves = state.gibbs_sweep(**mcmc_args)
        elif multiflip and hasattr(state, "multiflip_mcmc_sweep"):
            delta, nattempts, nmoves = state.multiflip_mcmc_sweep(**mcmc_args)
        else:
            delta, nattempts, nmoves = state.mcmc_sweep(**mcmc_args)

        S += delta
        niter += 1
        total_nmoves += nmoves
        total_nattempts += nattempts

        if force_niter is not None:
            max_S = max((S, max_S))
            min_S = min((S, min_S))
            if niter >= force_niter:
                break
        else:
            if abs(delta) >= (S - delta) * epsilon:
                if S > max_S + m_eps:
                    max_S = S
                    count = 0
                elif S < min_S - m_eps:
                    min_S = S
                    count = 0
                else:
                    count += 1
            else:
                count += 1

            if count >= wait:
                break_count += 1
                if break_count < nbreaks:
                    count = 0
                    min_S = max_S = S

        extra = []
        if callback is not None:
            extra = callback(state)
            if extra is None:
                extra = []

        if check_verbose(verbose):
            print((verbose_pad(verbose) +
                   u"niter: %5d  count: %4d  breaks: %2d  min_S: %#8.8g  " +
                   u"max_S: %#8.8g  S: %#8.8g  ΔS: %#12.6g  moves: %5d %s") %
                   (niter, count, break_count, min_S, max_S, S, delta, nmoves,
                    str(extra) if len(extra) > 0 else ""))

        if history:
            hist.append(tuple([nattempts, nmoves, S] + extra))

        if niter >= max_niter:
            break

    if history:
        return hist
    else:
        return (S, total_nattempts, total_nmoves)

def mcmc_anneal(state, beta_range=(1., 10.), niter=100, history=False,
                mcmc_equilibrate_args={}, verbose=False):
    r"""Equilibrate a MCMC at a specified target temperature by performing simulated
    annealing.

    Parameters
    ----------
    state : Any state class (e.g. :class:`~graph_tool.inference.BlockState`)
        Initial state. This state will be modified during the algorithm.
    beta_range : ``tuple`` of two floats (optional, default: ``(1., 10.)``)
        Inverse temperature range.
    niter : ``int`` (optional, default: ``100``)
        Number of steps (in logspace) from the starting temperature to the final
        one.
    history : ``bool`` (optional, default: ``False``)
        If ``True``, a list of tuples of the form ``(nattempts, nmoves, beta, entropy)``
    mcmc_equilibrate_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to :func:`~graph_tool.inference.mcmc_equilibrate`.
    verbose : ``bool`` or ``tuple`` (optional, default: ``False``)
        If ``True``, progress information will be shown. Optionally, this
        accepts arguments of the type ``tuple`` of the form ``(level, prefix)``
        where ``level`` is a positive integer that specifies the level of
        detail, and ``prefix`` is a string that is prepended to the all output
        messages.

    Notes
    -----

    This algorithm employs exponential cooling, where the value of beta is
    multiplied by a constant at each iteration, so that starting from
    `beta_range[0]` the value of `beta_range[1]` is reached after `niter`
    iterations.

    At each iteration, the function
    :func:`~graph_tool.inference.mcmc_equilibrate` is called with the current
    value of `beta` (via the ``mcmc_args`` parameter).

    Returns
    -------

    history : list of tuples of the form ``(nattempts, nmoves, beta, entropy)``
        Summary of the MCMC run. This is returned only if ``history == True``.
    entropy : ``float``
        Current entropy value after run. This is returned only if ``history ==
        False``.
    nattempts : ``int``
        Number of node move attempts.
    nmoves : ``int``
        Number of node moves.

    References
    ----------

    .. [peixoto-efficient-2014] Tiago P. Peixoto, "Efficient Monte Carlo and
       greedy heuristic for the inference of stochastic block models", Phys.
       Rev. E 89, 012804 (2014), :doi:`10.1103/PhysRevE.89.012804`,
       :arxiv:`1310.4378`
    """

    beta = beta_range[0]
    hist = ([], [], [], [])
    nattempts = 0
    nmoves = 0
    speed = exp((log(beta_range[1]) - log(beta_range[0])) / niter)
    mcmc_args = mcmc_equilibrate_args.get("mcmc_args", {})
    while beta < beta_range[1] * speed:
        ret = mcmc_equilibrate(state,
                               **dict(mcmc_equilibrate_args,
                                      mcmc_args=dict(mcmc_args,
                                                     beta=beta),
                                      history=history,
                                      verbose=verbose_push(verbose,
                                                           ("β: %#8.6g  " %
                                                            beta))))
        if history:
            ret = list(zip(*ret))
            hist[0].extend(ret[0])
            hist[1].extend(ret[1])
            hist[2].extend([beta] * len(ret[0]))
            hist[3].extend(ret[2])
            S = ret[0][-1]
        else:
            S = ret[0]
            nattempts += ret[1]
            nmoves += ret[2]

        beta *= speed

    if history:
        return list(zip(hist))
    else:
        return S, nattempts, nmoves

class MulticanonicalState(object):
    r"""The density of states of a multicanonical Monte Carlo algorithm. It is used
    by :func:`graph_tool.inference.multicanonical_equilibrate`.

    Parameters
    ----------
    state : :class:`~graph_tool.inference.BlockState` or :class:`~graph_tool.inference.OverlapBlockState` or :class:`~graph_tool.inference.NestedBlockState`
        Block state to be used.
    S_min : ``float``
        Minimum energy.
    S_max : ``float``
        Maximum energy.
    nbins : ``int`` (optional, default: ``1000``)
        Number of bins.
    """

    def __init__(self, state, S_min, S_max, nbins=1000):
        self._state = state
        self._g = state.g
        self._N = self._g.num_vertices()
        self._S_min = S_min
        self._S_max = S_max
        self._density = Vector_double()
        self._density.resize(nbins)
        self._hist = Vector_size_t()
        self._hist.resize(nbins)
        self._perm_hist = numpy.zeros(nbins, dtype=self._hist.a.dtype)
        self._f = None

    def __getstate__(self):
        state = [self._state, self._S_min, self._S_max,
                 numpy.array(self._density.a), numpy.array(self._hist.a),
                 numpy.array(self._perm_hist), self._f]
        return state

    def __setstate__(self, state):
        bstate, S_min, S_max, density, hist, phist, self._f = state
        self.__init__(bstate, S_min, S_max, len(hist))
        self._density.a[:] = density
        self._hist.a[:] = hist
        self._perm_hist[:] = phist

    def sweep(self, **kwargs):
        self._state.multicanonical_sweep(self, **kwargs)

    def get_energies(self):
        "Get energy bounds."
        return self._S_min, self._S_max

    def get_allowed_energies(self):
        "Get allowed energy bounds."
        h = self._hist.a.copy()
        h += self._perm_hist
        Ss = self.get_range()
        Ss = Ss[h > 0]
        return Ss[0], Ss[-1]

    def get_range(self):
        "Get energy range."
        return numpy.linspace(self._S_min, self._S_max, len(self._hist))

    def get_density(self, B=None):
        """Get density of states, normalized so that total sum is :math:`B^N`, where
        :math:`B` is the number of groups, and :math:`N` is the number of
        nodes. If not supplied :math:`B=N` is assumed.
        """
        r = numpy.array(self._density.a)
        r -= r.max()
        r -= log(exp(r).sum())
        if B is None:
            B = self._g.num_vertices()
        r += self._g.num_vertices() * log(B)
        return r

    def get_entropy(self, S, B=None):
        r = self.get_density(B)
        j = self.get_bin()
        return r[j]

    def get_bin(self, S):
        return int(round((len(self._hist) - 1) * ((S - self._S_min) /
                                                  (self._S_max - self._S_min))))

    def get_hist(self):
        "Get energy histogram."
        return numpy.array(self._hist.a)

    def get_perm_hist(self):
        "Get permanent energy histogram."
        return self._perm_hist

    def get_flatness(self, h=None, allow_gaps=True):
        "Get energy histogram flatness."
        if h is None:
            h = self._hist.a
        if h.sum() == 0:
            return 0
        if allow_gaps:
            idx = (h + self._perm_hist) > 0
        else:
            Ss = self.get_range()
            S_min, S_max = self.get_allowed_energies()
            idx =numpy.logical_and(Ss >= S_min, Ss <= S_max)

        h = array(h[idx], dtype="float")

        if len(h) == 1:
            h = array([1e-6] + list(h))

        h_mean = h.mean()
        return min((h.min() / h_mean,
                    h_mean / h.max()))

    def get_posterior(self, N=None):
        "Get posterior probability."
        r = self.get_density(N)
        Ss = numpy.linspace(self._S_min, self._S_max, len(r))
        y = -Ss + r
        y_max = y.max()
        y -= y_max
        return y_max + log(exp(y).sum())

    def reset_hist(self):
        "Reset energy histogram."
        self._perm_hist += self._hist.a
        self._hist.a = 0

def multicanonical_equilibrate(m_state, f_range=(1., 1e-6), r=2,
                               flatness=.95, allow_gaps=True, callback=None,
                               multicanonical_args={}, verbose=False):
    r"""Equilibrate a multicanonical Monte Carlo sampling using the Wang-Landau
    algorithm.

    Parameters
    ----------
    m_state :  :class:`~graph_tool.inference.MulticanonicalState`
        Initial multicanonical state, where the state density will be stored.
    f_range : ``tuple`` of two floats (optional, default: ``(1., 1e-6)``)
        Range of density updates.
    r : ``float`` (optional, default: ``2.``)
        Greediness of convergence. At each iteration, the density updates will
        be reduced by a factor ``r``.
    flatness : ``float`` (optional, default: ``.95``)
        Sufficient histogram flatness threshold used to continue the algorithm.
    allow_gaps : ``bool`` (optional, default: ``True``)
        If ``True``, gaps in the histogram (regions with zero count) will be
        ignored when computing the flatness.
    callback : ``function`` (optional, default: ``None``)
        If given, this function will be called after each iteration. The
        function must accept the current ``state`` and ``m_state`` as arguments.
    multicanonical_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to ``state.multicanonical_sweep`` (e.g.
        :meth:`graph_tool.inference.BlockState.multicanonical_sweep`).
    verbose : ``bool`` or ``tuple`` (optional, default: ``False``)
        If ``True``, progress information will be shown. Optionally, this
        accepts arguments of the type ``tuple`` of the form ``(level, prefix)``
        where ``level`` is a positive integer that specifies the level of
        detail, and ``prefix`` is a string that is prepended to the all output
        messages.

    Returns
    -------

    niter : ``int``
        Number of iterations required for convergence.

    References
    ----------

    .. [wang-efficient-2001] Fugao Wang, D. P. Landau, "An efficient, multiple
       range random walk algorithm to calculate the density of states", Phys.
       Rev. Lett. 86, 2050 (2001), :doi:`10.1103/PhysRevLett.86.2050`,
       :arxiv:`cond-mat/0011174`
    .. [belardinelli-wang-2007] R. E. Belardinelli, V. D. Pereyra,
       "Wang-Landau algorithm: A theoretical analysis of the saturation of
       the error", J. Chem. Phys. 127, 184105 (2007),
       :doi:`10.1063/1.2803061`, :arxiv:`cond-mat/0702414`
    """

    count = 0
    if m_state._f is None:
        m_state._f = f_range[0]
    while m_state._f >= f_range[1]:
        m_state.sweep(**multicanonical_args)
        hf = m_state.get_flatness(allow_gaps=allow_gaps)

        if callback is not None:
            callback(m_state)

        if check_verbose(verbose):
            print(verbose_pad(verbose) +
                  "count: %d  f: %#8.8g  flatness: %#8.8g  nonempty bins: %d  S: %#8.8g  B: %d" % \
                  (count, m_state._f, hf, (m_state._hist.a > 0).sum(),
                   m_state._state.entropy(**multicanonical_args.get("entropy_args", {})),
                   m_state._state.get_nonempty_B()))

        if hf > flatness:
            m_state._f /= r
            if m_state._f >= f_range[1]:
                m_state.reset_hist()

        count += 1

    return count


class TemperingState(object):
    """This class aggregates several state classes and corresponding
    inverse-temperature values to implement `parallel tempering MCMC
    <https://en.wikipedia.org/wiki/Parallel_tempering>`_.

    This is meant to be used with :func:`~graph_tool.inference.mcmc_equilibrate`.

    Parameters
    ----------
    states : list of state objects (e.g. :class:`~graph_tool.inference.BlockState`)
        Initial parallel states.
    betas : list of floats
        Inverse temperature values.
    """

    def __init__(self, states, betas, idx=None, beta_dl=False):
        if not (len(states) == len(betas)):
            raise ValueError("states and betas must be of the same size")
        self.states = states
        self.betas = betas
        if idx is None:
            self.idx = list(range(len(betas)))
        self.beta_dl = beta_dl
        self.swap_attempts = 0
        self.swaps = numpy.zeros(len(states), dtype="int")

    def entropy(self, **kwargs):
        """Returns the sum of the entropy of the parallel states. All keyword
        arguments are propagated to the individual states' `entropy()`
        method.
        """
        if self.beta_dl:
            return sum(s.entropy(beta_dl=beta, **kwargs) for s, beta in
                       zip(self.states, self.betas))
        else:
            return sum(s.entropy(**kwargs) * beta for s, beta in
                       zip(self.states, self.betas))

    def entropies(self, **kwargs):
        """Returns the entropies of the parallel states. All keyword
        arguments are propagated to the individual states' `entropy()`
        method.
        """
        if self.beta_dl:
            return [s.entropy(beta_dl=beta, **kwargs) for s, beta in
                    zip(self.states, self.betas)]
        else:
            return [s.entropy(**kwargs) * beta for s, beta in
                    zip(self.states, self.betas)]

    def states_swap(self,  **kwargs):
        """Perform a full sweep of the parallel states, where swaps are attempted. All
        relevant keyword arguments are propagated to the individual states'
        `entropy()` method."""

        verbose = kwargs.get("verbose", False)
        eargs = kwargs.get("entropy_args", {})

        self.swap_attempts += 1
        idx = numpy.arange(len(self.states) - 1)
        numpy.random.shuffle(idx)
        nswaps = 0
        dS = 0
        for i in idx:
            j = i + 1

            s1 = self.states[i]
            s2 = self.states[j]
            b1 = self.betas[i]
            b2 = self.betas[j]

            if self.beta_dl:
                P1_f = -s1.entropy(beta_dl=b2, **eargs)
                P2_f = -s2.entropy(beta_dl=b1, **eargs)

                P1_b = -s1.entropy(beta_dl=b1, **eargs)
                P2_b = -s2.entropy(beta_dl=b2, **eargs)
            else:
                S1 = s1.entropy(**eargs)
                S2 = s2.entropy(**eargs)
                P1_f = -S1 * b2
                P2_f = -S2 * b1

                P1_b = -S1 * b1
                P2_b = -S2 * b2

            ddS = -(P1_f + P2_f - P1_b - P2_b)

            if ddS < 0 or numpy.random.random() < exp(-ddS):
                self.states[j], self.states[i], self.idx[j], self.idx[i] = \
                            self.states[i], self.states[j], self.idx[i], self.idx[j]
                nswaps += 1
                self.swaps[i] += 1
                dS += ddS
                if check_verbose(verbose):
                    print(verbose_pad(verbose)
                          + u"swapped states: %d [β = %g] <-> %d [β = %g], a: %g" % \
                            (i, b1, j, b2, exp(-ddS)))
        return dS, nswaps

    def states_move(self, sweep_algo, **kwargs):
        """Perform a full sweep of the parallel states, where state moves are
        attempted by calling `sweep_algo(state, beta=beta, **kwargs)`."""
        algo_states = []

        if isinstance(self.states[0], NestedBlockState):
            ls = list(kwargs.pop("ls", range(len(self.states[0].levels))))
            if kwargs.pop("ls_shuffle", True):
                numpy.random.shuffle(ls)
            kwargs["ls"] = ls
            kwargs["ls_shuffle"] = False

        for state, beta in zip(self.states, self.betas):
            entropy_args = dict(kwargs.get("entropy_args", {}))
            if self.beta_dl:
                algo_state = sweep_algo[0](state,
                                           dispatch=False,
                                           **dict(kwargs,
                                                  entropy_args=dict(entropy_args,
                                                                    beta_dl=beta)))
            else:
                algo_state = sweep_algo[0](state,
                                           dispatch=False,
                                           **dict(kwargs,
                                                  beta=beta))
            algo_states.append(algo_state)

        return sweep_algo[1](self.states, algo_states)

    def _sweep(self, algo, r=0.1, adjacent=True, **kwargs):
        if numpy.random.random() < r:
            return self.states_swap(adjacent=adjacent, **kwargs)
        else:
            return self.states_move(algo, **dict(kwargs, verbose=False))

    def mcmc_sweep(self, **kwargs):
        """Perform a full mcmc sweep of the parallel states, where swap or moves are
        chosen randomly. It accepts an keyword argument ``r`` (default: ``0.1``)
        specifying the relative probability with which state swaps are performed
        with respect to node moves.  All remaining keyword arguments are
        propagated to the individual states' `mcmc_sweep()` method.
        """
        algo = (lambda s, **kw: s.mcmc_sweep(**kw),
                lambda states, sweeps: type(self.states[0])._mcmc_sweep_parallel_dispatch(states, sweeps))
        return self._sweep(algo, **kwargs)

    def multiflip_mcmc_sweep(self, **kwargs):
        """Perform a full mcmc sweep of the parallel states, where swap or moves are
        chosen randomly. It accepts an keyword argument ``r`` (default: ``0.1``)
        specifying the relative probability with which state swaps are performed
        with respect to node moves.  All remaining keyword arguments are
        propagated to the individual states' `mcmc_sweep()` method.
        """
        algo = (lambda s, **kw: s.multiflip_mcmc_sweep(**kw),
                lambda states, sweeps: type(self.states[0])._multiflip_mcmc_sweep_parallel_dispatch(states, sweeps))
        return self._sweep(algo, **kwargs)

    def gibbs_sweep(self, **kwargs):
        """Perform a full Gibbs mcmc sweep of the parallel states, where swap or moves
        are chosen randomly. It accepts an keyword argument ``r`` (default:
        ``0.1``) specifying the relative probability with which state swaps are
        performed with respect to node moves.  All remaining keyword arguments
        are propagated to the individual states' `gibbs_sweep()` method.
        """
        algo = (lambda s, **kw: s.gibbs_sweep(**kw),
                lambda states, sweeps: type(self.states[0])._gibbs_sweep_parallel_dispatch(states, sweeps))
        return self._sweep(algo, **kwargs)
