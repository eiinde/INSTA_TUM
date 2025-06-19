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

from .. import _prop, Graph, GraphView

from . base_states import _bm_test
from . base_states import *

from . blockmodel import *
from . overlap_blockmodel import *
from . layered_blockmodel import *

from numpy import *
import numpy
import copy

class NestedBlockState(object):
    r"""The nested stochastic block model state of a given graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modeled.
    bs : ``list`` of :class:`~graph_tool.VertexPropertyMap` or :class:`numpy.ndarray` (optional, default: ``None``)
        Hierarchical node partition. If not provided it will correspond to a
        single-group hierarchy of length :math:`\lceil\log_2(N)\rceil`.
    base_type : ``type`` (optional, default: :class:`~graph_tool.inference.BlockState`)
        State type for lowermost level
        (e.g. :class:`~graph_tool.inference.BlockState`,
        :class:`~graph_tool.inference.OverlapBlockState` or
        :class:`~graph_tool.inference.LayeredBlockState`)
    hstate_args : ``dict`` (optional, default: `{}`)
        Keyword arguments to be passed to the constructor of the higher-level
        states.
    hentropy_args : ``dict`` (optional, default: `{}`)
        Keyword arguments to be passed to the ``entropy()`` method of the
        higher-level states.
    state_args : ``dict`` (optional, default: ``{}``)
        Keyword arguments to be passed to base type constructor.
    **kwargs :  keyword arguments
        Keyword arguments to be passed to base type constructor. The
        ``state_args`` parameter overrides this.

    """

    def __init__(self, g, bs=None, base_type=BlockState, state_args={},
                 hstate_args={}, hentropy_args={}, **kwargs):
        self.g = g

        self.base_type = base_type
        if base_type is LayeredBlockState:
            self.Lrecdx = []
        else:
            self.Lrecdx = libcore.Vector_double()
        self.state_args = dict(kwargs, **state_args)
        self.state_args["Lrecdx"] = self.Lrecdx
        if "rec_params" not in self.state_args:
            recs = self.state_args.get("recs", None)
            if recs is not None:
                self.state_args["rec_params"] = ["microcanonical"] * len(recs)
        self.hstate_args = dict(dict(deg_corr=False, vweight="nonempty"),
                                **hstate_args)
        self.hstate_args["Lrecdx"] = self.Lrecdx
        self.hstate_args["copy_bg"] = False
        self.hentropy_args = dict(hentropy_args,
                                  adjacency=True,
                                  dense=True,
                                  multigraph=True,
                                  dl=True,
                                  partition_dl=True,
                                  degree_dl=True,
                                  degree_dl_kind="distributed",
                                  edges_dl=False,
                                  exact=True,
                                  recs=True,
                                  recs_dl=False,
                                  beta_dl=1.)

        g_ = self.state_args.pop("g", g) # for OverlapBlockState
        self.levels = [base_type(g_, b=bs[0] if bs is not None else None,
                                 **self.state_args)]

        if bs is None:
            if base_type is OverlapBlockState:
                N = 2 * self.levels[0].get_N()
            else:
                N = self.levels[0].get_N()
            L = int(numpy.ceil(numpy.log2(N)))
            bs = [None] * (L + 1)

        for i, b in enumerate(bs[1:]):
            state = self.levels[-1]
            args = self.hstate_args
            if i == len(bs[1:]) - 1:
                args = dict(args, clabel=None, pclabel=None)
            bstate = state.get_block_state(b=b, **args)
            self.levels.append(bstate)

        self._regen_Lrecdx()

        self._couple_levels(self.hentropy_args, None)

        if _bm_test():
            self._consistency_check()

    def _regen_Lrecdx(self, lstate=None):
        if not hasattr(self.levels[0], "recdx"):
            return

        if lstate is None:
            levels = self.levels
            Lrecdx = self.Lrecdx
        else:
            levels = [s for s in self.levels]
            l, s = lstate
            levels[l] = s
            s = s.get_block_state(**dict(self.hstate_args,
                                         b=s.get_bclabel(),
                                         copy_bg=False))
            if l < len(levels) - 1:
                levels[l+1] = s
            else:
                levels.append(s)
            if self.base_type is LayeredBlockState:
                Lrecdx = [x.copy() for x in self.Lrecdx]
            else:
                Lrecdx = self.Lrecdx.copy()

        if self.base_type is not LayeredBlockState:
            Lrecdx.a = 0
            Lrecdx[0] = len([s for s in levels if s._state.get_B_E_D() > 0])
            for s in levels:
                Lrecdx.a[1:] += s.recdx.a * s._state.get_B_E_D()
                s.epsilon.a = levels[0].epsilon.a
            for s in levels:
                s.Lrecdx.a = Lrecdx.a
        else:
            Lrecdx[0].a = 0
            Lrecdx[0][0] = len([s for s in levels if s._state.get_B_E_D() > 0])
            for j in range(levels[0].C):
                Lrecdx[j+1].a = 0
                Lrecdx[j+1][0] = len([s for s in levels if s._state.get_layer(j).get_B_E_D() > 0])
            for s in levels:
                Lrecdx[0].a[1:] += s.recdx.a * s._state.get_B_E_D()
                s.epsilon.a = levels[0].epsilon.a
                for j in range(levels[0].C):
                    Lrecdx[j+1].a[1:] += s.layer_states[j].recdx.a * s._state.get_layer(j).get_B_E_D()
                    s.layer_states[j].epsilon.a = levels[0].epsilon.a

            for s in self.levels:
                for x, y in zip(s.Lrecdx, Lrecdx):
                    x.a = y.a

        if lstate is not None:
            return Lrecdx


    def _regen_levels(self):
        for l in range(1, len(self.levels)):
            state = self.levels[l]
            nstate = self.levels[l-1].get_block_state(b=state.b,
                                                      **self.hstate_args)
            self.levels[l] = nstate
        self._regen_Lrecdx()

    def __repr__(self):
        return "<NestedBlockState object, with base %s, and %d levels of sizes %s at 0x%x>" % \
            (repr(self.levels[0]), len(self.levels),
             str([(s.get_N(), s.get_nonempty_B()) for s in self.levels]), id(self))

    def __copy__(self):
        return self.copy()

    def copy(self, **kwargs):
        r"""Copies the block state. The parameters override the state properties,
        and have the same meaning as in the constructor."""
        state = dict(self.__getstate__(), **kwargs)
        return NestedBlockState(**state)

    def __getstate__(self):
        base_state = self.levels[0].__getstate__()
        base_state.pop("Lrecdx", None)
        base_state.pop("epsilon", None)
        base_state.pop("drec", None)
        state_args = dict(self.state_args, **base_state)
        if self.base_type is not OverlapBlockState:
            state_args.pop("g", None)
        state_args.pop("b", None)
        state = dict(g=self.g, bs=self.get_bs(),
                     base_type=type(self.levels[0]),
                     hstate_args=self.hstate_args,
                     hentropy_args=self.hentropy_args,
                     state_args=state_args)
        return state

    def __setstate__(self, state):
        self.__init__(**state)

    def get_bs(self):
        """Get hierarchy levels as a list of :class:`numpy.ndarray` objects with the
        group memberships at each level.
        """
        return [s.b.copy() if l == 0 else s.b.fa.copy() for l, s in enumerate(self.levels)]

    def get_state(self):
        """Alias to :meth:`~NestedBlockState.get_bs`."""
        return self.get_bs()

    def set_state(self, bs):
        r"""Sets the internal nested partition of the state."""
        for i in range(len(bs)):
            self.levels[i].set_state(bs[i])

    def get_levels(self):
        """Get hierarchy levels as a list of :class:`~graph_tool.inference.BlockState`
        instances."""
        return self.levels

    def project_partition(self, j, l):
        """Project partition of level ``j`` onto level ``l``, and return it."""
        b = self.levels[l].b.copy()
        for i in range(l + 1, j + 1):
            clabel = self.levels[i].b.copy()
            pmap(b, clabel)
        return b

    def propagate_clabel(self, l):
        """Project base clabel to level ``l``."""
        clabel = self.levels[0].clabel.copy()
        for j in range(l):
            bg = self.levels[j].bg
            bclabel = bg.new_vertex_property("int")
            reverse_map(self.levels[j].b, bclabel)
            pmap(bclabel, clabel)
            clabel = bclabel
        return clabel

    def get_clabel(self, l):
        """Get clabel for level ``l``."""
        clabel = self.propagate_clabel(l)
        if l < len(self.levels) - 1:
            b = self.project_partition(l + 1, l)
            clabel.fa += (clabel.fa.max() + 1) * b.fa
        return clabel

    def _consistency_check(self):
        for l in range(1, len(self.levels)):
            b = self.levels[l].b.fa.copy()
            state = self.levels[l-1]
            args = self.hstate_args
            if l == len(self.levels) - 1:
                args = dict(args, clabel=None, pclabel=None)
            bstate = state.get_block_state(b=b, **args)
            b2 = bstate.b.fa.copy()
            b = contiguous_map(b)
            b2 = contiguous_map(b2)
            assert ((b == b2).all() and
                    math.isclose(bstate.entropy(dl=False),
                                 self.levels[l].entropy(dl=False),
                                 abs_tol=1e-8)), \
                "inconsistent level %d (%s %g,  %s %g): %s" % \
                (l, str(bstate), bstate.entropy(), str(self.levels[l]),
                 self.levels[l].entropy(), str(self))
            assert (bstate.get_N() >= bstate.get_nonempty_B()), \
                (l, bstate.get_N(), bstate.get_nonempty_B(), str(self))

    def level_entropy(self, l, bstate=None, **kwargs):
        """Compute the entropy of level ``l``."""

        if bstate is None:
            bstate = self.levels[l]

        kwargs = kwargs.copy()
        hentropy_args = dict(self.hentropy_args,
                             **kwargs.pop("hentropy_args", {}))
        hentropy_args_top = dict(dict(hentropy_args, edges_dl=True,
                                      recs_dl=True),
                                 **kwargs.pop("hentropy_args_top", {}))

        if l > 0:
            if l == (len(self.levels) - 1):
                eargs = hentropy_args_top
            else:
                eargs = hentropy_args
        else:
            eargs = dict(kwargs, edges_dl=False)

        S = bstate.entropy(**eargs)

        if l > 0:
            S *= kwargs.get("beta_dl", 1.)

        return S

    def _Lrecdx_entropy(self, Lrecdx=None):
        if not hasattr(self.levels[0], "recdx"):
            return 0

        if self.base_type is not LayeredBlockState:
            S_D = 0

            if Lrecdx is None:
                Lrecdx = self.Lrecdx
                for s in self.levels:
                    B_E_D = s._state.get_B_E_D()
                    if B_E_D > 0:
                        S_D -= log(B_E_D)

            S = 0
            for i in range(len(self.levels[0].rec)):
                if self.levels[0].rec_types[i] != libinference.rec_type.real_normal:
                    continue
                assert not _bm_test() or Lrecdx[i+1] >= 0, (i, Lrecdx[i+1])
                S += -libinference.positive_w_log_P(Lrecdx[0], Lrecdx[i+1],
                                                    numpy.nan, numpy.nan,
                                                    self.levels[0].epsilon[i])
                S += S_D
            return S
        else:
            S_D = [0 for j in range(self.levels[0].C)]
            if Lrecdx is None:
                Lrecdx = self.Lrecdx
                for s in self.levels:
                    for j in range(self.levels[0].C):
                        B_E_D = s._state.get_layer(j).get_B_E_D()
                        if B_E_D > 0:
                            S_D[j] -= log(B_E_D)

            S = 0
            for i in range(len(self.levels[0].rec)):
                if self.levels[0].rec_types[i] != libinference.rec_type.real_normal:
                    continue
                for j in range(self.levels[0].C):
                    assert not _bm_test() or Lrecdx[j+1][i+1] >= 0, (i, j, Lrecdx[j+1][i+1])
                    S += -libinference.positive_w_log_P(Lrecdx[j+1][0],
                                                        Lrecdx[j+1][i+1],
                                                        numpy.nan, numpy.nan,
                                                        self.levels[0].epsilon[i])
                    S += S_D[j]
            return S

    @copy_state_wrap
    def entropy(self, **kwargs):
        """Obtain the description length (i.e. negative joint log-likelihood)
        for the hierarchical partition.

        The keyword arguments are passed to the ``entropy()`` method of the
        underlying state objects
        (e.g. :class:`graph_tool.inference.BlockState.entropy`,
        :class:`graph_tool.inference.OverlapBlockState.entropy`, or
        :class:`graph_tool.inference.LayeredBlockState.entropy`).

        """
        S = 0
        for l in range(len(self.levels)):
            S += self.level_entropy(l, **dict(kwargs, test=False))

        S += kwargs.get("beta_dl", 1.) * self._Lrecdx_entropy()

        return S

    def move_vertex(self, v, s):
        r"""Move vertex ``v`` to block ``s``."""
        self.levels[0].move_vertex(v, s)
        self._regen_levels()

    def remove_vertex(self, v):
        r"""Remove vertex ``v`` from its current group.

        This optionally accepts a list of vertices to remove.

        .. warning::

           This will leave the state in an inconsistent state before the vertex
           is returned to some other group, or if the same vertex is removed
           twice.
        """
        self.levels[0].remove_vertex(v)
        self._regen_levels()

    def add_vertex(self, v, r):
        r"""Add vertex ``v`` to block ``r``.

        This optionally accepts a list of vertices and blocks to add.

        .. warning::

           This can leave the state in an inconsistent state if a vertex is
           added twice to the same group.
        """
        self.levels[0].add_vertex(v, r)
        self._regen_levels()

    def get_edges_prob(self, missing, spurious=[], entropy_args={}):
        r"""Compute the joint log-probability of the missing and spurious edges given by
        ``missing`` and ``spurious`` (a list of ``(source, target)``
        tuples, or :meth:`~graph_tool.Edge` instances), together with the
        observed edges.

        More precisely, the log-likelihood returned is

        .. math::

            \ln \frac{P(\boldsymbol G + \delta \boldsymbol G | \boldsymbol b)}{P(\boldsymbol G| \boldsymbol b)}

        where :math:`\boldsymbol G + \delta \boldsymbol G` is the modified graph
        (with missing edges added and spurious edges deleted).

        The values in ``entropy_args`` are passed to
        :meth:`graph_tool.inference.BlockState.entropy()` to calculate the
        log-probability.
        """

        entropy_args = entropy_args.copy()
        hentropy_args = dict(self.hentropy_args,
                             **entropy_args.pop("hentropy_args", {}))
        hentropy_args_top = dict(dict(hentropy_args, edges_dl=True,
                                      recs_dl=True),
                                 **entropy_args.pop("hentropy_args_top", {}))

        L = 0
        for l, lstate in enumerate(self.levels):
            if l > 0:
                if l == (len(self.levels) - 1):
                    eargs = hentropy_args_top
                else:
                    eargs = hentropy_args
            else:
                eargs = entropy_args

            lstate._couple_state(None, None)
            if l > 0:
                lstate._state.sync_emat()
                lstate._state.clear_egroups()

            L += lstate.get_edges_prob(missing, spurious, entropy_args=eargs)
            if isinstance(self.levels[0], LayeredBlockState):
                missing = [(lstate.b[u], lstate.b[v], l_) for u, v, l_ in missing]
                spurious = [(lstate.b[u], lstate.b[v], l_) for u, v, l_ in spurious]
            else:
                missing = [(lstate.b[u], lstate.b[v]) for u, v in missing]
                spurious = [(lstate.b[u], lstate.b[v]) for u, v in spurious]

        return L

    def get_bstack(self):
        """Return the nested levels as individual graphs.

        This returns a list of :class:`~graph_tool.Graph` instances
        representing the inferred hierarchy at each level. Each graph has two
        internal vertex and edge property maps named "count" which correspond to
        the vertex and edge counts at the lower level, respectively. Additionally,
        an internal vertex property map named "b" specifies the block partition.
        """

        bstack = []
        for l, bstate in enumerate(self.levels):
            cg = bstate.g
            if l == 0:
                cg = GraphView(cg, skip_properties=True)
            cg.vp["b"] = bstate.b.copy()
            if bstate.is_weighted:
                cg.ep["count"] = cg.own_property(bstate.eweight.copy())
                cg.vp["count"] = cg.own_property(bstate.vweight.copy())
            else:
                cg.ep["count"] = cg.new_ep("int", 1)

            bstack.append(cg)
            if bstate.get_N() == 1:
                break
        return bstack

    def project_level(self, l):
        """Project the partition at level ``l`` onto the lowest level, and return the
        corresponding state."""
        b = self.project_partition(l, 0)
        return self.levels[0].copy(b=b)

    def print_summary(self):
        """Print a hierarchy summary."""
        for l, state in enumerate(self.levels):
            print("l: %d, N: %d, B: %d" % (l, state.get_N(),
                                           state.get_nonempty_B()))
            if state.get_N() == 1:
                break

    def _couple_levels(self, hentropy_args, hentropy_args_top):
        if hentropy_args_top is None:
            hentropy_args_top = dict(hentropy_args, edges_dl=True, recs_dl=True)
        for l in range(len(self.levels) - 1):
            if l + 1 == len(self.levels) - 1:
                eargs = hentropy_args_top
            else:
                eargs = hentropy_args
            self.levels[l]._couple_state(self.levels[l + 1], eargs)

    def _clear_egroups(self):
        for lstate in self.levels:
            lstate._clear_egroups()

    def _h_sweep_gen(self, **kwargs):

        verbose = kwargs.get("verbose", False)
        entropy_args = dict(kwargs.get("entropy_args", {}), edges_dl=False)
        hentropy_args = dict(self.hentropy_args,
                             **entropy_args.pop("hentropy_args", {}))
        hentropy_args_top = dict(dict(hentropy_args, edges_dl=True,
                                      recs_dl=True),
                                 **entropy_args.pop("hentropy_args_top", {}))

        self._couple_levels(hentropy_args, hentropy_args_top)

        c = kwargs.get("c", None)

        lrange = list(kwargs.pop("ls", range(len(self.levels))))
        if kwargs.pop("ls_shuffle", True):
            numpy.random.shuffle(lrange)
        for l in lrange:
            if check_verbose(verbose):
                print(verbose_pad(verbose) + "level:", l)
            if l > 0:
                if l == len(self.levels) - 1:
                    eargs = hentropy_args_top
                else:
                    eargs = hentropy_args
            else:
                eargs = entropy_args

            if c is None:
                args = dict(kwargs, entropy_args=eargs)
            else:
                args = dict(kwargs, entropy_args=eargs, c=c[l])

            if l > 0:
                if "beta_dl" in entropy_args:
                    args = dict(args, beta=args.get("beta", 1.) * entropy_args["beta_dl"])
                for p in ["B_max", "B_min", "b_max", "b_min", "vertices"]:
                    args.pop(p, None)

            yield l, self.levels[l], args

    def _h_sweep(self, algo, **kwargs):
        entropy_args = kwargs.get("entropy_args", {})

        dS = 0
        nattempts = 0
        nmoves = 0

        for l, lstate, args in self._h_sweep_gen(**kwargs):

            ret = algo(self.levels[l], **dict(args, test=False))

            if l > 0 and "beta_dl" in entropy_args:
                dS += ret[0] * entropy_args["beta_dl"]
            else:
                dS += ret[0]
            nattempts += ret[1]
            nmoves += ret[2]

        return dS, nattempts, nmoves

    def _h_sweep_states(self, algo, **kwargs):
        entropy_args = kwargs.get("entropy_args", {})
        for l, lstate, args in self._h_sweep_gen(**kwargs):
            beta_dl = entropy_args.get("beta_dl", 1) if l > 0 else 1
            yield l, lstate, algo(self.levels[l], dispatch=False, **args), beta_dl

    def _h_sweep_parallel_dispatch(states, sweeps, algo):
        ret = None
        for lsweep in zip(*sweeps):
            ls = [x[0] for x in lsweep]
            lstates = [x[1] for x in lsweep]
            lsweep_states = [x[2] for x in lsweep]
            beta_dl = [x[3] for x in lsweep]
            lret = algo(type(lstates[0]), lstates, lsweep_states)
            if ret is None:
                ret = lret
            else:
                ret = [(ret[i][0] + lret[i][0] * beta_dl[i],
                        ret[i][1] + lret[i][1],
                        ret[i][2] + lret[i][2]) for i in range(len(lret))]
        return ret

    @mcmc_sweep_wrap
    def mcmc_sweep(self, **kwargs):
        r"""Perform ``niter`` sweeps of a Metropolis-Hastings acceptance-rejection
        MCMC to sample hierarchical network partitions.

        The arguments accepted are the same as in
        :meth:`graph_tool.inference.BlockState.mcmc_sweep`.

        If the parameter ``c`` is a scalar, the values used at each level are
        ``c * 2 ** l`` for ``l`` in the range ``[0, L-1]``. Optionally, a list
        of values may be passed instead, which specifies the value of ``c[l]``
        to be used at each level.

        .. warning::

           This function performs ``niter`` sweeps at each hierarchical level
           once. This means that in order for the chain to equilibrate, we need
           to call this function several times, i.e. it is not enough to call
           it once with a large value of ``niter``.

        """

        c = kwargs.pop("c", 1)
        if not isinstance(c, collections.abc.Iterable):
            c = [c * 2 ** l for l in range(0, len(self.levels))]

        if kwargs.pop("dispatch", True):
            return self._h_sweep(lambda s, **a: s.mcmc_sweep(**a), c=c,
                                 **kwargs)
        else:
            return self._h_sweep_states(lambda s, **a: s.mcmc_sweep(**a),
                                        c=c, **kwargs)

    def _mcmc_sweep_parallel_dispatch(states, sweeps):
        algo = lambda s, lstates, lsweep_states: s._mcmc_sweep_parallel_dispatch(lstates, lsweep_states)
        return NestedBlockState._h_sweep_parallel_dispatch(states, sweeps, algo)

    @mcmc_sweep_wrap
    def multiflip_mcmc_sweep(self, **kwargs):
        r"""Perform ``niter`` sweeps of a Metropolis-Hastings acceptance-rejection MCMC
        with multiple moves to sample hierarchical network partitions.

        The arguments accepted are the same as in
        :meth:`graph_tool.inference.BlockState.multiflip_mcmc_sweep`.

        If the parameter ``c`` is a scalar, the values used at each level are
        ``c * 2 ** l`` for ``l`` in the range ``[0, L-1]``. Optionally, a list
        of values may be passed instead, which specifies the value of ``c[l]``
        to be used at each level.

        .. warning::

           This function performs ``niter`` sweeps at each hierarchical level
           once. This means that in order for the chain to equilibrate, we need
           to call this function several times, i.e. it is not enough to call
           it once with a large value of ``niter``.

        """

        kwargs["psingle"] = kwargs.get("psingle", self.levels[0].get_N())

        c = kwargs.pop("c", 1)
        if not isinstance(c, collections.abc.Iterable):
            c = [c * 2 ** l for l in range(0, len(self.levels))]

        if kwargs.pop("dispatch", True):
            def dispatch_level(s, **a):
                if s is not self.levels[0]:
                    a = dict(**a)
                    a.pop("B_min", None)
                    a.pop("B_max", None)
                    a.pop("b_min", None)
                    a.pop("b_max", None)
                return s.multiflip_mcmc_sweep(**a)

            return self._h_sweep(dispatch_level, c=c, **kwargs)
        else:
            return self._h_sweep_states(lambda s, **a: s.multiflip_mcmc_sweep(**a),
                                        c=c, **kwargs)

    def _multiflip_mcmc_sweep_parallel_dispatch(states, sweeps):
        algo = lambda s, lstates, lsweep_states: s._multiflip_mcmc_sweep_parallel_dispatch(lstates, lsweep_states)
        return NestedBlockState._h_sweep_parallel_dispatch(states, sweeps, algo)

    @mcmc_sweep_wrap
    def multilevel_mcmc_sweep(self, **kwargs):
        r"""Perform ``niter`` sweeps of a Metropolis-Hastings acceptance-rejection MCMC
        with multilevel moves to sample hierarchical network partitions.

        The arguments accepted are the same as in
        :meth:`graph_tool.inference.BlockState.multilevel_mcmc_sweep`.

        If the parameter ``c`` is a scalar, the values used at each level are
        ``c * 2 ** l`` for ``l`` in the range ``[0, L-1]``. Optionally, a list
        of values may be passed instead, which specifies the value of ``c[l]``
        to be used at each level.

        .. warning::

           This function performs ``niter`` sweeps at each hierarchical level
           once. This means that in order for the chain to equilibrate, we need
           to call this function several times, i.e. it is not enough to call
           it once with a large value of ``niter``.

        """

        c = kwargs.pop("c", 1)
        if not isinstance(c, collections.abc.Iterable):
            c = [c * 2 ** l for l in range(0, len(self.levels))]

        if kwargs.pop("dispatch", True):
            return self._h_sweep(lambda s, **a: s.multilevel_mcmc_sweep(**a),
                                 c=c, **kwargs)
        else:
            return self._h_sweep_states(lambda s, **a: s.multilevel_mcmc_sweep(**a),
                                        c=c, **kwargs)

    def _multilevel_mcmc_sweep_parallel_dispatch(states, sweeps):
        algo = lambda s, lstates, lsweep_states: s._multilevel_mcmc_sweep_parallel_dispatch(lstates, lsweep_states)
        return NestedBlockState._h_sweep_parallel_dispatch(states, sweeps, algo)

    @mcmc_sweep_wrap
    def gibbs_sweep(self, **kwargs):
        r"""Perform ``niter`` sweeps of a rejection-free Gibbs sampling MCMC
        to sample network partitions.

        The arguments accepted are the same as in
        :meth:`graph_tool.inference.BlockState.gibbs_sweep`.

        .. warning::

           This function performs ``niter`` sweeps at each hierarchical level
           once. This means that in order for the chain to equilibrate, we need
           to call this function several times, i.e. it is not enough to call
           it once with a large value of ``niter``.

        """
        return self._h_sweep(lambda s, **a: s.gibbs_sweep(**a),
                             **kwargs)

    def _gibbs_sweep_parallel_dispatch(states, sweeps):
        algo = lambda s, lstates, lsweep_states: s._gibbs_sweep_parallel_dispatch(lstates, lsweep_states)
        return NestedBlockState._h_sweep_parallel_dispatch(states, sweeps, algo)

    @mcmc_sweep_wrap
    def multicanonical_sweep(self, m_state, **kwargs):
        r"""Perform ``niter`` sweeps of a non-Markovian multicanonical sampling using the
        Wang-Landau algorithm.

        The arguments accepted are the same as in
        :meth:`graph_tool.inference.BlockState.multicanonical_sweep`.
        """

        def sweep(s, **kwargs):
            S = 0
            for l, state in enumerate(self.levels):
                if s is state:
                    continue
                S += self.level_entropy(l)
            return s.multicanonical_sweep(m_state, entropy_offset=S, **kwargs)

        return self._h_sweep(sweep)

    def collect_partition_histogram(self, h=None, update=1):
        r"""Collect a histogram of partitions.

        This should be called multiple times, e.g. after repeated runs of the
        :meth:`graph_tool.inference.NestedBlockState.mcmc_sweep` function.

        Parameters
        ----------
        h : :class:`~graph_tool.inference.PartitionHist` (optional, default: ``None``)
            Partition histogram. If not provided, an empty histogram will be created.
        update : float (optional, default: ``1``)
            Each call increases the current count by the amount given by this
            parameter.

        Returns
        -------
        h : :class:`~graph_tool.inference.PartitionHist` (optional, default: ``None``)
            Updated Partition histogram.

        """

        if h is None:
            h = PartitionHist()
        bs = [_prop("v", state.g, state.b) for state in self.levels]
        libinference.collect_hierarchical_partitions(bs, h, update)
        return h

    def draw(self, **kwargs):
        r"""Convenience wrapper to :func:`~graph_tool.draw.draw_hierarchy` that
        draws the hierarchical state."""
        import graph_tool.draw
        return graph_tool.draw.draw_hierarchy(self,
                                              vcmap=kwargs.pop("vcmap",
                                                               graph_tool.draw.default_cm),
                                              **kwargs)

def get_hierarchy_tree(state, empty_branches=False):
    r"""Obtain the nested hierarchical levels as a tree.

    This transforms a :class:`~graph_tool.inference.NestedBlockState` instance
    into a single :class:`~graph_tool.Graph` instance containing the hierarchy
    tree.

    Parameters
    ----------
    state : :class:`~graph_tool.inference.NestedBlockState`
       Nested block model state.
    empty_branches : ``bool`` (optional, default: ``False``)
       If ``empty_branches == False``, dangling branches at the upper layers
       will be pruned.

    Returns
    -------

    tree : :class:`~graph_tool.Graph`
       A directed graph, where vertices are blocks, and a directed edge points
       to an upper to a lower level in the hierarchy.
    label : :class:`~graph_tool.VertexPropertyMap`
       A vertex property map containing the block label for each node.
    order : :class:`~graph_tool.VertexPropertyMap`
       A vertex property map containing the relative ordering of each layer
       according to the total degree of the groups at the specific levels.
    """

    bstack = state.get_bstack()

    g = bstack[0]
    b = g.vp["b"]
    bstack = bstack[1:]

    if bstack[-1].num_vertices() > 1:
        bg = Graph(directed=g.is_directed())
        bg.add_vertex()
        e = bg.add_edge(0, 0)
        bg.vp.count = bg.new_vp("int", 1)
        bg.ep.count = bg.new_ep("int", g.ep.count.fa.sum())
        bg.vp.b = bg.new_vp("int", 0)
        bstack.append(bg)

    t = Graph()

    if g.get_vertex_filter()[0] is None:
        t.add_vertex(g.num_vertices())
    else:
        t.add_vertex(g.num_vertices(ignore_filter=True))
        filt = g.get_vertex_filter()
        t.set_vertex_filter(t.own_property(filt[0].copy()),
                            filt[1])
    label = t.vertex_index.copy("int")

    order = t.own_property(g.degree_property_map("total").copy())
    t_vertices = list(t.vertices())

    last_pos = 0
    for l, s in enumerate(bstack):
        pos = t.num_vertices()
        if s.num_vertices() > 1:
            t_vertices.extend(t.add_vertex(s.num_vertices()))
        else:
            t_vertices.append(t.add_vertex(s.num_vertices()))
        label.a[-s.num_vertices():] = arange(s.num_vertices())

        # relative ordering based on total degree
        count = s.ep["count"].copy("double")
        for e in s.edges():
            if e.source() == e.target():
                count[e] /= 2
        vs = []
        pvs = {}
        for vi in range(pos, t.num_vertices()):
            vs.append(t_vertices[vi])
            pvs[vs[-1]] = vi - pos
        vs = sorted(vs, key=lambda v: (s.vertex(pvs[v]).out_degree(count) +
                                       s.vertex(pvs[v]).in_degree(count)))
        for vi, v in enumerate(vs):
            order[v] = vi

        for vi, v in enumerate(g.vertices()):
            w = t_vertices[vi + last_pos]
            if s.num_vertices() == 1:
                u = t_vertices[pos]
            else:
                u = t_vertices[b[v] + pos]
            t.add_edge(u, w)

        last_pos = pos
        g = s
        if empty_branches:
            if g.num_vertices() == 1:
                break
        else:
            if g.vp.count.fa.sum() == 1:
                break
        b = g.vp["b"]

    if not empty_branches:
        vmask = t.new_vertex_property("bool", True)
        t = GraphView(t, vfilt=vmask)
        vmask = t.get_vertex_filter()[0]
        N = t.num_vertices()
        for vi, v in enumerate(list(t.vertices())):
            if vi < state.g.num_vertices():
                continue
            if v.out_degree() == 0:
                vmask[v] = False
        label = t.own_property(label)
        order = t.own_property(order)

    return t, label, order

from . minimize import *
