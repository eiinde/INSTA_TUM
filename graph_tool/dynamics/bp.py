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

from .. import _prop, Graph, GraphView, _get_rng, EdgePropertyMap, \
    VertexPropertyMap, group_vector_property, _check_prop_vector, \
    _check_prop_scalar, _parallel

from collections.abc import Iterable
from abc import ABC, abstractmethod
import numpy
import numpy.random

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_dynamics as lib_dynamics")

class BPBaseState(ABC):
    """Base class for belief propagation (BP) states."""

    @abstractmethod
    def __init__(self):
        pass

    def copy(self):
        """Return a copy of the state."""
        return type(self)(**self.__getstate__())

    @abstractmethod
    def __getstate__(self):
        pass

    def __setstate__(self, state):
        self.__init__(**state, converge=False)


    @_parallel
    def iterate(self, niter=1, parallel=True, update_marginals=True):
        """Updates meassages synchronously (or asyncrhonously if
        ``parallel=False``), `niter` number of times. This function returns the
        interation delta of the last iteration.

        If ``update_marignals=True``, this function calls
        :meth:`~BPBaseState.update_marginals()` at the end.

        @parallel@
        """
        if parallel:
            delta = self._state.iterate_parallel(self.g._Graph__graph, niter)
        else:
            delta = self._state.iterate(self.g._Graph__graph, niter)
        if update_marginals:
            self.update_marginals()
        return delta

    def converge(self, epsilon=1e-8, max_niter=1000, update_marginals=True,
                 **kwargs):
        """Calls :meth:`~BPBaseState.iterate()` until delta falls below
        ``epsilon`` or the number of iterations exceeds ``max_niter``.

        If ``update_marignals=True``, this function calls
        :meth:`~BPBaseState.update_marginals()` at the end.

        The remaining keyword arguments are passed to
        :meth:`~BPBaseState.iterate()`.
        """
        delta = epsilon + 1
        niter = 0
        while delta > epsilon and niter < max_niter:
            delta = self.iterate(**kwargs)
            niter += kwargs.get("niter", 1)
        self.update_marginals()
        return niter, delta

    def update_marginals(self):
        """Update the node marginals from the current messages."""
        return self._state.update_marginals(self.g._Graph__graph)

    def log_Z(self):
        """Obtains the log-partition function from the current messages."""
        return self._state.log_Z(self.g._Graph__graph)

    def energy(self, s):
        """Obtains the energy (Hamiltonean) of state ``s`` (a
        :class:`~graph_tool.VertexPropertyMap`).

        If ``s`` is vector valued, it's assumed to correspond to multiple
        states, and the total energy sum is returned.
        """
        if "vector" in s.value_type():
            _check_prop_vector(s, scalar=True)
            return self._state.energies(self.g._Graph__graph,
                                        _prop("v", self.g, s))
        else:
            _check_prop_scalar(s)
            return self._state.energy(self.g._Graph__graph,
                                      _prop("v", self.g, s))

    def log_prob(self, s):
        """Obtains the log-probability of state ``s`` (a
        :class:`~graph_tool.VertexPropertyMap`).

        If ``s`` is vector valued, it's assumed to correspond to multiple
        states, and the total log-probability sum is returned.
        """
        H = BPBaseState.energy(self, s)
        lZ = self.log_Z()
        if "vector" in s.value_type():
            _check_prop_vector(s, scalar=True)
            return -H - lZ * len(s[next(self.g.vertices())])
        else:
            return -H - lZ

    def marginal_log_prob(self, s):
        """Obtains the marginal log-probability of state ``s`` (a
        :class:`~graph_tool.VertexPropertyMap`).

        If ``s`` is vector valued, it's assumed to correspond to multiple
        states, and the total marginal log-probability sum is returned.
        """
        if "vector" in s.value_type():
            _check_prop_vector(s, scalar=True)
            return self._state.marginal_lprobs(self.g._Graph__graph,
                                               _prop("v", self.g, s))
        else:
            _check_prop_scalar(s)
            return self._state.marginal_lprob(self.g._Graph__graph,
                                              _prop("v", self.g, s))

    def sample(self, update_marginals=True, val_type="int"):
        """Samples a state from the marignal distribution. This functio returns
        a :class:`~graph_tool.VertexPropertyMap` of type given by ``val_type``.

        If ``update_marignals=True``, this function calls
        :meth:`~BPBaseState.update_marginals()` before sampling.
        """
        if update_marginals:
            self.update_marginals()
        s = self.g.new_vp(val_type)
        self._state.sample(self.g._Graph__graph, _prop("v", self.g, s),
                           _get_rng())
        return s

class GenPottsBPState(BPBaseState):
    def __init__(self, g, f, x=1, theta=0, em=None, vm=None, marginal_init=False,
                 frozen=None, converge=True):
        r"""Belief-propagtion equations for a genralized Potts model.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics.
        f : :class:`~numpy.ndarray` or list of list
           :math:`q\times q` 2D symmetric with iteraction energies between the
           :math:`q` spin values.
        x : ``float`` or :class:`~graph_tool.EdgePropertyMap` (optional, default: ``1``)
            Edge coupling weights. If a :class:`~graph_tool.EdgePropertyMap` is
            given, it needs to be of type ``double``. If a scalar is given, this
            will be determine the value for every edge.
        theta : ``float`` or iterable or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``0.``)
            Vertex fields. If :class:`~graph_tool.VertexPropertyMap`, this needs
            to be of type ``vector<double>``, containing :math:`q` field values
            for every node. If it's an iterable, it should contains :math:`q`
            field values, which are the same for every node. If a scalar is
            given, this will be determine the value for every field and vertex.
        em : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
            If provided, it should be an :class:`~graph_tool.EdgePropertyMap`
            of type ``vector<double>``, containing the edge messages.
        vm : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            If provided, it should be an :class:`~graph_tool.VertexPropertyMap`
            of type ``vector<double>``, containing the node marginals.
        marginal_init : ``boolean`` (optional, default: ``False``)
           If ``True``, the messages will be initialized from the node marginals.
        frozen : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            If provided, it should be an :class:`~graph_tool.VertexPropertyMap`
            of type ``bool``, where a value `True` means that a vertex is not
            a variable, but a fixed field.
        converge : ``boolean`` (optional, default: ``True``)
           If ``True``, the function :meth:`GenPottsBPState.converge()` will be
           called just after construction.

        Notes
        -----

        This implements BP equations [mezard_information_2009]_, for a
        generalized Potts model given by

        .. math::

           P(\boldsymbol s | \boldsymbol A, \boldsymbol x, \boldsymbol\theta)
           = \frac{\exp\left(\sum_{i<j}A_{ij}x_{ij}f_{s_i,s_j} + \sum_i\theta_{i,s_i}\right)}
           {Z(\boldsymbol A, \boldsymbol x, \boldsymbol\theta)}

        where :math:`Z(\boldsymbol A, \boldsymbol x, \boldsymbol\theta)` is the
        partition function.

        The BP equations consist in the Bethe approximation

        .. math::

             \log Z(\boldsymbol A, \boldsymbol x, \boldsymbol\theta) = \log Z_i
             - \sum_{i<j}A_{ij}\log Z_{ij}

        with :math:`Z_{ij}=Z_j/Z_{j\to i}=Z_i/Z_{i\to j}`, obtained from the
        message-passing equations

        .. math::

           P_{i\to j}(s_i) = \frac{e^{\theta_{i,s_i}}}{Z_{i\to j}}
           \prod_{k\in \partial i\setminus j}\sum_{s_k=1}^{q}P_{k\to i}(s_k)e^{x_{ik}f_{x_i,x_k}},

        where :math:`Z_{i\to j}` is a normalization constant. From these
        equations, the marginal node probabilities are similarly obtained:

        .. math::

           P_i(s_i) = \frac{e^{\theta_{i,s_i}}}{Z_i}
           \prod_{j\in \partial i}\sum_{s_j=1}^{q}P_{j\to i}(s_j)e^{x_{ij}f_{x_i,x_j}},

        Examples
        --------

        .. testsetup:: BPPotts

           gt.seed_rng(43)
           np.random.seed(43)

        .. doctest:: BPPotts

           >>> g = gt.GraphView(gt.collection.data["polblogs"].copy(), directed=False)
           >>> gt.remove_parallel_edges(g)
           >>> g = gt.extract_largest_component(g, prune=True)
           >>> state = gt.GenPottsBPState(g, f=array([[-1,  0,  1],
           ...                                        [ 0, -1,  1],
           ...                                        [ 1,  1, -1.25]])/20)
           >>> s = state.sample()
           >>> gt.graph_draw(g, g.vp.pos, vertex_fill_color=s,
           ...               output="bp-potts.svg")
           <...>

        .. figure:: bp-potts.svg
           :align: center
           :width: 80%

           Marginal sample of a 3-state Potts model.

        References
        ----------
        .. [mezard_information_2009] Marc Mézard, and Andrea Montanari,
           "Information, physics, and computation", Oxford University Press, 2009.
           https://web.stanford.edu/~montanar/RESEARCH/book.html

        """

        self.g = g
        self.f = numpy.asarray(f, dtype="float")
        if not isinstance(x, EdgePropertyMap):
            x = g.new_ep("double", val=x)
        elif x.value_type() != "double":
            x = x.copy("double")
        self.x = self.g.own_property(x)
        if not isinstance(theta, VertexPropertyMap):
            if isinstance(theta, Iterable):
                theta = g.new_vp("vector<double>", val=theta)
            else:
                theta = g.new_vp("vector<double>", val=[theta] * self.f.shape[0])
        elif theta.value_type() != "vector<double>":
            theta = theta.copy("vector<double>")
        self.theta = self.g.own_property(theta)
        if em is None:
            em = g.new_ep("vector<double>")
        self.em = em
        if vm is None:
            vm = g.new_vp("vector<double>")
        self.vm = self.g.own_property(vm)
        if frozen is None:
            frozen = g.new_vp("bool")
        elif frozen.value_type() != "bool":
            frozen = frozen.copy("bool")
        self.frozen = self.g.own_property(frozen)
        self._state = lib_dynamics.make_potts_bp_state(self.g._Graph__graph,
                                                       self.f,
                                                       _prop("e", g, self.x),
                                                       _prop("v", g, self.theta),
                                                       _prop("e", g, self.em),
                                                       _prop("v", g, self.vm),
                                                       marginal_init,
                                                       _prop("v", g, self.frozen),
                                                       _get_rng())
        if converge:
            self.converge()

    def __getstate__(self):
        return dict(g=self.g, f=self.f, x=self.x, theta=self.theta, em=self.em,
                    vm=self.vm, frozen=self.frozen)

class IsingBPState(GenPottsBPState):
    def __init__(self, g, x=1, theta=0, em=None, vm=None, marginal_init=False,
                 frozen=None, has_zero=False, converge=True):
        r"""Belief-propagation equations for the Ising model.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics.
        x : ``float`` or :class:`~graph_tool.EdgePropertyMap` (optional, default: ``1``)
            Edge coupling weights. If a :class:`~graph_tool.EdgePropertyMap` is
            given, it needs to be of type ``double``. If a scalar is given, this
            will be determine the value for every edge.
        theta : ``float`` or iterable or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``0.``)
            Vertex fields. If :class:`~graph_tool.VertexPropertyMap`, this needs
            to be of type ``double``. If a scalar is given, this will be
            determine the value for every vertex.
        em : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
            If provided, it should be an :class:`~graph_tool.EdgePropertyMap`
            of type ``vector<double>``, containing the edge messages.
        vm : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            If provided, it should be an :class:`~graph_tool.VertexPropertyMap`
            of type ``vector<double>``, containing the node marginals.
        marginal_init : ``boolean`` (optional, default: ``False``)
           If ``True``, the messages will be initialized from the node marginals.
        frozen : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            If provided, it should be an :class:`~graph_tool.VertexPropertyMap`
            of type ``bool``, where a value `True` means that a vertex is not
            a variable, but a fixed field.
        converge : ``boolean`` (optional, default: ``True``)
           If ``True``, the function :meth:`GenPottsBPState.converge()` will be
           called just after construction.

        Notes
        -----

        This implements BP equations [mezard_information_2009]_ for the Ising
        model given by

        .. math::

           P(\boldsymbol \sigma | \boldsymbol A, \boldsymbol x, \boldsymbol\theta)
           = \frac{\exp\left(\sum_{i<j}A_{ij}x_{ij}\sigma_i\sigma_j + \sum_i\theta_{i}\sigma_i\right)}
           {Z(\boldsymbol A, \boldsymbol x, \boldsymbol\theta)}

        where :math:`\sigma_i\in\{-1,1\}` and :math:`Z(\boldsymbol A,
        \boldsymbol x, \boldsymbol\theta)` is the partition function. This is
        equivalent to a gereralized Potts model with :math:`s_i=(\sigma_i +
        1)/2` and :math:`f_{rs} = -(2r-1)(2s-1)`. See
        :class:`~graph_tool.dynamics.GenPottsBPState` for more details.

        If ``has_zero == True``, then it is assumed :math:`\sigma_i\in\{-1,0,1\}`.

        Examples
        --------

        .. testsetup:: BPIsing

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: BPIsing

           >>> g = gt.GraphView(gt.collection.data["polblogs"].copy(), directed=False)
           >>> gt.remove_parallel_edges(g)
           >>> g = gt.extract_largest_component(g, prune=True)
           >>> state = gt.IsingBPState(g, x=1/20,
           ...                         theta=g.vp.value.t(lambda x: np.arctanh((2*x-1)*.9)))
           >>> s = state.sample()
           >>> gt.graph_draw(g, g.vp.pos, vertex_fill_color=s,
           ...               output="bp-ising.svg")
           <...>

        .. figure:: bp-ising.svg
           :align: center
           :width: 80%

           Marginal sample of an Ising model.

        References
        ----------
        .. [mezard_information_2009] Marc Mézard, and Andrea Montanari,
           "Information, physics, and computation", Oxford University Press, 2009.
           https://web.stanford.edu/~montanar/RESEARCH/book.html

        """
        if not has_zero:
            f = [[-1,  1],
                 [ 1, -1]]
        else:
            f = [[-1,  0,  1],
                 [ 0,  0,  0],
                 [ 1,  0, -1]]
        if not isinstance(theta, VertexPropertyMap):
            if not has_zero:
                theta = g.new_vp("vector<double>", val=[theta, -theta])
            else:
                theta = g.new_vp("vector<double>", val=[theta, 0, -theta])
        elif theta.value_type() == "double":
            ntheta = theta.copy()
            ntheta.a *= -1
            if not has_zero:
                theta = group_vector_property([theta, ntheta])
            else:
                zero = g.new_vp("double")
                theta = group_vector_property([ntheta, zero, theta])
        elif theta.value_type() != "vector<double>":
            theta = theta.copy("vector<double>")
        self.has_zero = has_zero
        super().__init__(g=g, f=f, x=x, theta=theta, em=em, vm=vm,
                         marginal_init=marginal_init, frozen=frozen,
                         converge=converge)

    def __getstate__(self):
        return dict(g=self.g, x=self.x, theta=self.theta, em=self.em,
                    vm=self.vm, frozen=self.frozen, has_zero=self.has_zero)

    def from_spin(self, s):
        s = s.copy()
        f = 2 if not self.has_zero else 1
        if "vector" in s.value_type():
            for v in self.g.vertices():
                s[v].a = (s[v].a + 1)/f
        else:
            s.fa = (s.fa + 1)/f
        return s

    def to_spin(self, s):
        s = s.copy()
        f = 2 if not self.has_zero else 1
        if "vector" in s.value_type():
            for v in self.g.vertices():
                s[v].a = f * s[v].a - 1
        else:
            s.fa = f * s.fa - 1
        return s

    def energy(self, s):
        return GenPottsBPState.energy(self, self.from_spin(s))

    def log_prob(self, s):
        return GenPottsBPState.log_prob(self, self.from_spin(s))

    def marginal_log_prob(self, s):
        return GenPottsBPState.marginal_log_prob(self, self.from_spin(s))

    def sample(self, update_marginals=True, val_type="int"):
        s = GenPottsBPState.sample(self, update_marginals=update_marginals,
                                   val_type=val_type)
        return self.to_spin(s)

class NormalBPState(BPBaseState):
    def __init__(self, g, x=1, mu=0, theta=1, em_m=None, em_s=None, vm_m=None,
                 vm_s=None, marginal_init=False, frozen=None, converge=True):
        r"""Belief-propagation equations for the multivariate Normal distribution.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics.
        x : ``float`` or :class:`~graph_tool.EdgePropertyMap` (optional, default: ``1.``)
            Inverse covariance couplings. If a :class:`~graph_tool.EdgePropertyMap` is
            given, it needs to be of type ``double``. If a scalar is given, this
            will be determine the value for every edge.
        mu : ``float`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``0.``)
            Node means. If a :class:`~graph_tool.VertexPropertyMap` is given, it
            needs to be of type ``double``. If a scalar is given, this will be
            determine the value for every vertex.
        theta : ``float`` or iterable or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``1.``)
            Diagonal of the inverse covariance matrix. If
            :class:`~graph_tool.VertexPropertyMap`, this needs to be of type
            ``double``. If a scalar is given, this will be determine the value
            for every vertex.
        em_m : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
            If provided, it should be an :class:`~graph_tool.EdgePropertyMap` of
            type ``vector<double>``, containing the edge messages for the means.
        em_s : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
            If provided, it should be an :class:`~graph_tool.EdgePropertyMap` of
            type ``vector<double>``, containing the edge messages for the
            variances.
        vm_m : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            If provided, it should be an :class:`~graph_tool.VertexPropertyMap`
            of type ``vector<double>``, containing the node marginal means.
        vm_s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            If provided, it should be an :class:`~graph_tool.VertexPropertyMap`
            of type ``vector<double>``, containing the node marginal variances.
        marginal_init : ``boolean`` (optional, default: ``False``)
           If ``True``, the messages will be initialized from the node marginals.
        frozen : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            If provided, it should be an :class:`~graph_tool.VertexPropertyMap`
            of type ``bool``, where a value `True` means that a vertex is not
            a variable, but a fixed field.
        converge : ``boolean`` (optional, default: ``True``)
           If ``True``, the function :meth:`GenPottsBPState.converge()` will be
           called just after construction.

        Notes
        -----

        This implements BP equations [mezard_information_2009]_ for the
        mutivariate Normal distribution given by

        .. math::

           P(\boldsymbol s | \boldsymbol A, \boldsymbol x, \boldsymbol \mu \boldsymbol\theta)
           = \frac{\exp\left(-\frac{1}{2}(\boldsymbol s-\boldsymbol\mu)^{\intercal} \boldsymbol X (\boldsymbol s - \boldsymbol\mu)\right)}
           {Z(\boldsymbol X)}

        where :math:`X_{ij}=A_{ij}x_{ij}` for :math:`i\neq j`,
        :math:`X_{ii}=\theta_i`, and :math:`Z(\boldsymbol X) =
        (2\pi)^{N/2}\left|\boldsymbol X\right|^{-1/2}`.

        The BP equations consist in the Bethe approximation

        .. math::

             \log Z(\boldsymbol X) = \log Z_i
             - \sum_{i<j}A_{ij}\log Z_{ij}

        with :math:`Z_{ij}=Z_j/Z_{j\to i}=Z_i/Z_{i\to j}`, obtained from the
        message-passing equations

        .. math::

           \begin{aligned}
           m_{i\to j} &= \frac{\sum_{k\in \partial i\setminus j}A_{ik}x_{ik}m_{k\to i} - \mu_i}
           {\theta_i - \sum_{k\in \partial i\setminus j}A_{ik}x_{ik}^2\sigma_{k\to i}^2},\\
           \sigma_{i\to j}^2 &= \frac{1}{\theta_i - \sum_{k\in \partial i\setminus j}A_{ik}x_{ik}^2\sigma_{k\to i}^2},
           \end{aligned}

        with

        .. math::
           \begin{aligned}
           \log Z_{i\to j} &= \frac{\beta_{i\to j}^2}{4\alpha_{i\to j}} - \frac{1}{2}\log\alpha_{i\to j} + \frac{1}{2}\log\pi\\
           \log Z_{i} &= \frac{\beta_{i}^2}{4\alpha_{i}} - \frac{1}{2}\log\alpha_{i} + \frac{1}{2}\log\pi
           \end{aligned}

        where

        .. math::
           \begin{aligned}
           \alpha_{i\to j} &= \frac{\theta_i - \sum_{k\in \partial i\setminus j}A_{ik}x_{ik}^2\sigma_{k\to i}^2}{2}\\
           \beta_{i\to j} &= \sum_{k\in \partial i\setminus j}A_{ik}x_{ik}m_{k\to i} - \mu_i\\
           \alpha_{i} &= \frac{\theta_i - \sum_{j\in \partial i}A_{ij}x_{ij}^2\sigma_{j\to i}^2}{2}\\
           \beta_{i} &= \sum_{j\in \partial i}A_{ij}x_{ij}m_{j\to i} - \mu_i.
           \end{aligned}

        From these equations, the marginal node probability densities are normal
        distributions with mean and variance given by

        .. math::

           \begin{aligned}
           m_i &= \frac{\sum_{j}A_{ij}x_{ij}m_{j\to i} - \mu_i}
           {\theta_i - \sum_{j}A_{ij}x_{ij}^2\sigma_{j\to i}^2},\\
           \sigma_i^2 &= \frac{1}{\theta_i - \sum_{j}A_{ij}x_{ij}^2\sigma_{j\to i}^2}.
           \end{aligned}

        Examples
        --------

        .. testsetup:: BPnormal

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: BPnormal

           >>> g = gt.GraphView(gt.collection.data["polblogs"].copy(), directed=False)
           >>> gt.remove_parallel_edges(g)
           >>> g = gt.extract_largest_component(g, prune=True)
           >>> state = gt.NormalBPState(g, x=1/200, mu=g.vp.value.t(lambda x: arctanh((2*x-1)*.9)))
           >>> s = state.sample()
           >>> gt.graph_draw(g, g.vp.pos, vertex_fill_color=s,
           ...               output="bp-normal.svg")
           <...>

        .. figure:: bp-normal.svg
           :align: center
           :width: 80%

           Marginal sample of a multivariate normal distribution.

        References
        ----------
        .. [mezard_information_2009] Marc Mézard, and Andrea Montanari,
           "Information, physics, and computation", Oxford University Press, 2009.
           https://web.stanford.edu/~montanar/RESEARCH/book.html

        """

        self.g = g
        if not isinstance(x, EdgePropertyMap):
            x = g.new_ep("double", val=x)
        elif x.value_type() != "double":
            x = x.copy("double")
        self.x = self.g.own_property(x)
        if not isinstance(mu, VertexPropertyMap):
            mu = g.new_vp("double", val=mu)
        elif mu.value_type() != "double":
            mu = theta.copy("double")
        self.mu = self.g.own_property(mu)
        if not isinstance(theta, VertexPropertyMap):
            theta = g.new_vp("double", val=theta)
        elif theta.value_type() != "double":
            theta = theta.copy("double")
        self.theta = self.g.own_property(theta)
        if em_m is None:
            em_m = g.new_ep("vector<double>")
        if em_s is None:
            em_s = g.new_ep("vector<double>")
        self.em_m = self.g.own_property(em_m)
        self.em_s = self.g.own_property(em_s)
        if vm_m is None:
            vm_m = g.new_vp("double")
        if vm_s is None:
            vm_s = g.new_vp("double")
        self.vm_m = self.g.own_property(vm_m)
        self.vm_s = self.g.own_property(vm_s)
        if frozen is None:
            frozen = g.new_vp("bool")
        elif frozen.value_type() != "bool":
            frozen = frozen.copy("bool")
        self.frozen = self.g.own_property(frozen)
        self._state = lib_dynamics.make_normal_bp_state(self.g._Graph__graph,
                                                        _prop("e", g, self.x),
                                                        _prop("v", g, self.mu),
                                                        _prop("v", g, self.theta),
                                                        _prop("e", g, self.em_m),
                                                        _prop("e", g, self.em_s),
                                                        _prop("v", g, self.vm_m),
                                                        _prop("v", g, self.vm_s),
                                                        marginal_init,
                                                        _prop("v", g, self.frozen),
                                                        _get_rng())
        if converge:
            self.converge()

    def __getstate__(self):
        return dict(g=self.g, x=self.x, mu=self.mu, theta=self.theta,
                    em_m=self.em_m, em_s=self.em_s, vm_m=self.vm_m,
                    vm_s=self.vm_s, frozen=self.frozen)

    def sample(self, update_marginals=True):
        return BPBaseState.sample(self, update_marginals=update_marginals,
                                  val_type="double")
