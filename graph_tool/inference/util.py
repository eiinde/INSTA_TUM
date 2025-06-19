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

import scipy.special
from numpy import *

from .. import PropertyMap

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

class DictState(dict):
    """Dictionary with (key,value) pairs accessible via attributes."""
    def __init__(self, d):
        self.update(d)
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, val):
        self[attr] = val

def dmask(d, ks):
    """Copy dictionary ``d`` and remove key list ``ks``."""
    d = d.copy()
    for k in ks:
        if k in d:
            del d[k]
    return d

def dselect(d, ks):
    """Copy dictionary ``d`` considering only key list ``ks``."""
    nd = {}
    for k in ks:
        if k in d:
            nd[k] = d[k]
    return nd

def check_verbose(verbose):
    if isinstance(verbose, tuple):
        return verbose[0] != False
    return verbose != False


def verbose_pad(verbose):
    if isinstance(verbose, tuple):
        return verbose_pad(verbose[1])
    if verbose == True:
        return ""
    return verbose

def verbose_push(verbose, push):
    if check_verbose(verbose):
        if isinstance(verbose, tuple):
            return (verbose[0] - 1, verbose_push(verbose[1], push))
        if isinstance(verbose, bool):
            return push
        return verbose + push
    return False

def lbinom(n, k):
    """Return log of binom(n, k)."""
    return (scipy.special.gammaln(float(n + 1)) -
            scipy.special.gammaln(float(n - k + 1)) -
            scipy.special.gammaln(float(k + 1)))

def lbinom_careful(n, k):
    return libinference.lbinom_careful(n, k)

def lbinom_fast(n, k):
    return libinference.lbinom_fast(n, k)

def partition_entropy(B, N, nr=None):
    if nr is None:
        S = N * log(B) + log1p(-(1 - 1./B) ** N)
    else:
        S = lbinom(N - 1, B - 1)
        S += (scipy.special.gammaln(N + 1) -
              scipy.special.gammaln(nr + 1).sum())
    return S

def pmap(prop, value_map):
    """Maps all the values of `prop` to the values given by `value_map` in-place
    according to: ``prop[i] = value_map[prop[i]]``."""
    if isinstance(prop, PropertyMap):
        a = prop.fa
    else:
        a = prop
    if len(a) == 0:
        return
    if isinstance(value_map, PropertyMap):
        value_map = value_map.fa
    if a.max() >= len(value_map):
        raise ValueError("value_map is not large enough!" +
                         " max val: %s, map size: %s" % (a.max(),
                                                         len(value_map)))
    libinference.vector_map(a, value_map)
    if isinstance(prop, PropertyMap):
        prop.fa = a

def reverse_map(prop, value_map):
    """Modify `value_map` such that the positions indexed by the values in `prop`
    correspond to their index in `prop`."""
    if isinstance(prop, PropertyMap):
        prop = prop.fa
    if isinstance(value_map, PropertyMap):
        a = value_map.fa
    else:
        a = value_map
    if prop.max() >= len(a):
        raise ValueError("value map is not large enough!" +
                         " max val: %s, map size: %s" % (prop.max(), len(a)))
    if prop.dtype != a.dtype:
        prop = array(prop, dtype=a.dtype)
    libinference.vector_rmap(prop, a)
    if isinstance(value_map, PropertyMap):
        value_map.fa = a

def contiguous_map(prop):
    """Remap the values of ``prop`` in the contiguous range :math:`[0, N-1]`."""
    prop = prop.copy()
    if isinstance(prop, PropertyMap):
        a = prop.fa
    else:
        a = prop
    libinference.vector_contiguous_map(a)
    if isinstance(prop, PropertyMap):
        prop.fa = a
    return prop

def nested_contiguous_map(bs):
    """Remap the values of the nested partition ``bs`` in the contiguous range :math:`[0, N_l-1]` for each level :math:`l`."""
    cs = [b for b in bs]
    for i, b in enumerate(cs):
        c = contiguous_map(b)
        cs[i] = c
        if i == len(cs) - 1:
            break
        nb = zeros(c.max() + 1, dtype=c.dtype)
        for j, r in enumerate(c):
            nb[r] = cs[i+1][b[j]]
        cs[i+1] = nb
    return cs
