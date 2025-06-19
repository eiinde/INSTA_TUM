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

"""
``graph_tool``
==============

This is the core module providing the fundamental data structures and functions.

Fundamental classes
+++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary
   :template: graph.rst

   Graph
   GraphView
   Vertex
   Edge

Property Maps
+++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary
   :template: graph.rst

   PropertyMap
   VertexPropertyMap
   EdgePropertyMap
   GraphPropertyMap
   PropertyArray
   group_vector_property
   ungroup_vector_property
   map_property_values
   infect_vertex_property
   edge_endpoint_property
   incident_edges_op
   perfect_prop_hash
   value_types

Graph IO
++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   load_graph
   load_graph_from_csv

OpenMP configuration
++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   openmp_enabled
   openmp_get_num_threads
   openmp_set_num_threads
   openmp_get_schedule
   openmp_set_schedule
   openmp_get_thresh
   openmp_set_thresh

System information
++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   show_config

"""

__author__ = "Tiago de Paula Peixoto <tiago@skewed.de>"
__copyright__ = "Copyright 2006-2024 Tiago de Paula Peixoto"
__license__ = "LGPL version 3 or above"
__URL__ = "https://graph-tool.skewed.de"

# set default OMP_WAIT_POLICY to "passive"
import os
if "OMP_WAIT_POLICY" not in os.environ:
    os.environ["OMP_WAIT_POLICY"] = "passive"

# import numpy and scipy before everything to avoid weird segmentation faults
# depending on the order things are imported.

import numpy
import numpy.ma
import scipy
import scipy.sparse
import numpy.random

from .dl_import import *
dl_import("from . import libgraph_tool_core as libcore")
__version__ = libcore.mod_info().version

from . import gt_io  # sets up libcore io routines

import sys
import re
import gzip
import bz2
try:
    import lzma
except ImportError:
    pass
try:
    import zstandard
except ImportError:
    pass
import weakref
import copy
import textwrap
import io
import collections.abc
import itertools
import csv
import contextlib
import threading

from .decorators import _require, _limit_args, _copy_func, _parallel

__all__ = ["Graph", "GraphView", "Vertex", "Edge", "VertexBase", "EdgeBase",
           "Vector_bool", "Vector_int16_t", "Vector_int32_t", "Vector_int64_t",
           "Vector_double", "Vector_long_double", "Vector_string",
           "Vector_size_t", "Vector_cdouble", "value_types", "load_graph",
           "load_graph_from_csv", "VertexPropertyMap", "EdgePropertyMap",
           "GraphPropertyMap", "PropertyMap", "PropertyArray",
           "group_vector_property", "ungroup_vector_property",
           "map_property_values", "infect_vertex_property",
           "edge_endpoint_property", "incident_edges_op", "perfect_prop_hash",
           "seed_rng", "show_config", "openmp_enabled",
           "openmp_get_num_threads", "openmp_set_num_threads",
           "openmp_get_schedule", "openmp_set_schedule", "openmp_get_thresh",
           "openmp_set_thresh", "openmp_context", "__author__", "__copyright__",
           "__URL__", "__version__"]

# this is rather pointless, but it works around a sphinx bug
graph_tool = sys.modules[__name__]

################################################################################
# Utility functions
################################################################################


def _prop(t, g, prop):
    """Return either a property map, or an internal property map with a given
    name."""
    if isinstance(prop, str):
        try:
            pmap = g.properties[(t, prop)]
        except KeyError:
            raise KeyError("no internal %s property named: %s" %\
                           ("vertex" if t == "v" else \
                            ("edge" if t == "e" else "graph"), prop))
    else:
        pmap = prop
    if pmap is None:
        return libcore.any()
    if t != prop.key_type():
        names = {'e': 'edge', 'v': 'vertex', 'g': 'graph'}
        raise ValueError("Expected '%s' property map, got '%s'" %
                         (names[t], names[prop.key_type()]))
    u = pmap.get_graph()
    if u is None:
        raise ValueError("Received orphaned property map")
    if g.base is not u.base:
        raise ValueError("Received property map for graph %s (base: %s), expected: %s (base: %s)" %
                         (str(g), str(g.base), str(u), str(u.base)))
    return pmap._get_any()


def _degree(g, name):
    """Retrieve the degree type from string, or returns the corresponding
    property map."""
    deg = name
    if name == "in-degree" or name == "in":
        deg = libcore.Degree.In
    elif name == "out-degree" or name == "out":
        deg = libcore.Degree.Out
    elif name == "total-degree" or name == "total":
        deg = libcore.Degree.Total
    else:
        deg = _prop("v", g, deg)
    return deg


def _type_alias(type_name):
    alias = {"int8_t": "bool",
             "boolean": "bool",
             "short": "int16_t",
             "int": "int32_t",
             "unsigned int": "int32_t",
             "long": "int64_t",
             "long long": "int64_t",
             "unsigned long": "int64_t",
             "object": "python::object",
             "float": "double"}
    if type_name in alias:
        return alias[type_name]
    if type_name in value_types():
        return type_name
    ma = re.compile(r"vector<(.*)>").match(type_name)
    if ma:
        t = ma.group(1)
        if t in alias:
            return "vector<%s>" % alias[t]
    raise ValueError("invalid property value type: " + type_name)


def _python_type(type_name):
    type_name = _type_alias(type_name)
    if "vector" in type_name:
        ma = re.compile(r"vector<(.*)>").match(type_name)
        t = ma.group(1)
        return list, _python_type(t)
    if "int" in type_name:
        return int
    if type_name == "bool":
        return bool
    if "double" in type_name:
        return float
    if type_name == "string":
        return str
    return object

def _gt_type(obj):
    if isinstance(obj, numpy.dtype):
        t = obj.type
    else:
        t = type(obj)
    if issubclass(t, (numpy.int16, numpy.uint16, numpy.int8, numpy.uint8)):
        return "int16_t"
    if issubclass(t, (int, numpy.int32, numpy.uint32)):
        return "int32_t"
    if issubclass(t, (numpy.longlong, numpy.uint64, numpy.int64)):
        return "int64_t"
    if issubclass(t, (float, numpy.float16, numpy.float32, numpy.float64)):
        return "double"
    if issubclass(t, numpy.float128):
        return "long double"
    if issubclass(t, str):
        return "string"
    if issubclass(t, bool):
        return "bool"
    if issubclass(t, (list, numpy.ndarray)):
        return "vector<%s>" % _gt_type(obj[0])
    return "object"

def _converter(val_type):
    # attempt to convert to a compatible python type. This is useful,
    # for instance, when dealing with numpy types.
    vtype = _python_type(val_type)
    if type(vtype) is tuple:
        def convert(val):
            return [vtype[1](x) for x in val]
    elif vtype is object:
        def convert(val):
            return val
    elif vtype is str:
        return str
    else:
        def convert(val):
            return vtype(val)
    return convert

def show_config():
    """Show ``graph_tool`` build configuration."""
    info = libcore.mod_info()
    print("version:", info.version)
    print("gcc version:", info.gcc_version)
    print("compilation flags:", info.cxxflags)
    print("install prefix:", info.install_prefix)
    print("python dir:", info.python_dir)
    print("graph filtering:", libcore.graph_filtering_enabled())
    print("openmp:", libcore.openmp_enabled())
    print("uname:", " ".join(os.uname()))

def terminal_size():
    try:
        import fcntl, termios, struct
        h, w, hp, wp = struct.unpack('HHHH',
            fcntl.ioctl(0, termios.TIOCGWINSZ,
            struct.pack('HHHH', 0, 0, 0, 0)))
    except IOError:
        w, h = 80, 100
    return w, h

try:
    libcore.mod_info("wrong")
except BaseException as e:
    ArgumentError = type(e)

################################################################################
# Property Maps
################################################################################

class PropertyMap(object):
    """This base class provides a mapping from vertices, edges or whole graphs to
    arbitrary properties.

    See :ref:`sec_property_maps` for more details.

    The possible property value types are listed below.

    .. table::

        =======================     ======================
         Type name                  Alias
        =======================     ======================
        ``bool``                    ``uint8_t``
        ``int16_t``                 ``short``
        ``int32_t``                 ``int``
        ``int64_t``                 ``long``, ``long long``
        ``double``                  ``float``
        ``long double``
        ``string``
        ``vector<bool>``            ``vector<uint8_t>``
        ``vector<int16_t>``         ``short``
        ``vector<int32_t>``         ``vector<int>``
        ``vector<int64_t>``         ``vector<long>``, ``vector<long long>``
        ``vector<double>``          ``vector<float>``
        ``vector<long double>``
        ``vector<string>``
        ``python::object``          ``object``
        =======================     ======================
    """
    def __init__(self, pmap, g, key_type):
        if type(self) == PropertyMap:
            raise Exception("PropertyMap cannot be instantiated directly")
        self.__map = pmap
        self.__g = weakref.ref(g)
        self.__base_g = weakref.ref(g.base)  # keep reference to the
                                             # base graph, in case the
                                             # graph view is deleted.
        self.__key_type = key_type
        self.__convert = _converter(self.value_type())
        self.__register_map()

    def _get_any(self):
        t = self.key_type()
        g = self.get_graph()
        if t == "v":
            N = g.num_vertices(True)
        elif t == "e":
            N = g.edge_index_range
        else:
            N = 1
        self.reserve(N)
        return self.__map.get_map()

    def __register_map(self):
        for g in [self.__g(), self.__base_g()]:
            if g is not None:
                g._Graph__known_properties[id(self)] = weakref.ref(self)

    def __unregister_map(self):
        for g in [self.__g(), self.__base_g()]:
            if g is not None and id(self) in g._Graph__known_properties:
                del g._Graph__known_properties[id(self)]

    def __del__(self):
        if type(self) != PropertyMap:
            self.__unregister_map()

    def __repr__(self):
        # provide some more useful information
        if self.key_type() == "e":
            k = "Edge"
        elif self.key_type() == "v":
            k = "Vertex"
        else:
            k = "Graph"
        g = self.get_graph()
        if g is None:
            g = "a non-existent graph"
        else:
            g = "Graph 0x%x" % id(g)
        return ("<%sPropertyMap object with value type '%s',"
                + " for %s, at 0x%x>") % (k, self.value_type(), g, id(self))

    def copy(self, value_type=None, full=True):
        """Return a copy of the property map. If ``value_type`` is specified,
        the value type is converted to the chosen type. If ``full == False``, in
        the case of filtered graphs only the unmasked values are copied (with
        the remaining ones taking the type-dependent default value).
        """
        return self.get_graph().copy_property(self, value_type=value_type,
                                              full=full)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        if self.value_type() != "python::object":
            return self.copy()
        else:
            pmap = self.copy()
            g = self.get_graph()
            if self.key_type() == "g":
                iters = [g]
            elif self.key_type() == "v":
                iters = g.vertices()
            else:
                iters = g.edges()
            for v in iters:
                pmap[v] = copy.deepcopy(self[v], memo)
            return pmap

    def coerce_type(self, full=True):
        """Return a copy of the property map with the most appropriate type,
        i.e. the simplest type necessary to accomodate all the values exactly.
        If ``full == False``, in the case of filtered graphs only the unmasked
        values are copied (with the remaining ones taking the type-dependent
        default value).
        """
        types = ["bool", "int16_t", "int32_t", "int64_t", "double",
                 "long double", "vector<bool>", "vector<int16_t>",
                 "vector<int32_t>", "vector<int64_t>", "vector<double>",
                 "vector<long double>", "string", "vector<string>"]
        p = None
        for t in types:
            try:
                p = self.copy(value_type=t, full=full)
                if t == "bool":
                    a = p.a if full else p.fa
                    if p.a.max() > 1:
                        continue
                if p.copy(value_type=self.value_type(), full=full) == self:
                    break
            except (TypeError, ValueError, OverflowError, AttributeError):
                pass
        if p is None:
            p = self.copy()
        return p


    def get_graph(self):
        """Get the graph class to which the map refers."""
        g = self.__g()
        if g is None:
            g = self.__base_g()
        return g

    def key_type(self):
        """Return the key type of the map. Either 'g', 'v' or 'e'."""
        return self.__key_type

    def value_type(self):
        """Return the value type of the map."""
        return self.__map.value_type()

    def python_value_type(self):
        """Return the python-compatible value type of the map."""
        return _python_type(self.__map.value_type())

    def get_array(self):
        """Get a :class:`numpy.ndarray` subclass (:class:`~graph_tool.PropertyArray`)
        with the property values.

        .. note::

           An array is returned *only if* the value type of the property map is
           a scalar. For vector, string or object types, ``None`` is returned
           instead. For vector and string objects, indirect array access is
           provided via the :func:`~graph_tool.PropertyMap.get_2d_array()` and
           :func:`~graph_tool.PropertyMap.set_2d_array()` member functions.

        .. warning::

           The returned array does not own the data, which belongs to the
           property map. Therefore, if the graph changes, the array may become
           *invalid*. Do **not** store the array if the graph is to be modified;
           store a **copy** instead.

        """
        return self._get_data()

    def __set_array(self, v):
        a = self.get_array()
        if a is None:
            raise TypeError("cannot set property map values from array for" +
                            " property map of type: " + self.value_type())
        a[:] = v

    a = property(get_array, __set_array,
                 doc=r"""Shortcut to the :meth:`~PropertyMap.get_array` method
                 as an attribute. This makes assignments more convenient, e.g.:

                 >>> g = gt.Graph()
                 >>> g.add_vertex(10)
                 <...>
                 >>> prop = g.new_vertex_property("double")
                 >>> prop.a = np.random.random(10)           # Assignment from array
                 """)

    def __get_set_f_array(self, v=None, get=True):
        g = self.get_graph()
        if g is None:
            return None
        a = self.get_array()
        filt = [None]
        if self.__key_type == 'v':
            filt = g.get_vertex_filter()
            N = g.num_vertices()
        elif self.__key_type == 'e':
            filt = g.get_edge_filter()
            if g.get_vertex_filter()[0] is not None:
                filt = (g.new_edge_property("bool"), filt[1])
                libcore.mark_edges(g._Graph__graph, _prop("e", g, filt[0]))
                if filt[1]:
                    filt[0].a = numpy.logical_not(filt[0].a)
            elif g.edge_index_range != g.num_edges():
                filt = (g.new_edge_property("bool"), False)
                libcore.mark_edges(g._Graph__graph, _prop("e", g, filt[0]))
            if filt[0] is None:
                N = g.edge_index_range
            else:
                N = (filt[0].a == (not filt[1])).sum()
        if get:
            if a is None:
                return a
            if filt[0] is None:
                return a
            return a[filt[0].a == (not filt[1])][:N]
        else:
            if a is None:
                raise TypeError("cannot set property map values from array for" +
                                " property map of type: " + self.value_type())
            if filt[0] is None:
                try:
                    a[:] = v
                except ValueError:
                    a[:] = v[:len(a)]
            else:
                m = filt[0].a == (not filt[1])
                m *= m.cumsum() <= N
                try:
                    a[m] = v
                except ValueError:
                    a[m] = v[:len(m)][m]

    fa = property(__get_set_f_array,
                  lambda self, v: self.__get_set_f_array(v, False),
                  doc=r"""The same as the :attr:`~PropertyMap.a` attribute, but
                  instead an *indexed* array is returned, which contains only
                  entries for vertices/edges which are not filtered out. If
                  there are no filters in place, the array is not indexed, and
                  is identical to the :attr:`~PropertyMap.a` attribute.

                  Note that because advanced indexing is triggered, a **copy**
                  of the array is returned, not a view, as for the
                  :attr:`~PropertyMap.a` attribute. Nevertheless, the assignment
                  of values to the *whole* array at once works as expected.""")

    def __get_set_m_array(self, v=None, get=True):
        g = self.get_graph()
        if g is None:
            return None
        a = self.get_array()
        filt = [None]
        if self.__key_type == 'v':
            filt = g.get_vertex_filter()
        elif self.__key_type == 'e':
            filt = g.get_edge_filter()
            if g.get_vertex_filter()[0] is not None:
                filt = (g.new_edge_property("bool"), filt[1])
                libcore.mark_edges(g._Graph__graph, _prop("e", g, filt[0]))
                if filt[1]:
                    filt[0].a = 1 - filt[0].a
        if filt[0] is None or a is None:
            if get:
                return a
            else:
                return
        ma = numpy.ma.array(a, mask=(filt[0].a == False) if not filt[1] else (filt[0].a == True))
        if get:
            return ma
        else:
            ma[:] = v

    ma = property(__get_set_m_array,
                  lambda self, v: self.__get_set_m_array(v, False),
                  doc=r"""The same as the :attr:`~PropertyMap.a` attribute, but
                  instead a :class:`numpy.ma.MaskedArray` object is returned,
                  which contains only entries for vertices/edges which are not
                  filtered out. If there are no filters in place, a regular
                  :class:`~graph_tool.PropertyArray` is returned, which
                  is identical to the :attr:`~PropertyMap.a` attribute.""")

    def get_2d_array(self, pos=None, dtype=None):
        r"""Return a two-dimensional array of shape ``(M,N)``, where ``N`` is
        the number of vertices or edges, and ``M`` is the size of each property
        vector, which contains a copy of all entries of the vector-valued
        property map.

        The parameter ``pos`` must be a sequence of integers which
        specifies the indices of the property values which will be copied.

        If ``pos`` is not given (the default), then it will be assumed to
        correspond to the entire range of the first entry in the map.

        The parameter ``dtype`` determines the desired data-type for the array.
        If not given, it will be determined automatically (by applying
        promotion rules when necessary.)
        """

        if pos is None:
            pos = range(len(next(iter(self))))
        if self.key_type() == "g":
            raise ValueError("Cannot create multidimensional array for graph property maps.")
        if "vector" not in self.value_type() and (len(pos) > 1 or pos[0] != 0):
            raise ValueError("Cannot create array of dimension %d (indices %s) from non-vector property map of type '%s'." \
                             % (len(pos), str(pos), self.value_type()))
        if "string" in self.value_type():
            if "vector" in self.value_type():
                p = ungroup_vector_property(self, pos)
            else:
                p = [self]
            g = self.get_graph()
            vfilt = g.get_vertex_filter()
            efilt = g.get_edge_filter()
            if self.key_type() == "v":
                iters = g.vertices()
            else:
                iters = [None for i in range(g.edge_index_range)]
                idx = g.edge_index
                for e in g.edges():
                    iters[idx[e]] = e
                iters = [e for e in iters if e is not None]
            a = [[] for i in range(len(p))]
            for v in iters:
                for i in range(len(p)):
                    a[i].append(p[i][v])
            a = numpy.array(a, dtype=dtype)
            return a

        p = ungroup_vector_property(self, pos)
        a = numpy.array([x.fa for x in p], dtype=dtype)
        return a

    def set_2d_array(self, a, pos=None):
        r"""Set the entries of the vector-valued property map from a two-dimensional
        array ``a`` of shape ``(M,N)``, where ``N`` is the number of vertices or
        edges, and ``M`` is the size of each property vector. If given, the
        parameter ``pos`` must be a sequence of integers which specifies the
        indices of the property values which will be set (i.e. rows if the ``a``
        matrix).
        """

        if self.key_type() == "g":
            raise ValueError("Cannot set multidimensional array for graph property maps.")
        if "vector" not in self.value_type():
            if len(a.shape) != 1:
                raise ValueError("Cannot set array of shape %s to non-vector property map of type %s" % \
                                 (str(a.shape), self.value_type()))
            if self.value_type() != "string":
                self.fa = a
            else:
                g = self.get_graph()
                if self.key_type() == "v":
                    iters = g.vertices()
                else:
                    iters = [None for i in range(g.edge_index_range)]
                    idx = g.edge_index
                    for e in g.edges():
                        iters[idx[e]] = e
                    iters = [e for e in iters if e is not None]
                for j, v in enumerate(iters):
                    self[v] = a[j]
            return

        val = self.value_type()[7:-1]
        ps = []
        for i in range(a.shape[0]):
            ps.append(self.get_graph().new_property(self.key_type(), val))
            if self.value_type() != "string":
                ps[-1].fa = a[i]
            else:
                g = self.get_graph()
                if self.key_type() == "v":
                    iters = g.vertices()
                else:
                    iters = [None for i in range(g.edge_index_range)]
                    idx = g.edge_index
                    for e in g.edges():
                        iters[idx[e]] = e
                    iters = [e for e in iters if e is not None]
                for j, v in enumerate(iters):
                    ps[-1][v] = a[i, j]
        group_vector_property(ps, val, self, pos)

    def is_writable(self):
        """Return True if the property is writable."""
        return self.__map.is_writable()

    def set_value(self, val):
        """Sets all values in the property map to the same ``val``."""
        g = self.get_graph()
        val = self.__convert(val)
        if self.key_type() == "v":
            libcore.set_vertex_property(g._Graph__graph, _prop("v", g, self), val)
        elif self.key_type() == "e":
            libcore.set_edge_property(g._Graph__graph, _prop("e", g, self), val)
        else:
            self[g] = val

    def set_values(self, vals):
        """Sets values in the property map to the iterable ``vals``."""
        if self.a is None:
            it = iter(vals)
            for v in self._get_iter():
                self[v] = next(it)
        else:
            self.fa = vals

    def reserve(self, size):
        """Reserve enough space for ``size`` elements in underlying container. If the
           original size is already equal or larger, nothing will happen."""
        self.__map.reserve(size)

    def resize(self, size):
        """Resize the underlying container to contain exactly ``size`` elements."""
        self.__map.resize(size)

    def shrink_to_fit(self):
        """Shrink size of underlying container to accommodate only the necessary amount,
        and thus potentially freeing memory."""
        g = self.get_graph()
        if self.key_type() == "v":
            size = g.num_vertices(True)
        elif self.key_type() == "e":
            size = g.edge_index_range
        else:
            size = 1
        self.__map.resize(size)
        self.__map.shrink_to_fit()

    def swap(self, other):
        """Swap internal storage with ``other``."""
        if self.key_type() != other.key_type():
            raise ValueError("property maps must have the same key type")
        if self.value_type() != other.value_type():
            raise ValueError("property maps must have the same value type")
        self.__map.swap(other.__map)

    def transform(self, f, value_type=None, inplace=False):
        """Return a copy of the property map with the values transformed by the
        user-supplied function ``f``. If given, ``value_type`` specifies the
        value type for the new property map.

        If the value type of the original map allows, the values will be passed
        to ``f`` as a :class:`numpy.ndarray`. Otherwise, they will be passed
        individually.

        If ``inplace == True`` the transformation happens in-place, without
        copying the property map. In this case the parameter ``value_type`` is
        ignored.

        """
        p = self._new_pmap(value_type=value_type) if not inplace else self
        a = self.fa
        if a is None or p.fa is None:
            for v in self._get_iter():
                p[v] = f(self[v])
        else:
            p.fa = f(a)
        return p

    t = _copy_func(transform, "t")
    t.__doc__ = "Alias to :func:`~graph_tool.PropertyMap.transform`."

    def data_ptr(self):
        """Return the pointer to memory where the data resides."""
        return self.__map.data_ptr()

    def __getstate__(self):
        g = self.get_graph()
        if g is None:
            raise ValueError("cannot pickle orphaned property map")
        value_type = self.value_type()
        key_type = self.key_type()
        if not self.is_writable():
            vals = None
        else:
            u = GraphView(g, skip_vfilt=True, skip_efilt=True)
            if key_type == "v":
                vals = self.fa
                if vals is None:
                    if "vector" in value_type and value_type != "vector<string>":
                        vals = [self[v].a for v in u.vertices()]
                    else:
                        vals = [self.__convert(self[v]) for v in u.vertices()]
            elif key_type == "e":
                vals = self.a
                if vals is None:
                    if "vector" in value_type and value_type != "vector<string>":
                        vals = [self[e].a for e in u.edges()]
                    else:
                        vals = [self.__convert(self[e]) for e in u.edges()]
                else:
                    idx = u.get_edges([u.edge_index])[:,2]
                    vals = vals[idx]
            else:
                if "vector" in value_type and value_type != "vector<string>":
                    vals = self[g].a
                else:
                    vals = self.__convert(self[g])

        state = dict(g=g, value_type=value_type,
                     key_type=key_type, vals=vals,
                     is_vindex=self is g.vertex_index,
                     is_eindex=self is g.edge_index)

        return state

    def __setstate__(self, state):
        g = state["g"]
        key_type = state["key_type"]
        value_type = state["value_type"]
        vals = state["vals"]

        if state["is_vindex"]:
            pmap = g.vertex_index
        elif state["is_eindex"]:
            pmap = g.edge_index
        else:
            u = GraphView(g, skip_vfilt=True, skip_efilt=True)
            if key_type == "v":
                pmap = u.new_vertex_property(value_type, vals=vals)
            elif key_type == "e":
                pmap = u.new_edge_property(value_type, vals=vals)
            else:
                pmap = u.new_graph_property(value_type)
                pmap[u] = vals
            pmap = g.own_property(pmap)

        self.__map = pmap.__map
        self.__g = pmap.__g
        self.__base_g = pmap.__base_g
        self.__key_type = key_type
        self.__convert = _converter(self.value_type())
        self.__register_map()

    # Guarantee pickle backward compatibility

    def __getitem__(self, k):
        if self.key_type() == "v":
            return VertexPropertyMap.__getitem__(self, k)
        if self.key_type() == "e":
            return EdgePropertyMap.__getitem__(self, k)
        if self.key_type() == "g":
            return GraphPropertyMap.__getitem__(self, k)

    def __setitem__(self, k, v):
        if self.key_type() == "v":
            VertexPropertyMap.__setitem__(self, k, v)
        if self.key_type() == "e":
            EdgePropertyMap.__setitem__(self, k, v)
        if self.key_type() == "g":
            GraphPropertyMap.__setitem__(self, k, v)

    def __iter__(self):
        if self.key_type() == "v":
            return VertexPropertyMap.__iter__(self)
        if self.key_type() == "e":
            return EdgePropertyMap.__iter__(self)
        if self.key_type() == "g":
            return GraphPropertyMap.__iter__(self)

    def _get_data(self):
        if self.key_type() == "v":
            return VertexPropertyMap._get_data(self)
        if self.key_type() == "e":
            return EdgePropertyMap._get_data(self)
        if self.key_type() == "g":
            return GraphPropertyMap._get_data(self)


class VertexPropertyMap(PropertyMap):
    """This class provides a mapping from vertices to arbitrary properties.

    See :ref:`sec_property_maps` and :class:`PropertyMap` for more details.

    """

    def __init__(self, pmap, g):
        PropertyMap.__init__(self, pmap, g, "v")

    def __getitem__(self, k):
        return self._PropertyMap__map[int(k)]

    def __setitem__(self, k, v):
        k = int(k)
        try:
            self._PropertyMap__map[k] = v
        except TypeError:
            self._PropertyMap__map[k] = self._PropertyMap__convert(v)

    def _new_pmap(self, value_type=None):
        g = self.get_graph()
        return g.new_vp(self.value_type() if value_type is None else value_type)

    def _get_iter(self):
        g = self.get_graph()
        return g.vertices()

    def __iter__(self):
        for v in self._get_iter():
            yield self[v]

    def __len__(self):
        g = self.get_graph()
        return g.num_vertices()

    def _get_data(self):
        g = self.get_graph()
        if g is None:
            raise ValueError("Cannot get array for an orphaned property map")
        n = g._Graph__graph.get_num_vertices(False)
        a = self._PropertyMap__map.get_array(n)
        if a is None:
            return None
        return PropertyArray(a, self)

    def __eq__(self, other):
        if not isinstance(other, VertexPropertyMap):
            return False
        g = self.get_graph()
        if g.base is not other.get_graph().base:
            return False
        return libcore.compare_vertex_properties(g._Graph__graph,
                                                 self._get_any(),
                                                 other._get_any())

class EdgePropertyMap(PropertyMap):
    """This class provides a mapping from edges to arbitrary properties.

    See :ref:`sec_property_maps` and :class:`PropertyMap` for more details.

    """

    def __init__(self, pmap, g):
        PropertyMap.__init__(self, pmap, g, "e")

    def __getitem__(self, k):
        try:
            return self._PropertyMap__map[k]
        except ArgumentError:
            try:
                u, v = k
            except ValueError:
                raise TypeError("Edge property map indices must be Edge objects or integer pairs")
            g = self.get_graph()
            try:
                return self._PropertyMap__map[g.edge(u, v)]
            except ArgumentError:
                raise IndexError(f"nonexistent edge ({u}, {v})")

    def __setitem__(self, k, x):
        try:
            try:
                self._PropertyMap__map[k] = x
            except TypeError:
                self._PropertyMap__map[k] = self._PropertyMap__convert(x)
        except ArgumentError:
            try:
                u, v = k
            except ValueError:
                raise TypeError("Edge property map indices must be Edge objects or integer pairs")
            g = self.get_graph()
            try:
                e = g.edge(u, v)
                try:
                    self._PropertyMap__map[e] = x
                except TypeError:
                    self._PropertyMap__map[e] = self._PropertyMap__convert(x)
            except ArgumentError:
                raise IndexError(f"nonexistent edge ({u}, {v})")

    def _new_pmap(self, value_type=None):
        g = self.get_graph()
        return g.new_ep(self.value_type() if value_type is None else value_type)

    def _get_iter(self):
        g = self.get_graph()
        return g.edges()

    def __iter__(self):
        for e in self._get_iter():
            yield self[e]

    def __len__(self):
        g = self.get_graph()
        return g.num_edges()

    def _get_data(self):
        g = self.get_graph()
        if g is None:
            raise ValueError("Cannot get array for an orphaned property map")
        n = g.edge_index_range
        a = self._PropertyMap__map.get_array(n)
        if a is None:
            return None
        return PropertyArray(a, self)

    def __eq__(self, other):
        if not isinstance(other, EdgePropertyMap):
            return False
        g = self.get_graph()
        if g.base is not other.get_graph().base:
            return False
        return libcore.compare_edge_properties(g._Graph__graph,
                                               self._get_any(),
                                               other._get_any())

class GraphPropertyMap(PropertyMap):
    """This class provides a mapping from graphs to arbitrary properties.

    See :ref:`sec_property_maps` and :class:`PropertyMap` for more details.

    """

    def __init__(self, pmap, g):
        PropertyMap.__init__(self, pmap, g, "g")

    def __getitem__(self, k):
        return self._PropertyMap__map[self.get_graph()._Graph__graph]

    def __setitem__(self, k, v):
        g = self.get_graph()._Graph__graph
        try:
            self._PropertyMap__map[g] = v
        except TypeError:
            self._PropertyMap__map[g] = self._PropertyMap__convert(v)

    def _new_pmap(self, value_type=None):
        g = self.get_graph()
        return g.new_gp(self.value_type() if value_type is None else value_type)

    def _get_iter(self):
        return [self.get_graph()]

    def __iter__(self):
        return _get_iter()

    def __len__(self):
        return 1

    def _get_data(self):
        g = self.get_graph()
        if g is None:
            raise ValueError("Cannot get array for an orphaned property map")
        a = self.__map.get_array(1)
        if a is None:
            return None
        return PropertyArray(a, self)


class PropertyArray(numpy.ndarray):
    """This is a :class:`numpy.ndarray` subclass which keeps a reference of its
    :class:`~graph_tool.PropertyMap` owner.
    """

    __array_priority__ = -10

    def _get_pmap(self):
        return self._prop_map

    def _set_pmap(self, value):
        self._prop_map = value

    prop_map = property(_get_pmap, _set_pmap,
                        doc=":class:`~graph_tool.PropertyMap` owner instance.")

    def __new__(cls, input_array, prop_map):
        obj = numpy.asarray(input_array).view(cls)
        obj.prop_map = prop_map
        return obj

def _check_prop_writable(prop, name=None):
    if not prop.is_writable():
        raise ValueError("property map%s is not writable." %\
                         ((" '%s'" % name) if name is not None else ""))


def _check_prop_scalar(prop, name=None, floating=False):
    scalars = ["bool", "int16_t", "int32_t", "int64_t", "unsigned long",
               "double", "long double"]
    if floating:
        scalars = ["double", "long double"]

    if prop.value_type() not in scalars:
        raise ValueError("property map%s is not of scalar%s type." %\
                         (((" '%s'" % name) if name is not None else ""),
                          (" floating" if floating else "")))


def _check_prop_vector(prop, name=None, scalar=True, floating=False):
    scalars = ["bool", "int16_t", "int32_t", "int64_t", "unsigned long",
               "double", "long double"]
    if not scalar:
        scalars += ["string"]
    if floating:
        scalars = ["double", "long double"]
    vals = ["vector<%s>" % v for v in scalars]
    if prop.value_type() not in vals:
        raise ValueError("property map%s is not of vector%s type." %\
                         (((" '%s'" % name) if name is not None else ""),
                          (" floating" if floating else "")))


def group_vector_property(props, value_type=None, vprop=None, pos=None):
    """Group list of properties ``props`` into a vector property map of the same type.

    Parameters
    ----------
    props : list of :class:`~graph_tool.PropertyMap`
        Properties to be grouped.
    value_type : string (optional, default: None)
        If supplied, defines the value type of the grouped property.
    vprop : :class:`~graph_tool.PropertyMap` (optional, default: None)
        If supplied, the properties are grouped into this property map.
    pos : list of ints (optional, default: None)
        If supplied, should contain a list of indices where each corresponding
        element of ``props`` should be inserted.

    Returns
    -------
    vprop : :class:`~graph_tool.PropertyMap`
       A vector property map with the grouped values of each property map in
       ``props``.

    Examples
    --------
    >>> from numpy.random import seed, randint
    >>> from numpy import array
    >>> seed(42)
    >>> gt.seed_rng(42)
    >>> g = gt.random_graph(100, lambda: (3, 3))
    >>> props = [g.new_vertex_property("int") for i in range(3)]
    >>> for i in range(3):
    ...    props[i].a = randint(0, 100, g.num_vertices())
    >>> gprop = gt.group_vector_property(props)
    >>> print(gprop[g.vertex(0)].a)
    [51 25  8]
    >>> print(array([p[g.vertex(0)] for p in props]))
    [51 25  8]
    """

    g = props[0].get_graph()
    vtypes = set()
    keys = set()
    for i, p in enumerate(props):
        if "vector" in p.value_type():
            raise ValueError("property map 'props[%d]' is a vector property." %
                             i)
        vtypes.add(p.value_type())
        keys.add(p.key_type())
    if len(keys) > 1:
        raise ValueError("'props' must be of the same key type.")
    k = keys.pop()

    if vprop is None:
        if value_type is None and len(vtypes) == 1:
            value_type = vtypes.pop()

        if value_type is not None:
            value_type = "vector<%s>" % value_type
            if k == 'v':
                vprop = g.new_vertex_property(value_type)
            elif k == 'e':
                vprop = g.new_edge_property(value_type)
            else:
                vprop = g.new_graph_property(value_type)
        else:
            raise ValueError("Can't automatically determine property map value" +
                             " type. Please provide the 'value_type' parameter.")
    _check_prop_vector(vprop, name="vprop", scalar=False)

    for i, p in enumerate(props):
        if k != "g":
            u = GraphView(g, directed=True, reversed=g.is_reversed(),
                          skip_properties=True)
            libcore.group_vector_property(u._Graph__graph, _prop(k, g, vprop),
                                          _prop(k, g, p),
                                          i if pos is None else pos[i],
                                          k == 'e')
        else:
            vprop[g][i if pos is None else pos[i]] = p[g]
    return vprop


def ungroup_vector_property(vprop, pos=None, props=None):
    """Ungroup vector property map ``vprop`` into a list of non-vector property maps.

    Parameters
    ----------
    vprop : :class:`~graph_tool.PropertyMap`
        Vector property map to be ungrouped.
    pos : list of ints (optional, default: ``None``)
        A list of indices corresponding to where each element of ``vprop``
        should be inserted into the ungrouped list. If not provided, it will
        correspond to the entire range of the first element in the map.
    props : list of :class:`~graph_tool.PropertyMap`  (optional, default: ``None``)
        If supplied, should contain a list of property maps to which ``vprop``
        should be ungroupped.

    Returns
    -------
    props : list of :class:`~graph_tool.PropertyMap`
       A list of property maps with the ungrouped values of ``vprop``.

    Examples
    --------
    >>> from numpy.random import seed, randint
    >>> from numpy import array
    >>> seed(42)
    >>> gt.seed_rng(42)
    >>> g = gt.random_graph(100, lambda: (3, 3))
    >>> prop = g.new_vertex_property("vector<int>")
    >>> for v in g.vertices():
    ...    prop[v] = randint(0, 100, 3)
    >>> uprops = gt.ungroup_vector_property(prop)
    >>> print(prop[g.vertex(0)].a)
    [51 92 14]
    >>> print(array([p[g.vertex(0)] for p in uprops]))
    [51 92 14]

    """

    if pos is None:
        pos = range(len(next(iter(vprop))))
    g = vprop.get_graph()
    _check_prop_vector(vprop, name="vprop", scalar=False)
    k = vprop.key_type()
    value_type = vprop.value_type().split("<")[1].split(">")[0]
    if props is None:
        if k == 'v':
            props = [g.new_vertex_property(value_type) for i in pos]
        elif k == 'e':
            props = [g.new_edge_property(value_type) for i in pos]
        else:
            props = [g.new_graph_property(value_type) for i in pos]

    for i, p in enumerate(pos):
        if props[i].key_type() != k:
            raise ValueError("'props' must be of the same key type as 'vprop'.")

        if k != 'g':
            u = GraphView(g, directed=True, reversed=g.is_reversed(),
                          skip_properties=True)
            libcore.ungroup_vector_property(u._Graph__graph,
                                            _prop(k, g, vprop),
                                            _prop(k, g, props[i]),
                                            p, k == 'e')
        else:
            if len(vprop[g]) <= pos[i]:
                vprop[g].resize(pos[i] + 1)
            props[i][g] = vprop[g][pos[i]]
    return props

def map_property_values(src_prop, tgt_prop, map_func):
    """Map the values of ``src_prop`` to ``tgt_prop`` according to the mapping
    function ``map_func``.

    Parameters
    ----------
    src_prop : :class:`~graph_tool.PropertyMap`
        Source property map.
    tgt_prop : :class:`~graph_tool.PropertyMap`
        Target property map.
    map_func : function or callable object
        Function mapping values of ``src_prop`` to values of ``tgt_prop``.

    Returns
    -------
    None

    Examples
    --------
    >>> g = gt.collection.data["lesmis"]
    >>> label_len = g.new_vertex_property("int64_t")
    >>> gt.map_property_values(g.vp.label, label_len,
    ...                        lambda x: len(x))
    >>> print(label_len.a)
    [ 6  8 14 11 12  8 12  8  5  6  7  7 10  6  7  7  9  9  7 11  9  6  7  7
     13 10  7  6 12 10  8  8 11  6  5 12  6 10 11  9 12  7  7  6 14  7  9  9
      8 12  6 16 12 11 14  6  9  6  8 10  9  7 10  7  7  4  9 14  9  5 10 12
      9  6  6  6 12]
    """

    if src_prop.key_type() != tgt_prop.key_type():
        raise ValueError("src_prop and tgt_prop must be of the same key type")
    g = src_prop.get_graph()
    k = src_prop.key_type()
    if k == "g":
        tgt_prop[g] = map_func(src_prop[g])
        return
    u = GraphView(g, directed=True, reversed=g.is_reversed(),
                  skip_properties=True)
    libcore.property_map_values(u._Graph__graph,
                                _prop(k, g, src_prop),
                                _prop(k, g, tgt_prop),
                                map_func, k == 'e')

def infect_vertex_property(g, prop, vals=None):
    """Propagate the `prop` values of vertices with value `val` to all their
    out-neighbors.

    Parameters
    ----------
    prop : :class:`~graph_tool.VertexPropertyMap`
        Property map to be modified.
    vals : list (optional, default: `None`)
        List of values to be propagated. If not provided, all values
        will be propagated.

    Returns
    -------
    None : ``None``

    Examples
    --------
    >>> from numpy.random import seed
    >>> seed(42)
    >>> gt.seed_rng(42)
    >>> g = gt.random_graph(100, lambda: (3, 3))
    >>> prop = g.vertex_index.copy("int32_t")
    >>> gt.infect_vertex_property(g, prop, [10])
    >>> print(sum(prop.a == 10))
    4
    """
    libcore.infect_vertex_property(g._Graph__graph, _prop("v", g, prop),
                                   vals)


@_limit_args({"endpoint": ["source", "target"]})
def edge_endpoint_property(g, prop, endpoint, eprop=None):
    """Return an edge property map corresponding to the vertex property `prop` of
    either the target and source of the edge, according to `endpoint`.

    Parameters
    ----------
    prop : :class:`~graph_tool.VertexPropertyMap`
        Vertex property map to be used to propagated to the edge.
    endpoint : `"source"` or `"target"`
        Edge endpoint considered. If the graph is undirected, the source is
        always the vertex with the lowest index.
    eprop : :class:`~graph_tool.EdgePropertyMap` (optional, default: `None`)
        If provided, the resulting edge properties will be stored here.

    Returns
    -------
    eprop : :class:`~graph_tool.EdgePropertyMap`
        Propagated edge property.

    Examples
    --------
    >>> gt.seed_rng(42)
    >>> g = gt.random_graph(100, lambda: (3, 3))
    >>> esource = gt.edge_endpoint_property(g, g.vertex_index, "source")
    >>> print(esource.a)
    [55 29 10 31 41 19 53 45 18 97 58 83  3  1 95 76 15 74 43 68 43 57  2  0
     50 37 53 64 59  2 11 91 28 19 17 87 54  4 91 38 73 62 23  7 97 87 15 55
      8 50 82 30 58 93 77  6 96 15 47 68  8  8 14 71 86 81 74 98 33 57 21 77
     21 49 32  7 78 58 51 34 14 45 85 78 46 65 52 23 78 40 51 39 56 22 65 62
     71  3 24 10 63 60 72 60 20 28 92 49 88 70 84 13 21 26 65 47 59 80  3 99
     93  6 12 92 42 18 70 39 48 22 89 33 26 38 24 93 84 81 67 68 94 20 73 35
     25 51 57 88 52  6 86 70 36 62 95 83 22 89 54 24 37 43 13 94  5 48 81 44
     38 69 27 29 61 63  4 45  1 61 46 59 74 80 40 17 40 76 69 53 29  9 54 12
     72 12 82 72 11  0 44 94 17 90 79 14 86 26 71 79 19 35 61  9 56 73 52 23
     34 66 75 67 10 99 33 66 97 42 30  5 20 35 79 85 41 36 31 36 98  4 46  2
     82 27 30 47 49 91  1 84 13 98 85  7 99 77 64 90  0 95 31 90 67 88 55 27
     16 25 16 32 96 32 69 42 37 18 11  9 50 76 75 56 39 87 66 96 25 44 16 48
     75 89 63 41 64 34  5 28 60 80 83 92]
    """

    val_t = prop.value_type()
    if val_t == "unsigned long" or val_t == "unsigned int":
        val_t = "int64_t"
    if eprop is None:
        eprop = g.new_edge_property(val_t)
    if eprop.value_type() != val_t:
        raise ValueError("'eprop' must be of the same value type as 'prop': " +
                         val_t)
    libcore.edge_endpoint(g._Graph__graph, _prop("v", g, prop),
                          _prop("e", g, eprop), endpoint)
    return eprop

@_limit_args({"direction": ["in", "out"], "op": ["sum", "prod", "min", "max"]})
def incident_edges_op(g, direction, op, eprop, vprop=None):
    """Return a vertex property map corresponding to a specific operation (sum,
    product, min or max) on the edge property `eprop` of incident edges on each
    vertex, following the direction given by `direction`.

    Parameters
    ----------
    direction : `"in"` or `"out"`
        Direction of the incident edges.
    op : `"sum"`, `"prod"`, `"min"` or `"max"`
        Operation performed on incident edges.
    eprop : :class:`~graph_tool.EdgePropertyMap`
        Edge property map to be summed.
    vprop : :class:`~graph_tool.VertexPropertyMap` (optional, default: `None`)
        If provided, the resulting vertex properties will be stored here.

    Returns
    -------
    vprop : :class:`~graph_tool.VertexPropertyMap`
        Resulting vertex property.

    Examples
    --------
    >>> gt.seed_rng(42)
    >>> g = gt.random_graph(100, lambda: (3, 3))
    >>> vsum = gt.incident_edges_op(g, "out", "sum", g.edge_index)
    >>> print(vsum.a)
    [476 435 290 227 448 685 325 369 169 675 321 500 506 521 345 119 816 417
     406 246 473 254 378 344 391 693 450 674 432 360 519 495 610 421 588 581
     620 457 340 498 455 527 620 199 650 263 500 416 580 424 349 313 448 219
     384 309 583 236 139 323 500 559 289 563 573 293 722 617 217 625 386 365
     489 395 263 784 477 378 247 639 595 368 484 464 493 563 418 361 516 576
     715 314 528 308 502 425 607 277 552 592]
    """

    val_t = eprop.value_type()
    if val_t == "unsigned long" or val_t == "unsigned int":
        val_t = "int64_t"
    if vprop is None:
        vprop = g.new_vertex_property(val_t)
    orig_vprop = vprop
    if vprop.value_type != val_t:
        vprop = g.new_vertex_property(val_t)
    if direction == "in" and not g.is_directed():
        return orig_vprop
    if direction == "in":
        g = GraphView(g, reversed=True, skip_properties=True)
    libcore.out_edges_op(g._Graph__graph, _prop("e", g, eprop),
                          _prop("v", g, vprop), op)
    if vprop is not orig_vprop:
        g.copy_property(vprop, orig_vprop)
    return orig_vprop

@_limit_args({"htype": ["int8_t", "int32_t", "int64_t"]})
def perfect_prop_hash(props, htype="int32_t"):
    """Given a list of property maps `props` of the same type, a derived list of
    property maps with integral type `htype` is returned, where each value is
    replaced by a perfect (i.e. unique) hash value.

    .. note::
       The hash value is deterministic, but it will not be necessarily the same
       for different values of `props`.
    """

    val_types = set([p.value_type() for p in props])
    if len(val_types) > 1:
        raise ValueError("All properties must have the same value type")
    hprops = [p.get_graph().new_property(p.key_type(), htype) for p in props]

    eprops = [p for p in props if p.key_type() == "e"]
    heprops = [p for p in hprops if p.key_type() == "e"]

    vprops = [p for p in props if p.key_type() == "v"]
    hvprops = [p for p in hprops if p.key_type() == "v"]

    hdict = libcore.any()

    for eprop, heprop in zip(eprops, heprops):
        g = eprop.get_graph()
        g = GraphView(g, directed=True, skip_properties=True)
        libcore.perfect_ehash(g._Graph__graph, _prop('e', g, eprop),
                              _prop('e', g, heprop), hdict)

    for vprop, hvprop in zip(vprops, hvprops):
        g = vprop.get_graph()
        g = GraphView(g, directed=True, skip_properties=True)
        libcore.perfect_vhash(g._Graph__graph, _prop('v', g, vprop),
                              _prop('v', g, hvprop), hdict)

    return hprops



class InternalPropertyDict(dict):
    """Internal dictionary of property maps. It only accepts string keys and
    :class:`PropertyMap` instances as values."""

    def __init__(self, g):
        self.g = weakref.ref(g)
        dict.__init__(self)

    @_require("key", tuple)
    @_require("val", PropertyMap)
    def __setitem__(self, key, val):
        t, k = key
        u = val.get_graph()
        if u is None:
            raise ValueError("Received orphaned property map")
        g = self.g()
        if u.base is not g.base:
            raise ValueError("Received property map for graph %s (base: %s), expected: %s (base: %s)" %
                         (str(u), str(u.base), str(g), str(g.base)))
        self.__set_property(t, k, val)

    @_limit_args({"t": ["v", "e", "g"]})
    @_require("key", str)
    def __set_property(self, t, key, v):
        dict.__setitem__(self, (t, key), v)

    @_require("key", tuple)
    def __delitem__(self, key):
        dict.__delitem__(self, key)

    @_require("key", tuple)
    def setdefault(self, key, default=None):
        if not isinstance(default, PropertyMap):
            raise ValueError("default parameter must be of type PropertyMap, not: %s" % type(default))
        v = self.get(key, None)
        if v is None:
            self[key] = v = default
        return v

    def update(self, *args, **kwargs):
        temp = dict(*args, **kwargs)
        for k, v in temp.items():
            self[k] = v


class PropertyDict(object):
    """Wrapper for the dict of vertex, graph or edge properties, which sets the
    value on the property map when changed in the dict.

    For convenience, the dictionary entries are also available via attributes.
    """
    def __init__(self, properties, t):
        super(PropertyDict, self).__setattr__("_PropertyDict__properties", properties)
        super(PropertyDict, self).__setattr__("_PropertyDict__t", t)

    def __contains__(self, key):
        return (self.__t, key) in self.__properties

    def __getitem__(self, key):
        if self.__t == "g":
            p = self.__properties[(self.__t, key)]
            return p[p.get_graph()]
        return self.__properties[(self.__t, key)]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key, default=None):
        try:
            x = self[key]
            del self[key]
            return x
        except KeyError:
            return default

    def __setitem__(self, key, val):
        k = (self.__t, key)
        if self.__t == "g" and not isinstance(val, PropertyMap) and k in self.__properties:
            p = self.__properties[k]
            p[p.get_graph()] = val
        else:
            if not isinstance(val, PropertyMap):
                raise ValueError("value must be of type PropertyMap, not %s" % str(type(val)))
            if val.key_type() != self.__t:
                def name(t):
                    if t == "e":
                        return "Edge"
                    if t == "v":
                        return "Vertex"
                    if t == "g":
                        return "Graph"
                raise ValueError("wanted a property map of type '%s', not '%s'" %
                                 (name(self.__t), name(val.key_type())))
            self.__properties[k] = val

    def setdefault(self, key, default=None):
        self.__properties.setdefault((self.__t, key), default)

    def update(self, *args, **kwargs):
        temp = dict(*args, **kwargs)
        for k, v in temp.items():
            self.__properties[(self.__t, k)] = v

    def __delitem__(self, key):
        del self.__properties[(self.__t, key)]

    def clear(self):
        keys = []
        for k in self.__properties.keys():
            if k[0] == self.__t:
                keys.append(k)
        for k in keys:
            del self.__properties[k]

    def __len__(self):
        count = 0
        for k in self.__properties.keys():
            if k[0] == self.__t:
                count += 1
        return count

    def __iter__(self):
        return self.keys()

    def iterkeys(self):
        for k in self.__properties.keys():
            if k[0] == self.__t:
                yield k[1]

    def items(self):
        for k, v in self.__properties.items():
            if k[0] == self.__t:
                yield k[1], v

    def itervalues(self):
        for k, v in self.__properties.items():
            if k[0] == self.__t:
                yield v

    def keys(self):
        return self.iterkeys()

    def values(self):
        return self.itervalues()

    def __repr__(self):
        temp = dict([(k[1], v) for k, v in self.__properties.items() if k[0] == self.__t])
        return repr(temp)

    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            return getattr(super(PropertyDict, self), attr)

    def __setattr__(self, attr, val):
        return self.__setitem__(attr, val)


################################################################################
# Graph class
# The main graph interface
################################################################################

from .libgraph_tool_core import Vertex, EdgeBase, Vector_bool, Vector_int16_t, \
    Vector_int32_t, Vector_int64_t, Vector_double, Vector_long_double, \
    Vector_string, Vector_size_t, Vector_cdouble, new_vertex_property, \
    new_edge_property, new_graph_property


class Graph(object):
    """General multigraph class.

    This class encapsulates either a directed multigraph (default or if
    ``directed == True``) or an undirected multigraph (if ``directed ==
    False``), with optional internal edge, vertex or graph properties.

    If ``g`` is specified, it can be one of:

        1. Another :class:`Graph` object, in which case the corresponding graph
           (and its internal properties) will be copied.

        2. An integer, in which case it corresponds to the number of vertices in
           the graph, which is initially empty.

        3. An edge list, i.e. an iterable over (source, target) pairs, which
           will be used to populate the graph.

           This is equivalent to calling :meth:`~graph_tool.Graph.add_edge_list`
           with an empty graph, as follows:

           .. doctest:: init_list

              >>> elist = [(0, 1), (0, 2)]
              >>> g = gt.Graph()
              >>> g.add_edge_list(elist)

        4. An adjacency list, i.e. a dictionary with vertex keys mapping to an
           interable of vertices, which will be used to populate the graph. For
           directed graphs, the adjacency should list the out-neighbors.

           This is equivalent to calling :meth:`~graph_tool.Graph.add_edge_list`
           as such:

           .. doctest:: init_adj

              >>> adj = {0: [1, 2], 2: [3], 4: []}
              >>> def elist():
              ...     for u, vw in adj.items():
              ...         k = 0
              ...         for v in vw:
              ...             k += 1
              ...             yield u, v
              ...         if k == 0:
              ...             yield u, None
              >>> g = gt.Graph()
              >>> g.add_edge_list(elist())

           .. note::

              For undirected graphs, if a vertex ``u`` appears in the adjacency
              list of ``v`` and vice versa, then the edge ``(u,v)`` is added
              twice in the graph. To prevent this from happening the adjancecy
              list should mention an edge only once.

        5. A sparse adjacency matrix of type :class:`scipy.sparse.sparray` or
           :class:`scipy.sparse.spmatrix`. The matrix entries will be stored as
           an internal :class:`~graph_tool.EdgePropertyMap` named ``"weight"``.
           If ``directed == False``, only the upper triangular portion of this
           matrix will be considered, and the remaining entries will be ignored.

           This is equivalent to calling :meth:`~graph_tool.Graph.add_edge_list`
           as such:

           .. doctest:: init_mat

              >>> a = scipy.sparse.coo_array([[0, 2, 1], [3, 1, 2], [0, 1, 0]])
              >>> s, t, w = scipy.sparse.find(a)
              >>> es = np.array([s, t, w]).T
              >>> g = gt.Graph(a.shape[0])
              >>> g.add_edge_list(es, eprops=[("weight", "int")])

    In cases 3 and 4 above, all remaining keyword parameters passed to
    :class:`Graph` will be passed along to the :meth:`Graph.add_edge_list`
    function. If the option ``hashed == True`` is passed, the vertex ids will be
    stored in an internal :class:`~graph_tool.VertexPropertyMap` called
    ``"ids"``.

    In case ``g`` is specified and points to a :class:`Graph` object, the
    following options take effect:

    * If ``prune`` is set to ``True``, only the filtered graph will be copied,
      and the new graph object will not be filtered. Optionally, a tuple of
      three booleans can be passed as value to ``prune``, to specify a different
      behavior to vertex, edge, and reversal filters, respectively.

    * If ``vorder`` is specified, it should correspond to a vertex
      :class:`~graph_tool.VertexPropertyMap` specifying the ordering of the
      vertices in the copied graph.

    The value of ``set_fast_edge_removal`` is passed to
    :meth:`~graph_tool.Graph.set_fast_edge_removal`.

    .. note::

       The graph is implemented internally as an `adjacency list`_, where both
       vertex and edge lists are C++ STL vectors.

    .. _adjacency list: http://en.wikipedia.org/wiki/Adjacency_list

    """

    def __init__(self, g=None, directed=True, prune=False, vorder=None,
                 fast_edge_removal=False, **kwargs):
        self.__properties = InternalPropertyDict(self)
        self.__graph_properties = PropertyDict(self.__properties, "g")
        self.__vertex_properties = PropertyDict(self.__properties, "v")
        self.__edge_properties = PropertyDict(self.__properties, "e")
        self.__known_properties = {}
        self.__filter_state = {"reversed": False,
                               "edge_filter": (None, False),
                               "vertex_filter": (None, False),
                               "directed": True}
        if g is None or isinstance(g, (collections.abc.Iterable, int)):
            self.__graph = libcore.GraphInterface()
            self.set_directed(directed)

            # internal index maps
            self.__vertex_index = \
                     VertexPropertyMap(libcore.get_vertex_index(self.__graph), self)
            self.__edge_index = \
                     EdgePropertyMap(libcore.get_edge_index(self.__graph), self)

            if g is not None:
                if isinstance(g, int):
                    self.add_vertex(g)
                else:
                    if isinstance(g, dict):
                        def elist():
                            for u, vw in g.items():
                                k = 0
                                for v in vw:
                                    k += 1
                                    yield u, v
                                if k == 0:
                                    yield u, None
                        vids = self.add_edge_list(elist(), **kwargs)
                    elif isinstance(g, (scipy.sparse.sparray,
                                        scipy.sparse.spmatrix)):
                        self.add_vertex(g.shape[0])
                        if not directed:
                            g = scipy.sparse.triu(g)
                        s, t, w = scipy.sparse.find(g)
                        es = numpy.array([s, t, w]).T
                        vids = self.add_edge_list(es,
                                                  eprops=[("weight",
                                                           _gt_type(w.dtype))])
                    else:
                        vids = self.add_edge_list(g, **kwargs)
                    if vids is not None:
                        self.vp.ids = vids
            elif len(kwargs) > 0:
                raise ValueError("unrecognized keyword arguments: " +
                                 str(list(kwargs.keys())))
        else:
            if isinstance(prune, bool):
                vprune = eprune = rprune = prune
            else:
                vprune, eprune, rprune = prune
            if not (vprune or eprune or rprune):
                gv = GraphView(g, skip_vfilt=True,
                               skip_efilt=True)
                if not rprune:
                    gv.set_reversed(False)
            else:
                gv = g

            # The filters may or may not not be in the internal property maps
            vfilt = g.get_vertex_filter()[0]
            efilt = g.get_edge_filter()[0]

            if (vorder is None and ((vfilt is None and efilt is None) or
                                    (not vprune and not eprune))):
                # Do a simpler, faster copy.
                self.__graph = libcore.GraphInterface(gv.__graph, False,
                                                      [], [], None)

                # internal index maps
                self.__vertex_index = \
                         VertexPropertyMap(libcore.get_vertex_index(self.__graph), self)
                self.__edge_index = \
                         EdgePropertyMap(libcore.get_edge_index(self.__graph), self)

                nvfilt = nefilt = None
                for k, m in g.properties.items():
                    nmap = self.copy_property(m, g=gv)
                    self.properties[k] = nmap
                    if m is vfilt:
                        nvfilt = nmap
                    if m is efilt:
                        nefilt = nmap
                if vfilt is not None:
                    if nvfilt is None:
                        nvfilt = self.copy_property(vfilt, g=gv)
                if efilt is not None:
                    if nefilt is None:
                        nefilt = self.copy_property(efilt, g=gv)
                self.set_filters(nefilt, nvfilt,
                                 inverted_edges=g.get_edge_filter()[1],
                                 inverted_vertices=g.get_vertex_filter()[1])
            else:

                # Copy all internal properties from original graph.
                vprops = []
                eprops = []
                ef_pos = vf_pos = None
                for k, m in gv.vertex_properties.items():
                    if not m.is_writable():
                        m = m.copy("int32_t")
                    if not vprune and m is vfilt:
                        vf_pos = len(vprops)
                    vprops.append([_prop("v", gv, m), libcore.any()])
                for k, m in gv.edge_properties.items():
                    if not m.is_writable():
                        m = m.copy("int32_t")
                    if not eprune and m is efilt:
                        ef_pos = len(eprops)
                    eprops.append([_prop("e", gv, m), libcore.any()])
                if not vprune and vf_pos is None and vfilt is not None:
                    vf_pos = len(vprops)
                    vprops.append([_prop("v", gv, vfilt), libcore.any()])
                if not eprune and ef_pos is None and efilt is not None:
                    ef_pos = len(eprops)
                    eprops.append([_prop("e", gv, efilt), libcore.any()])

                # The vertex ordering
                if vorder is None:
                    vorder = gv.new_vertex_property("int",
                                                    vals=numpy.arange(gv.num_vertices()))
                else:
                    vorder = vorder.copy("int")

                # The actual copying of the graph and property maps
                self.__graph = libcore.GraphInterface(gv.__graph, False,
                                                      vprops,
                                                      eprops,
                                                      _prop("v", gv, vorder))
                # Internal index maps
                self.__vertex_index = \
                         VertexPropertyMap(libcore.get_vertex_index(self.__graph), self)
                self.__edge_index = \
                         EdgePropertyMap(libcore.get_edge_index(self.__graph), self)

                # Put the copied properties in the internal dictionary
                for i, (k, m) in enumerate(gv.vertex_properties.items()):
                    pmap = new_vertex_property(m.value_type() if m.is_writable() else "int32_t",
                                               self.__graph.get_vertex_index(),
                                               vprops[i][1])
                    self.vertex_properties[k] = VertexPropertyMap(pmap, self)

                for i, (k, m) in enumerate(gv.edge_properties.items()):
                    pmap = new_edge_property(m.value_type() if m.is_writable() else "int32_t",
                                             self.__graph.get_edge_index(),
                                             eprops[i][1])
                    self.edge_properties[k] = EdgePropertyMap(pmap, self)

                for k, v in gv.graph_properties.items():
                    new_p = self.new_graph_property(v.value_type())
                    new_p[self] = v[gv]
                    self.graph_properties[k] = new_p

                epmap = vpmap = None
                if vf_pos is not None:
                    vpmap = new_vertex_property("bool",
                                                self.__graph.get_vertex_index(),
                                                vprops[vf_pos][1])
                    vpmap = VertexPropertyMap(vpmap, self)
                if ef_pos is not None:
                    epmap = new_edge_property("bool",
                                              self.__graph.get_edge_index(),
                                              eprops[ef_pos][1])
                    epmap = EdgePropertyMap(epmap, self)
                self.set_filters(epmap, vpmap,
                                 inverted_edges=g.get_edge_filter()[1],
                                 inverted_vertices=g.get_vertex_filter()[1])

            if not rprune:
                self.set_reversed(g.is_reversed())

            # directedness is always a filter
            self.set_directed(g.is_directed())
        self.set_fast_edge_removal(fast_edge_removal)

    def _get_any(self):
        return self.__graph.get_graph_view()

    def copy(self):
        """Return a deep copy of self. All :ref:`internal property maps <sec_internal_props>`
        are also copied."""
        return Graph(self)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        g = self.copy()
        for k, prop in [x for x in g.properties.items()
                        if x[1].value_type() == "python::object"]:
            g.properties[k] = copy.deepcopy(prop)
        return g

    def __repr__(self):
        # provide more useful information
        d = "directed" if self.is_directed() else "undirected"
        fr = ", reversed" if self.is_reversed() and self.is_directed() else ""
        p = []
        if len(self.vp) > 0:
            p.append("%d internal vertex %s" % (len(self.vp),
                                                "property" if len(self.vp) == 1 else "properties"))
        if len(self.ep) > 0:
            p.append("%d internal edge %s" % (len(self.ep),
                                              "property" if len(self.ep) == 1 else "properties"))
        if len(self.gp) > 0:
            p.append("%d internal graph %s" % (len(self.gp),
                                               "property" if len(self.gp) == 1 else "properties"))
        p = ", ".join(p)
        if len(p) > 0:
            p = ", " + p
        f = ""
        if self.get_edge_filter()[0] is not None:
            f += ", edges filtered by %s" % (str(self.get_edge_filter()))
        if self.get_vertex_filter()[0] is not None:
            f += ", vertices filtered by %s" % (str(self.get_vertex_filter()))
        n = self.num_vertices()
        e = self.num_edges()
        return "<%s object, %s%s, with %d %s and %d edge%s%s%s, at 0x%x>"\
               % (type(self).__name__, d, fr, n,
                  "vertex" if n == 1 else "vertices", e, "" if e == 1 else "s",
                  p, f, id(self))

    # Graph access
    # ============

    def vertices(self):
        """Return an :meth:`iterator <iterator.__iter__>` over the vertices.

        .. note::

           The order of the vertices traversed by the iterator **always**
           corresponds to the vertex index ordering, as given by the
           :attr:`~graph_tool.Graph.vertex_index` property map.

        Examples
        --------
        >>> g = gt.Graph()
        >>> vlist = list(g.add_vertex(5))
        >>> vlist2 = []
        >>> for v in g.vertices():
        ...     vlist2.append(v)
        ...
        >>> assert(vlist == vlist2)

        """
        viter = libcore.get_vertices(self.__graph)
        viter._g = self
        return viter

    def iter_vertices(self, vprops=[]):
        """Return an iterator over the vertex indices, and optional vertex properties
        list ``vprops``.

        .. note::

           This mode of iteration is more efficient than using
           :meth:`~graph_tool.Graph.vertices`, as descriptor objects are not
           created.

        Examples
        --------
        >>> g = gt.Graph()
        >>> g.add_vertex(5)
        <...>
        >>> for v in g.iter_vertices():
        ...     print(v)
        0
        1
        2
        3
        4
        """
        viter = libcore.get_vertex_iter(self.__graph, 0,
                                        [vp._get_any() for vp in vprops])
        viter._g = self
        return viter

    def get_vertices(self, vprops=[]):
        """Return a :class:`numpy.ndarray` containing the vertex indices, and optional
        vertex properties list ``vprops``. If ``vprops`` is not empty, the shape
        of the array will be ``(V, 1 + len(vprops))``, where ``V`` is the number
        of vertices, and each line will contain the vertex and the vertex
        property values.

        .. note::

           The order of the vertices is identical to
           :meth:`~graph_tool.Graph.vertices`.

        Examples
        --------
        >>> g = gt.Graph()
        >>> g.add_vertex(5)
        <...>
        >>> g.get_vertices()
        array([0, 1, 2, 3, 4])

        """
        vertices = libcore.get_vertex_list(self.__graph, 0,
                                           [vp._get_any() for vp in vprops])
        if len(vprops) == 0:
            return vertices
        else:
            V = vertices.shape[0] // (1 + len(vprops))
            return numpy.reshape(vertices, (V, 1 + len(vprops)))

    def vertex(self, i, use_index=True, add_missing=False):
        """Return the vertex with index ``i``. If ``use_index=False``, the
        ``i``-th vertex is returned (which can differ from the vertex with index
        ``i`` in case of filtered graphs).

        If ``add_missing == True``, and the vertex does not exist in the graph,
        the necessary number of missing vertices are inserted, and the new
        vertex is returned.
        """
        v = libcore.get_vertex(self.__graph, int(i), use_index)
        if not v.is_valid():
            if add_missing:
                self.add_vertex(int(i) - self.num_vertices(use_index) + 1)
                return self.vertex(int(i), use_index)
            raise ValueError("Invalid vertex index: %d" % int(i))
        return v

    def edge(self, s, t, all_edges=False, add_missing=False):
        r"""Return the edge from vertex ``s`` to ``t``, if it exists. If
        ``all_edges=True`` then a list is returned with all the parallel edges
        from ``s`` to ``t``, otherwise only one edge is returned.

        If ``add_missing == True``, a new edge is created and returned, if none
        currently exists.

        This operation will take :math:`O(\min(k(s), k(t)))` time, where
        :math:`k(s)` and :math:`k(t)` are the out-degree and in-degree (or
        out-degree if undirected) of vertices :math:`s` and :math:`t`.

        """
        s = self.vertex(int(s))
        t = self.vertex(int(t))
        edges = libcore.get_edge(self.__graph, int(s), int(t), all_edges)
        if add_missing and len(edges) == 0:
            edges.append(self.add_edge(s, t))
        if all_edges:
            return edges
        elif len(edges) > 0:
            return edges[0]
        else:
            return None

    def edges(self):
        """Return an :meth:`iterator <iterator.__iter__>` over the edges.

        .. note::

           The order of the edges traversed by the iterator **does not**
           necessarily correspond to the edge index ordering, as given by the
           :attr:`~graph_tool.Graph.edge_index` property map. This will only
           happen after :meth:`~graph_tool.Graph.reindex_edges` is called, or in
           certain situations such as just after a graph is loaded from a
           file. However, further manipulation of the graph may destroy the
           ordering.

        """
        eiter = libcore.get_edges(self.__graph)
        eiter._g = self
        return eiter

    def iter_edges(self, eprops=[]):
        """Return an iterator over the edge ```(source, target)`` pairs, and optional
        edge properties list ``eprops``.

        .. note::

           This mode of iteration is more efficient than using
           :meth:`~graph_tool.Graph.edges`, as descriptor objects are not
           created.

        Examples
        --------
        >>> g = gt.collection.data["karate"]
        >>> for s, t, i in g.iter_edges([g.edge_index]):
        ...     print(s, t, i)
        ...     if s == 5:
        ...         break
        1 0 0
        2 0 1
        2 1 2
        3 0 3
        3 1 4
        3 2 5
        4 0 6
        5 0 7
        """
        eiter = libcore.get_edge_iter(self.__graph, 0,
                                      [ep._get_any() for ep in eprops])
        eiter._g = self
        return eiter

    def get_edges(self, eprops=[]):
        """Return a :class:`numpy.ndarray` containing the edges, and optional edge
        properties list ``eprops``. The shape of the array will be ``(E, 2 +
        len(eprops))``, where ``E`` is the number of edges, and each line will
        contain the source, target and the edge property values.

        .. note::

           The order of the edges is identical to
           :meth:`~graph_tool.Graph.edges`.

        Examples
        --------
        >>> g = gt.random_graph(6, lambda: 1, directed=False)
        >>> g.get_edges([g.edge_index])
        array([[3, 2, 0],
               [4, 1, 2],
               [5, 0, 1]])

        """
        edges = libcore.get_edge_list(self.__graph, 0,
                                      [ep._get_any() for ep in eprops])
        E = edges.shape[0] // (2 + len(eprops))
        return numpy.reshape(edges, (E, 2 + len(eprops)))

    def iter_out_edges(self, v, eprops=[]):
        """Return an iterator over the out-edge ```(source, target)`` pairs for vertex
        ``v``, and optional edge properties list ``eprops``.

        .. note::

           This mode of iteration is more efficient than using
           :meth:`~graph_tool.Vertex.out_edges`, as descriptor objects are not
           created.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> for s, t, i in g.iter_out_edges(66, [g.edge_index]):
        ...     print(s, t, i)
        66 63 5266
        66 20369 5267
        66 13980 5268
        66 8687 5269
        66 38674 5270
        """
        eiter = libcore.get_out_edge_iter(self.__graph, int(v),
                                          [ep._get_any() for ep in eprops])
        eiter._g = self
        return eiter

    def get_out_edges(self, v, eprops=[]):
        """Return a :class:`numpy.ndarray` containing the out-edges of vertex ``v``, and
        optional edge properties list ``eprops``. The shape of the array will be
        ``(E, 2 + len(eprops))``, where ``E`` is the number of edges, and each
        line will contain the source, target and the edge property values.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> g.get_out_edges(66, [g.edge_index])
        array([[   66,    63,  5266],
               [   66, 20369,  5267],
               [   66, 13980,  5268],
               [   66,  8687,  5269],
               [   66, 38674,  5270]])

        """
        edges = libcore.get_out_edge_list(self.__graph, int(v),
                                          [ep._get_any() for ep in eprops])
        E = edges.shape[0] // (2 + len(eprops))
        return numpy.reshape(edges, (E, 2 + len(eprops)))

    def iter_in_edges(self, v, eprops=[]):
        """Return an iterator over the in-edge ```(source, target)`` pairs for vertex
        ``v``, and optional edge properties list ``eprops``.

        .. note::

           This mode of iteration is more efficient than using
           :meth:`~graph_tool.Vertex.in_edges`, as descriptor objects are not
           created.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> for s, t, i in g.iter_in_edges(66, [g.edge_index]):
        ...     print(s, t, i)
        8687 66 179681
        20369 66 255033
        38674 66 300230
        """
        eiter = libcore.get_in_edge_iter(self.__graph, int(v),
                                         [ep._get_any() for ep in eprops])
        eiter._g = self
        return eiter

    def get_in_edges(self, v, eprops=[]):
        """Return a :class:`numpy.ndarray` containing the in-edges of vertex ``v``, and
        optional edge properties list ``eprops``. The shape of the array will be
        ``(E, 2 + len(eprops))``, where ``E`` is the number of edges, and each
        line will contain the source, target and the edge property values.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> g.get_in_edges(66, [g.edge_index])
        array([[  8687,     66, 179681],
               [ 20369,     66, 255033],
               [ 38674,     66, 300230]])
        """
        edges = libcore.get_in_edge_list(self.__graph, int(v),
                                          [ep._get_any() for ep in eprops])
        E = edges.shape[0] // (2 + len(eprops))
        return numpy.reshape(edges, (E, 2 + len(eprops)))

    def iter_all_edges(self, v, eprops=[]):
        """Return an iterator over the in- and out-edge ```(source, target)`` pairs for
        vertex ``v``, and optional edge properties list ``eprops``.

        .. note::

           This mode of iteration is more efficient than using
           :meth:`~graph_tool.Vertex.all_edges`, as descriptor objects are not
           created.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> for s, t, i in g.iter_all_edges(66, [g.edge_index]):
        ...     print(s, t, i)
        66 63 5266
        66 20369 5267
        66 13980 5268
        66 8687 5269
        66 38674 5270
        8687 66 179681
        20369 66 255033
        38674 66 300230
        """
        eiter = libcore.get_all_edge_iter(self.__graph, int(v),
                                          [ep._get_any() for ep in eprops])
        eiter._g = self
        return eiter

    def get_all_edges(self, v, eprops=[]):
        """Return a :class:`numpy.ndarray` containing the in- and out-edges of vertex v,
        and optional edge properties list ``eprops``. The shape of the array
        will be ``(E, 2 + len(eprops))``, where ``E`` is the number of edges,
        and each line will contain the source, target and the edge property
        values.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> g.get_all_edges(66, [g.edge_index])
        array([[    66,     63,   5266],
               [    66,  20369,   5267],
               [    66,  13980,   5268],
               [    66,   8687,   5269],
               [    66,  38674,   5270],
               [  8687,     66, 179681],
               [ 20369,     66, 255033],
               [ 38674,     66, 300230]])
        """
        edges = libcore.get_all_edge_list(self.__graph, int(v),
                                          [ep._get_any() for ep in eprops])
        E = edges.shape[0] // (2 + len(eprops))
        return numpy.reshape(edges, (E, 2 + len(eprops)))

    def iter_out_neighbors(self, v, vprops=[]):
        """Return an iterator over the out-neighbors of vertex ``v``, and optional vertex
        properties list ``vprops``.

        .. note::

           This mode of iteration is more efficient than using
           :meth:`~graph_tool.Vertex.out_neighbors`, as descriptor objects are
           not created.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> for u, i in g.iter_out_neighbors(66, [g.vp.uid]):
        ...     print(u, i)
        63 ['paul wilders <webmaster@wilders.org>']
        20369 ['Zhen-Xjell <zhen-xjell@teamhelix.net>']
        13980 ['Hooman <Hooman@iname.com>']
        8687 ['H. Loeung (howe81) <howe81@unixque.com>', 'howe81 <howe81@bigpond.net.au>', 'Howie L (howe81) <howe81@bigpond.net.au>']
        38674 ['Howie L (howe81) <howe81@bigpond.net.au>']
        """
        viter = libcore.get_out_neighbors_iter(self.__graph, int(v),
                                               [vp._get_any() for vp in vprops])
        viter._g = self
        return viter

    iter_out_neighbours = iter_out_neighbors

    def get_out_neighbors(self, v, vprops=[]):
        """Return a :class:`numpy.ndarray` containing the out-neighbors of vertex ``v``,
        and optional vertex properties list ``vprops``. If ``vprops`` is not
        empty, the shape of the array will be ``(V, 1 + len(eprops))``, where
        ``V`` is the number of vertices, and each line will contain a vertex and
        its property values.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> g.get_out_neighbors(66)
        array([   63, 20369, 13980,  8687, 38674])

        """
        vertices = libcore.get_out_neighbors_list(self.__graph, int(v),
                                                  [vp._get_any() for vp in vprops])
        if len(vprops) == 0:
            return vertices
        else:
            V = vertices.shape[0] // (1 + len(vprops))
            return numpy.reshape(vertices, (V, 1 + len(vprops)))

    get_out_neighbours = get_out_neighbors

    def iter_in_neighbors(self, v, vprops=[]):
        """Return an iterator over the in-neighbors of vertex ``v``, and optional vertex
        properties list ``vprops``.

        .. note::

           This mode of iteration is more efficient than using
           :meth:`~graph_tool.Vertex.in_neighbors`, as descriptor objects are
           not created.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> for u, i in g.iter_in_neighbors(66, [g.vp.uid]):
        ...     print(u, i)
        8687 ['H. Loeung (howe81) <howe81@unixque.com>', 'howe81 <howe81@bigpond.net.au>', 'Howie L (howe81) <howe81@bigpond.net.au>']
        20369 ['Zhen-Xjell <zhen-xjell@teamhelix.net>']
        38674 ['Howie L (howe81) <howe81@bigpond.net.au>']
        """
        viter = libcore.get_in_neighbors_iter(self.__graph, int(v),
                                              [vp._get_any() for vp in vprops])
        viter._g = self
        return viter

    iter_in_neighbours = iter_in_neighbors

    def get_in_neighbors(self, v, vprops=[]):
        """Return a :class:`numpy.ndarray` containing the in-neighbors of vertex ``v``,
        and optional vertex properties list ``vprops``. If ``vprops`` is not
        empty, the shape of the array will be ``(V, 1 + len(eprops))``, where
        ``V`` is the number of vertices, and each line will contain a vertex and
        its property values.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> g.get_in_neighbors(66)
        array([ 8687, 20369, 38674])

        """
        vertices = libcore.get_in_neighbors_list(self.__graph, int(v),
                                                 [vp._get_any() for vp in vprops])
        if len(vprops) == 0:
            return vertices
        else:
            V = vertices.shape[0] // (1 + len(vprops))
            return numpy.reshape(vertices, (V, 1 + len(vprops)))

    get_in_neighbours = get_in_neighbors

    def iter_all_neighbors(self, v, vprops=[]):
        """Return an iterator over the in- and out-neighbors of vertex ``v``, and
        optional vertex properties list ``vprops``.

        .. note::

           This mode of iteration is more efficient than using
           :meth:`~graph_tool.Vertex.all_neighbors`, as descriptor objects are
           not created.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> for u, i in g.iter_all_neighbors(66, [g.vp.uid]):
        ...     print(u, i)
        63 ['paul wilders <webmaster@wilders.org>']
        20369 ['Zhen-Xjell <zhen-xjell@teamhelix.net>']
        13980 ['Hooman <Hooman@iname.com>']
        8687 ['H. Loeung (howe81) <howe81@unixque.com>', 'howe81 <howe81@bigpond.net.au>', 'Howie L (howe81) <howe81@bigpond.net.au>']
        38674 ['Howie L (howe81) <howe81@bigpond.net.au>']
        8687 ['H. Loeung (howe81) <howe81@unixque.com>', 'howe81 <howe81@bigpond.net.au>', 'Howie L (howe81) <howe81@bigpond.net.au>']
        20369 ['Zhen-Xjell <zhen-xjell@teamhelix.net>']
        38674 ['Howie L (howe81) <howe81@bigpond.net.au>']
        """
        viter = libcore.get_all_neighbors_iter(self.__graph, int(v),
                                               [vp._get_any() for vp in vprops])
        viter._g = self
        return viter

    iter_all_neighbours = iter_all_neighbors

    def get_all_neighbors(self, v, vprops=[]):
        """Return a :class:`numpy.ndarray` containing the in-neighbors and
        out-neighbors of vertex ``v``, and optional vertex properties list
        ``vprops``. If ``vprops`` is not empty, the shape of the array will be
        ``(V, 1 + len(eprops))``, where ``V`` is the number of vertices, and
        each line will contain a vertex and its property values.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> g.get_all_neighbors(66)
        array([   63, 20369, 13980,  8687, 38674,  8687, 20369, 38674])
        """
        vertices = libcore.get_all_neighbors_list(self.__graph, int(v),
                                                 [vp._get_any() for vp in vprops])
        if len(vprops) == 0:
            return vertices
        else:
            V = vertices.shape[0] // (1 + len(vprops))
            return numpy.reshape(vertices, (V, 1 + len(vprops)))

    get_all_neighbours = get_all_neighbors

    def get_out_degrees(self, vs, eweight=None):
        """Return a :class:`numpy.ndarray` containing the out-degrees of vertex list
        ``vs``. If supplied, the degrees will be weighted according to the edge
        :class:`~graph_tool.EdgePropertyMap` ``eweight``.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> g.get_out_degrees([42, 666])
        array([20, 38], dtype=uint64)
        """
        return libcore.get_degree_list(self.__graph,
                                       numpy.asarray(vs, dtype="uint64"),
                                       _prop("e", self, eweight), 0)

    def get_in_degrees(self, vs, eweight=None):
        """Return a :class:`numpy.ndarray` containing the in-degrees of vertex list
        ``vs``. If supplied, the degrees will be weighted according to the edge
        :class:`~graph_tool.EdgePropertyMap` ``eweight``.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> g.get_in_degrees([42, 666])
        array([20, 39], dtype=uint64)
        """
        return libcore.get_degree_list(self.__graph,
                                       numpy.asarray(vs, dtype="uint64"),
                                       _prop("e", self, eweight), 1)

    def get_total_degrees(self, vs, eweight=None):
        """Return a :class:`numpy.ndarray` containing the total degrees (i.e. in- plus
        out-degree) of vertex list ``vs``. If supplied, the degrees will be
        weighted according to the edge :class:`~graph_tool.EdgePropertyMap`
        ``eweight``.

        Examples
        --------
        >>> g = gt.collection.data["pgp-strong-2009"]
        >>> g.get_total_degrees([42, 666])
        array([40, 77], dtype=uint64)
        """
        return libcore.get_degree_list(self.__graph,
                                       numpy.asarray(vs, dtype="uint64"),
                                       _prop("e", self, eweight), 2)

    def add_vertex(self, n=1):
        """Add a vertex to the graph, and return it. If ``n != 1``, ``n``
        vertices are inserted and an iterator over the new vertices is returned.
        This operation is :math:`O(n)`.
        """
        v = libcore.add_vertex(self.__graph, n)

        if n == 1:
            return v
        else:
            pos = self.num_vertices(True) - n
            return (self.vertex(i) for i in range(pos, pos + n))

    def remove_vertex(self, vertex, fast=False):
        r"""Remove a vertex from the graph. If ``vertex`` is an iterable, it
        should correspond to a sequence of vertices to be removed.

        .. note::

           If the option ``fast == False`` is given, this operation is
           :math:`O(V + E)` (this is the default). Otherwise it is
           :math:`O(k + k_{\text{last}})`, where :math:`k` is the (total)
           degree of the vertex being deleted, and :math:`k_{\text{last}}` is
           the (total) degree of the vertex with the largest index.

        .. warning::

           This operation may invalidate vertex descriptors. Vertices are always
           indexed contiguously in the range :math:`[0, N-1]`, hence vertex
           descriptors with an index higher than ``vertex`` will be invalidated
           after removal (if ``fast == False``, otherwise only descriptors
           pointing to vertices with the largest index will be invalidated).

           Because of this, the only safe way to remove more than one vertex at
           once is to sort them in decreasing index order:

           .. code::

               # 'del_list' is a list of vertex descriptors
               for v in reversed(sorted(del_list)):
                   g.remove_vertex(v)

           Alternatively (and preferably), a list (or iterable) may be passed
           directly as the ``vertex`` parameter, and the above is performed
           internally (in C++).

        .. warning::

           If ``fast == True``, the vertex being deleted is 'swapped' with the
           last vertex (i.e. with the largest index), which will in turn inherit
           the index of the vertex being deleted. All property maps associated
           with the graph will be properly updated, but the index ordering of
           the graph will no longer be the same.

        """
        back = self.__graph.get_num_vertices(False) - 1
        is_iter = isinstance(vertex, collections.abc.Iterable)
        if is_iter:
            try:
                vs = numpy.asarray(vertex, dtype="int64")
            except TypeError:
                vs = numpy.asarray([int(v) for v in vertex], dtype="int64")
            if len(vs) == 0:
                return
            vs = numpy.unique(vs)[::-1]
            vmax, vmin = vs[0], vs[-1]
        else:
            vmax = vmin = int(vertex)
            vs = numpy.asarray((vertex,), dtype="int64")

        if vmax > back:
            raise ValueError("Vertex index %d is invalid" % vmax)

        # move / shift all known property maps
        # (unless removing a contiguous range at the end)
        if vmax - vmin >= len(vs) or vmax != back:
            vfilt = self.get_vertex_filter()[0]
            if vfilt is not None:
                vfiltptr = vfilt.data_ptr()
            else:
                vfiltptr = None
            for pmap_ in self.__known_properties.values():
                pmap = pmap_()
                if (pmap is not None and
                    pmap.key_type() == "v" and
                    pmap.is_writable() and
                    pmap.data_ptr() != vfiltptr):
                    if fast:
                        self.__graph.move_vertex_property(_prop("v", self, pmap), vs)
                    else:
                        self.__graph.shift_vertex_property(_prop("v", self, pmap), vs)

        if is_iter:
            libcore.remove_vertex_array(self.__graph, vs, fast)
        else:
            libcore.remove_vertex(self.__graph, vertex, fast)

    def clear_vertex(self, vertex):
        """Remove all in and out-edges from the given vertex."""
        libcore.clear_vertex(self.__graph, int(vertex))

    def add_edge(self, source, target, add_missing=True):
        """Add a new edge from ``source`` to ``target`` to the graph, and return
        it. This operation is :math:`O(1)`.

        If ``add_missing == True``, the source and target vertices are included
        in the graph if they don't yet exist.
        """
        e = libcore.add_edge(self.__graph,
                             self.vertex(int(source), add_missing=add_missing),
                             self.vertex(int(target), add_missing=add_missing))
        return e

    def remove_edge(self, edge):
        r"""Remove an edge from the graph.

        .. note::

           This operation is normally :math:`O(k_s + k_t)`, where :math:`k_s`
           and :math:`k_t` are the total degrees of the source and target
           vertices, respectively. However, if :meth:`~Graph.set_fast_edge_removal`
           is set to `True`, this operation becomes :math:`O(1)`.

        .. warning::

           The relative ordering of the remaining edges in the graph is kept
           unchanged, unless :meth:`~Graph.set_fast_edge_removal` is set to
           `True`, in which case it can change.
        """
        if isinstance(edge, (tuple, list)):
            edge = self.edge(edge[0], edge[1])
        return libcore.remove_edge(self.__graph, edge)

    def add_edge_list(self, edge_list, hashed=False, hash_type="string",
                      eprops=None):
        """Add a list of edges to the graph, given by ``edge_list``, which can
        be an iterator of ``(source, target)`` pairs where both ``source`` and
        ``target`` are vertex indices (or can be so converted), or a
        :class:`numpy.ndarray` of shape ``(E,2)``, where ``E`` is the number of
        edges, and each line specifies a ``(source, target)`` pair. If the list
        references vertices which do not exist in the graph, they will be
        created.

        Optionally, if ``hashed == True``, the vertex values in the edge list
        are not assumed to correspond to vertex indices directly. In this case
        they will be mapped to vertex indices according to the order in which
        they are encountered, and a vertex property map with the vertex values
        is returned. The option ``hash_type`` will determine the expected type
        used by the hash keys, and they can be any property map value type (see
        :class:`PropertyMap`), unless ``edge_list`` is a :class:`numpy.ndarray`,
        in which case the value of this option is ignored, and the type is
        determined automatically.

        If ``hashed == False`` and the target value of an edge corresponds to
        the maximum interger value (:data:`sys.maxsize`, or the maximum integer
        type of the :class:`numpy.ndarray` object), or is a :data:`numpy.nan` or
        :data:`numpy.inf` value, then only the source vertex will be added to
        the graph.

        If ``hashed == True``, and the target value corresponds to ``None``,
        then only the source vertex will be added to the graph.

        If given, ``eprops`` should specify an iterable containing edge property
        maps that will be filled with the remaining values at each row, if there
        are more than two. Alternatively, ``eprops`` can contain a list of
        ``(name, value_type)`` pairs, in which case new internal dege property
        maps will be created with the corresponding name name and value type.

        .. note::

           If ``edge_list`` is a :class:`numpy.ndarray` object, the execution of
           this function will be done entirely in C++, and hence much faster.

        Examples
        --------
        >>> edge_list = [(0, 1, .3, 10), (2, 3, .1, 0), (2, 0, .4, 42)]
        >>> g = gt.Graph()
        >>> eweight = g.new_ep("double")
        >>> elayer = g.new_ep("int")
        >>> g.add_edge_list(edge_list, eprops=[eweight, elayer])
        >>> print(eweight.fa)
        [0.3 0.1 0.4]
        >>> g.get_edges()
        array([[0, 1],
               [2, 3],
               [2, 0]])

        """

        if eprops is None:
            eprops = ()
            if not isinstance(edge_list, numpy.ndarray) and not hashed:
                def wrap(elist):
                    for row in elist:
                        yield (int(x) if x is not None else None for x in row)
                edge_list = wrap(edge_list)
        else:
            for i in range(len(eprops)):
                if not isinstance(eprops[i], EdgePropertyMap):
                    name, val_type = eprops[i]
                    eprops[i] = self.ep[name] = self.new_ep(val_type)
            convert = [_converter(x.value_type()) for x in eprops]
            eprops = [_prop("e", self, x) for x in eprops]
            if not isinstance(edge_list, numpy.ndarray):
                def wrap(elist):
                    if hashed:
                        conv = lambda x: x
                    else:
                        conv = lambda x: int(x) if x is not None else None
                    for row in elist:
                        yield (conv(val) if i < 2 else convert[i - 2](val)
                               for (i, val) in enumerate(row)
                               if len(convert) > i - 2)
                edge_list = wrap(edge_list)
        if not hashed:
            if isinstance(edge_list, numpy.ndarray):
                libcore.add_edge_list(self.__graph, edge_list, eprops)
            else:
                libcore.add_edge_list_iter(self.__graph, edge_list, eprops)
        else:
            if isinstance(edge_list, numpy.ndarray):
                vprop = self.new_vertex_property(_gt_type(edge_list.dtype))
            else:
                vprop = self.new_vertex_property(hash_type)
            libcore.add_edge_list_hashed(self.__graph, edge_list,
                                         _prop("v", self, vprop), eprops)
            return vprop

    def set_fast_edge_removal(self, fast=True):
        r"""If ``fast == True`` the fast :math:`O(1)` removal of edges will be
        enabled. This requires an additional data structure of size :math:`O(E)`
        to be kept at all times.  If ``fast == False``, this data structure is
        destroyed."""
        self.__graph.set_keep_epos(fast)

    def get_fast_edge_removal(self):
        r"""Return whether the fast :math:`O(1)` removal of edges is currently
        enabled."""
        return self.__graph.get_keep_epos()

    def clear(self):
        """Remove all vertices and edges from the graph."""
        self.__graph.clear()

    def clear_edges(self):
        """Remove all edges from the graph."""
        self.__graph.clear_edges()

    # Internal property maps
    # ======================

    properties = property(lambda self: self.__properties,
                          doc=
    """Dictionary of internal properties. Keys must always be a tuple, where the
    first element if a string from the set ``{'v', 'e', 'g'}``, representing a
    vertex, edge or graph property, respectively, and the second element is the
    name of the property map.

    Examples
    --------
    >>> g = gt.Graph()
    >>> g.properties[("e", "foo")] = g.new_edge_property("vector<double>")
    >>> del g.properties[("e", "foo")]
    """)

    # vertex properties
    vertex_properties = property(lambda self: self.__vertex_properties,
                                 doc="Dictionary of internal vertex properties. The keys are the property names.")
    vp = property(lambda self: self.__vertex_properties,
                  doc="Alias to :attr:`~Graph.vertex_properties`.")

    # edge properties
    edge_properties = property(lambda self: self.__edge_properties,
                               doc="Dictionary of internal edge properties. The keys are the property names.")
    ep = property(lambda self: self.__edge_properties,
                  doc="Alias to :attr:`~Graph.edge_properties`.")

    # graph properties
    graph_properties = property(lambda self: self.__graph_properties,
                                 doc="Dictionary of internal graph properties. The keys are the property names.")
    gp = property(lambda self: self.__graph_properties,
                  doc="Alias to :attr:`~Graph.graph_properties`.")

    def own_property(self, prop):
        """Return a version of the property map 'prop' (possibly belonging to
        another graph) which is owned by the current graph."""
        PMap = type(prop)
        if not isinstance(prop, (VertexPropertyMap, EdgePropertyMap,
                                 GraphPropertyMap)):  # pickle backward compatibility
            if prop.key_type() == "v":
                PMap = VertexPropertyMap
            elif prop.key_type() == "e":
                PMap = EdgePropertyMap
            elif prop.key_type() == "g":
                PMap = GraphPropertyMap
        return PMap(prop._PropertyMap__map, self)

    def list_properties(self):
        """Print a list of all internal properties.

        Examples
        --------
        >>> g = gt.Graph()
        >>> g.properties[("e", "foo")] = g.new_edge_property("vector<double>")
        >>> g.vertex_properties["foo"] = g.new_vertex_property("double")
        >>> g.vertex_properties["bar"] = g.new_vertex_property("python::object")
        >>> g.graph_properties["gnat"] = g.new_graph_property("string", "hi there!")
        >>> g.list_properties()
        gnat           (graph)   (type: string, val: hi there!)
        bar            (vertex)  (type: python::object)
        foo            (vertex)  (type: double)
        foo            (edge)    (type: vector<double>)
        """

        if len(self.__properties) == 0:
            return
        w = max(max([len(x[1]) for x in list(self.__properties.keys())]) + 4,
                14)

        for k, v in self.graph_properties.items():
            pref="%%-%ds (graph)   (type: %%s, val: " % w % (k, v.value_type())
            val = str(v[self])
            if len(val) > 1000:
                val = val[:1000] + "..."
            tw = terminal_size()[0]
            val = textwrap.fill(val,
                                width=max(tw - len(pref) - 1, 1))
            val = val.replace("\n", "\n" + " " * len(pref))
            print("%s%s)" % (pref, val))
        for k, v in sorted(self.vertex_properties.items(), key=lambda k: k[0]):
            print("%%-%ds (vertex)  (type: %%s)" % w % (k, v.value_type()))
        for k, v in sorted(self.edge_properties.items(), key=lambda k: k[0]):
            print("%%-%ds (edge)    (type: %%s)" % w % (k, v.value_type()))

    # index properties

    def _get_vertex_index(self):
        return self.__vertex_index
    vertex_index = property(_get_vertex_index,
                            doc="""Vertex index map.

                            It maps for each vertex in the graph an unique
                            integer in the range [0, :meth:`~graph_tool.Graph.num_vertices` - 1].

                            .. note::

                                Like :attr:`~graph_tool.Graph.edge_index`, this
                                is a special instance of a :class:`~graph_tool.VertexPropertyMap`
                                class, which is **immutable**, and cannot be
                                accessed as an array.""")

    def _get_edge_index(self):
        return self.__edge_index
    edge_index = property(_get_edge_index, doc="""Edge index map.

                            It maps for each edge in the graph an unique
                            integer.

                            .. note::

                                Like :attr:`~graph_tool.Graph.vertex_index`, this
                                is a special instance of a :class:`~graph_tool.EdgePropertyMap`
                                class, which is **immutable**, and cannot be
                                accessed as an array.

                                Additionally, the indices may not necessarily
                                lie in the range [0, :meth:`~graph_tool.Graph.num_edges` - 1].
                                However this will always happen whenever no
                                edges are deleted from the graph.""")

    def _get_edge_index_range(self):
        return self.__graph.get_edge_index_range()

    edge_index_range = property(_get_edge_index_range,
                                doc="The size of the range of edge indices.")

    def reindex_edges(self):
        """
        Reset the edge indices so that they lie in the [0, :meth:`~graph_tool.Graph.num_edges` - 1]
        range. The index ordering will be compatible with the sequence returned
        by the :meth:`~graph_tool.Graph.edges` function.

        .. warning::

           Calling this function will invalidate all existing edge property
           maps, if the index ordering is modified! The property maps will still
           be usable, but their contents will still be tied to the old indices,
           and thus may become scrambled.
        """
        self.__graph.re_index_edges()


    def shrink_to_fit(self):
        """Force the physical capacity of the underlying containers to match the graph's
        actual size, potentially freeing memory back to the system."""
        self.__graph.shrink_to_fit()

    # Property map creation

    def new_property(self, key_type, value_type, vals=None):

        """Create a new (uninitialized) vertex property map of key type
        ``key_type`` (``v``, ``e`` or ``g``), value type ``value_type``, and
        return it. If provided, the values will be initialized by ``vals``,
        which should be a sequence.
        """
        if key_type == "v" or key_type == "vertex":
            return self.new_vertex_property(value_type, vals)
        if key_type == "e" or key_type == "edge":
            return self.new_edge_property(value_type, vals)
        if key_type == "g" or key_type == "graph":
            return self.new_graph_property(value_type, vals)
        raise ValueError("unknown key type: " + key_type)

    def new_vertex_property(self, value_type, vals=None, val=None):
        """Create a new vertex property map of type ``value_type``, and return it. If
        provided, the values will be initialized by ``vals``, which should be
        sequence or by ``val`` which should be  a single value.
        """
        prop = VertexPropertyMap(new_vertex_property(_type_alias(value_type),
                                                     self.__graph.get_vertex_index(),
                                                     libcore.any()),
                           self)
        if vals is not None:
            try:
                prop.fa = vals
            except (IndexError, ValueError, TypeError):
                for v, x in zip(self.vertices(), vals):
                    prop[v] = x
        elif val is not None:
            prop.set_value(val)
        return prop

    new_vp = _copy_func(new_vertex_property, "new_vp")
    new_vp.__doc__ = "Alias to :func:`~graph_tool.Graph.new_vertex_property`."

    def new_edge_property(self, value_type, vals=None, val=None):
        """Create a new edge property map of type ``value_type``, and return it. If
        provided, the values will be initialized by ``vals``, which should be
        sequence or by ``val`` which should be a single value.
        """
        prop = EdgePropertyMap(new_edge_property(_type_alias(value_type),
                                                 self.__graph.get_edge_index(),
                                                 libcore.any()),
                               self)
        if vals is not None:
            try:
                prop.fa = vals
            except (IndexError, ValueError, TypeError):
                for e, x in zip(self.edges(), vals):
                    prop[e] = x
        elif val is not None:
            prop.set_value(val)
        return prop

    new_ep = _copy_func(new_edge_property, "new_ep")
    new_ep.__doc__ = "Alias to :func:`~graph_tool.Graph.new_edge_property`."

    def new_graph_property(self, value_type, val=None):
        """Create a new graph property map of type ``value_type``, and return
        it. If ``val`` is not None, the property is initialized to its value."""
        prop = GraphPropertyMap(new_graph_property(_type_alias(value_type),
                                                   self.__graph.get_graph_index(),
                                                   libcore.any()),
                                self)
        if val is not None:
            prop[self] = val
        return prop

    new_gp = _copy_func(new_graph_property, "new_gp")
    new_gp.__doc__ = "Alias to :func:`~graph_tool.Graph.new_graph_property`."

    # property map copying
    @_require("src", PropertyMap)
    @_require("tgt", (PropertyMap, type(None)))
    def copy_property(self, src, tgt=None, value_type=None, g=None, full=True):
        """Copy contents of ``src`` property to ``tgt`` property. If ``tgt`` is None,
        then a new property map of the same type (or with the type given by the
        optional ``value_type`` parameter) is created, and returned. The
        optional parameter ``g`` specifies the source graph to copy properties
        from (defaults to the graph than owns `src`). If ``full == False``, then
        in the case of filtered graphs only the unmasked values are copied (with
        the remaining ones taking the type-dependent default value).

        .. note::

           In case the source property map belongs to different graphs, this
           function behaves as follows.

           For vertex properties, the source and target graphs must have the
           same number of vertices, and the properties are copied according to
           the index values.

           For edge properties, the edge index is not important, and the
           properties are copied by matching edges between the different graphs
           according to the source and target vertex indices. In case the graph
           has parallel edges with the same source and target vertices, they are
           matched according to their iteration order. The edge sets do not have
           to be the same between source and target graphs, as the copying
           occurs only for edges that lie at their intersection.

        """
        if tgt is None:
            tgt = self.new_property(src.key_type(),
                                    (src.value_type()
                                     if value_type is None else value_type))
            ret = tgt
        else:
            ret = None

        if src.key_type() != tgt.key_type():
            raise ValueError("source and target properties must have the same key type")

        if g is None:
            g = src.get_graph()
        sf = self

        if full:
            if g is sf:
                g = GraphView(g, skip_properties=True, skip_efilt=True,
                              skip_vfilt=True)
                sf = g
            else:
                g = GraphView(g, skip_properties=True, skip_efilt=True,
                              skip_vfilt=True)
                sf = GraphView(sf, skip_properties=True, skip_efilt=True,
                               skip_vfilt=True)

        if src.key_type() == "v":
            if g.num_vertices() > sf.num_vertices():
                raise ValueError("graphs with incompatible number of vertices (%d, %d)" %
                                 (g.num_vertices(), sf.num_vertices()))
            try:
                sf.__graph.copy_vertex_property(g.__graph,
                                                _prop("v", g, src),
                                                _prop("v", sf, tgt))
            except ValueError as e:
                raise ValueError("error copying maps with types %s, %s: %s" %
                                 (src.value_type(), tgt.value_type(), str(e)))
        elif src.key_type() == "e":
            if g.num_edges() > sf.num_edges():
                raise ValueError("graphs with incompatible number of edges (%d, %d)" %
                                 (g.num_edges(), sf.num_edges()))
            try:
                if g is sf:
                    sf.__graph.copy_edge_property(g.__graph,
                                                  _prop("e", g, src),
                                                  _prop("e", sf, tgt))
                else:
                    libcore.copy_external_edge_property(g.__graph,
                                                        self.__graph,
                                                        _prop("e", g, src),
                                                        _prop("e", sf, tgt))
            except ValueError as e:
                raise ValueError("error copying maps with types %s, %s: %s" %
                                 (src.value_type(), tgt.value_type(), str(e)))
        else:
            tgt[sf] = src[g]
        return ret

    # degree property map
    @_limit_args({"deg": ["in", "out", "total"]})
    def degree_property_map(self, deg, weight=None):
        """Create and return a vertex property map containing the degree type
        given by ``deg``, which can be any of ``"in"``, ``"out"``, or ``"total"``.
        If provided, ``weight`` should be an edge :class:`~graph_tool.EdgePropertyMap`
        containing the edge weights which should be summed."""
        pmap = self.__graph.degree_map(deg, _prop("e", self, weight))
        return VertexPropertyMap(pmap, self)

    # I/O operations
    # ==============
    def __get_file_format(self, file_name):
        fmt = None
        for f in ["gt", "graphml", "xml", "dot", "gml"]:
            names = ["." + f, ".%s.gz" % f, ".%s.bz2" % f, ".%s.xz" % f,
                     ".%s.zst" % f]
            for name in names:
                if file_name.endswith(name):
                    fmt = f
                    break
        if fmt is None:
            raise ValueError("cannot determine file format of: " + file_name)
        return fmt

    def load(self, file_name, fmt="auto", ignore_vp=None, ignore_ep=None,
             ignore_gp=None):
        """Load graph from ``file_name`` (which can be either a string or a file-like
        object). The format is guessed from ``file_name``, or can be specified
        by ``fmt``, which can be either "gt", "graphml", "xml", "dot" or "gml".
        (Note that "graphml" and "xml" are synonyms).

        If provided, the parameters ``ignore_vp``, ``ignore_ep`` and
        ``ignore_gp``, should contain a list of property names (vertex, edge or
        graph, respectively) which should be ignored when reading the file.

        .. warning::

           The only file formats which are capable of perfectly preserving the
           internal property maps are "gt" and "graphml". Because of this,
           they should be preferred over the other formats whenever possible.

        """

        if isinstance(file_name, str):
            file_name = os.path.expanduser(file_name)
            with open(file_name) as f: # throw the appropriate exception
                pass
        if fmt == 'auto' and isinstance(file_name, str):
            fmt = self.__get_file_format(file_name)
        elif fmt == "auto":
            fmt = "gt"
        fctx = contextlib.suppress()
        if isinstance(file_name, str):
            if file_name.endswith(".xz"):
                try:
                    fctx = file_name = lzma.open(file_name, mode="rb")
                except NameError:
                    raise NotImplementedError("lzma compression is only available in Python >= 3.3")
            elif file_name.endswith(".zst"):
                try:
                    cctx = zstandard.ZstdDecompressor()
                except NameError:
                    raise NotImplementedError("zstandard module not installed, but it's required for decompression")
                fctx = contextlib.ExitStack()
                f = open(file_name, mode="rb")
                fctx.enter_context(f)
                file_name = cctx.stream_reader(f)
                fctx.enter_context(file_name)
        if fmt == "graphml":
            fmt = "xml"
        if ignore_vp is None:
            ignore_vp = []
        if ignore_ep is None:
            ignore_ep = []
        if ignore_gp is None:
            ignore_gp = []
        if isinstance(file_name, str):
            props = self.__graph.read_from_file(file_name, None,
                                                fmt, ignore_vp,
                                                ignore_ep, ignore_gp)
        else:
            with fctx:
                props = self.__graph.read_from_file("", file_name, fmt,
                                                    ignore_vp, ignore_ep,
                                                    ignore_gp)
        for name, prop in props[0].items():
            self.vp[name] = VertexPropertyMap(prop, self)
        for name, prop in props[1].items():
            self.ep[name] = EdgePropertyMap(prop, self)
        for name, prop in props[2].items():
            self.gp[name] = GraphPropertyMap(prop, self)
        if "_Graph__save__vfilter" in self.gp:
            self.set_vertex_filter(self.vp["_Graph__save__vfilter"],
                                   self.gp["_Graph__save__vfilter"])
            del self.vp["_Graph__save__vfilter"]
            del self.gp["_Graph__save__vfilter"]
        if "_Graph__save__efilter" in self.gp:
            self.set_edge_filter(self.ep["_Graph__save__efilter"],
                                 self.gp["_Graph__save__efilter"])
            del self.ep["_Graph__save__efilter"]
            del self.gp["_Graph__save__efilter"]
        if "_Graph__reversed" in self.gp:
            self.set_reversed(True)
            del self.gp["_Graph__reversed"]
        if "_Graph__fast_edge_removal" in self.gp:
            self.set_fast_edge_removal(self.gp["_Graph__fast_edge_removal"])
            del self.gp["_Graph__fast_edge_removal"]

        self.shrink_to_fit()

    def save(self, file_name, fmt="auto"):
        """Save graph to ``file_name`` (which can be either a string or a file-like
        object). The format is guessed from the ``file_name``, or can be
        specified by ``fmt``, which can be either "gt", "graphml", "xml", "dot"
        or "gml".  (Note that "graphml" and "xml" are synonyms).

        .. warning::

           The only file formats which are capable of perfectly preserving the
           internal property maps are "gt" and "graphml". Because of this,
           they should be preferred over the other formats whenever possible.

        """

        u = GraphView(self, reversed=self.is_reversed(), skip_vfilt=True,
                      skip_efilt=True)

        if self.get_vertex_filter()[0] is not None:
            u.gp["_Graph__save__vfilter"] = \
                self.new_gp("bool", val= self.get_vertex_filter()[1])
            u.vp["_Graph__save__vfilter"] = self.get_vertex_filter()[0]
        if self.get_edge_filter()[0] is not None:
            u.gp["_Graph__save__efilter"] = \
                self.new_gp("bool", val=self.get_edge_filter()[1])
            u.ep["_Graph__save__efilter"] = self.get_edge_filter()[0]

        if self.is_reversed():
            u.gp["_Graph__reversed"] = self.new_gp("bool", val=True)

        if self.get_fast_edge_removal():
            u.gp["_Graph__fast_edge_removal"] = self.new_gp("bool", val=True)

        if isinstance(file_name, str):
            file_name = os.path.expanduser(file_name)
        if fmt == 'auto' and isinstance(file_name, str):
            fmt = self.__get_file_format(file_name)
        elif fmt == "auto":
            fmt = "gt"
        if fmt == "graphml":
            fmt = "xml"

        fctx = contextlib.suppress()
        if isinstance(file_name, str):
            if file_name.endswith(".xz"):
                try:
                    fctx = file_name = lzma.open(file_name, mode="wb")
                except NameError:
                    raise NotImplementedError("lzma compression is only available in Python >= 3.3")
            elif file_name.endswith(".zst"):
                try:
                    cctx = zstandard.ZstdCompressor(level=19)
                except NameError:
                    raise NotImplementedError("zstandard module not installed, but it's required for compression")
                fctx = contextlib.ExitStack()
                f = open(file_name, mode="wb")
                fctx.enter_context(f)
                file_name = cctx.stream_writer(f)
                fctx.enter_context(file_name)

        props = [(name[1], prop._PropertyMap__map) for name, prop in \
                 u.__properties.items()]

        if isinstance(file_name, str):
            with open(file_name, "w"): # throw the appropriate exception, if
                pass                   # unable to open
            u.__graph.write_to_file(file_name, None, fmt, props)
        else:
            with fctx:
                u.__graph.write_to_file("", file_name, fmt, props)


    # Directedness
    # ============

    def set_directed(self, is_directed):
        """Set the directedness of the graph.

        .. note::

           This is a :math:`O(1)` operation that does not modify the storage of
           the graph.

        .. warning::

           Changing directedness will invalidate existing vertex and edge
           descriptors, which will still point to the original graph.

        """
        self.__graph.set_directed(is_directed)

    def is_directed(self):
        """Get the directedness of the graph."""
        return self.__graph.get_directed()

    # Reversedness
    # ============

    def set_reversed(self, is_reversed):
        """Reverse the direction of the edges, if ``is_reversed`` is ``True``,
        or maintain the original direction otherwise.

        .. note::

           This is a :math:`O(1)` operation that does not modify the storage of
           the graph.

        .. warning::

           Reversing the graph will invalidate existing vertex and edge
           descriptors, which will still point to the original graph.
        """
        self.__graph.set_reversed(is_reversed)

    def is_reversed(self):
        """Return ``True`` if the edges are reversed, and ``False`` otherwise.
        """
        return self.__graph.get_reversed()

    # Filtering
    # =========

    def set_filters(self, eprop, vprop, inverted_edges=False, inverted_vertices=False):
        """Set the boolean properties for edge and vertex filters, respectively.  Only
        the vertices and edges with value different than ``False`` are kept in
        the filtered graph. If either the ``inverted_edges`` or
        ``inverted_vertex`` options are supplied with the value ``True``, only
        the edges or vertices with value ``False`` are kept. If any of the
        supplied property is ``None``, an empty filter is constructed which
        allows all edges or vertices.

        .. note::

           This is a :math:`O(1)` operation that does not modify the storage of
           the graph.

        .. warning::

           Setting vertex or edge filters will invalidate existing vertex and
           edge descriptors, which will still point to the unfiltered graph.

        """

        if eprop is None and vprop is None:
            return

        if eprop is None:
            eprop = self.new_edge_property("bool")
            eprop.a = not inverted_edges
        else:
            eprop = self.own_property(eprop)

        if vprop is None:
            vprop = self.new_vertex_property("bool")
            vprop.a = not inverted_vertices
        else:
            vprop = self.own_property(vprop)

        self.__graph.set_vertex_filter_property(_prop("v", self, vprop),
                                                inverted_vertices)
        self.__filter_state["vertex_filter"] = (vprop, inverted_vertices)

        self.__graph.set_edge_filter_property(_prop("e", self, eprop),
                                              inverted_edges)
        self.__filter_state["edge_filter"] = (eprop, inverted_edges)

    def set_vertex_filter(self, prop, inverted=False):
        """Set the vertex boolean filter property. Only the vertices with value
        different than ``False`` are kept in the filtered graph. If the ``inverted``
        option is supplied with value ``True``, only the vertices with value
        ``False`` are kept. If the supplied property is ``None``, the filter is
        replaced by an uniform filter allowing all vertices.

        .. note::

           This is a :math:`O(1)` operation that does not modify the storage of
           the graph.

        .. warning::

           Setting vertex filters will invalidate existing vertex and edge
           descriptors, which will still point to the unfiltered graph.

        """

        if prop is not None and prop.value_type() != "bool":
            raise ValueError("filter property map must have 'bool' type")

        vfilt = self.own_property(prop) if prop is not None else prop
        efilt = None

        eprop = self.get_edge_filter()
        if eprop[0] is None and vfilt is not None:
            efilt = self.new_edge_property("bool")
            efilt.a = True
        if eprop[0] is not None and vfilt is None:
            vfilt = self.new_vertex_property("bool")
            vfilt.a = not inverted

        self.__graph.set_vertex_filter_property(_prop("v", self, vfilt),
                                                inverted)
        self.__filter_state["vertex_filter"] = (vfilt, inverted)

        if efilt is not None:
            self.set_edge_filter(efilt)

    def get_vertex_filter(self):
        """Return a tuple with the vertex filter property and bool value
        indicating whether or not it is inverted."""
        return self.__filter_state["vertex_filter"]

    def set_edge_filter(self, prop, inverted=False):
        """Set the edge boolean filter property. Only the edges with value
        different than ``False`` are kept in the filtered graph. If the ``inverted``
        option is supplied with value ``True``, only the edges with value ``False``
        are kept. If the supplied property is ``None``, the filter is
        replaced by an uniform filter allowing all edges.

        .. note::

           This is a :math:`O(1)` operation that does not modify the storage of
           the graph.

        .. warning::

           Setting edge filters will invalidate existing vertex and edge
           descriptors, which will still point to the unfiltered graph.

        """

        if prop is not None and prop.value_type() != "bool":
            raise ValueError("filter property map must have 'bool' type")

        efilt = self.own_property(prop) if prop is not None else prop
        vfilt = None

        vprop = self.get_vertex_filter()
        if vprop[0] is None and efilt is not None:
            vfilt = self.new_vertex_property("bool")
            vfilt.a = True
        if vprop[0] is not None and efilt is None:
            efilt = self.new_edge_property("bool")
            efilt.a = not inverted
        self.__graph.set_edge_filter_property(_prop("e", self, efilt), inverted)
        self.__filter_state["edge_filter"] = (efilt, inverted)

        if vfilt is not None:
            self.set_vertex_filter(vfilt)

    def get_edge_filter(self):
        """Return a tuple with the edge filter property and bool value
        indicating whether or not it is inverted."""
        return self.__filter_state["edge_filter"]

    def clear_filters(self):
        """Remove vertex and edge filters, and set the graph to the unfiltered
        state.

        .. note::

           This is a :math:`O(1)` operation that does not modify the storage of
           the graph.

        .. warning::

           Clearing vertex and edge filters will invalidate existing vertex and
           edge descriptors.

        """
        self.__graph.set_vertex_filter_property(_prop("v", self, None), False)
        self.__filter_state["vertex_filter"] = (None, False)
        self.__graph.set_edge_filter_property(_prop("e", self, None), False)
        self.__filter_state["edge_filter"] = (None, False)

    def purge_vertices(self, in_place=False):
        """Remove all vertices of the graph which are currently being filtered out. This
        operation is not reversible.

        If the option ``in_place == True`` is given, the algorithm will remove
        the filtered vertices and re-index all property maps which are tied with
        the graph. This is a slow operation which has an :math:`O(V^2)`
        complexity.

        If ``in_place == False``, the graph and its vertex and edge property
        maps are temporarily copied to a new unfiltered graph, which will
        replace the contents of the original graph. This is a fast operation
        with an :math:`O(V + E)` complexity. This is the default behaviour if no
        option is given.

        .. note :

           The graph will remain in a filtered state after this operation, since
           there might be edge filters present. To return the graph to an
           unfiltered state, use :meth:`~graph_tool.Graph.clear_filters`.

        """
        if in_place:
            old_indices = self.vertex_index.copy("int64_t")
            self.__graph.purge_vertices(_prop("v", self, old_indices))
            self.set_vertex_filter(None)
            for pmap in self.__known_properties.values():
                if (pmap() is not None and pmap().key_type() == "v" and
                    pmap().is_writable() and
                    pmap() not in [self.vertex_index, self.edge_index]):
                    self.__graph.re_index_vertex_property(_prop("v", self, pmap()),
                                                          _prop("v", self, old_indices))
        else:
            stamp = id(self)
            pmaps = []
            for pmap in self.__known_properties.values():
                if (pmap() is not None and pmap().key_type() in ["v", "e"] and
                    pmap() not in [self.vertex_index, self.edge_index]):
                    pmaps.append(pmap())
                    pname = "__tmp_purge_vertices_%d_%d" % (stamp, id(pmaps[-1]))
                    self.properties[(pmaps[-1].key_type(), pname)] = pmaps[-1]

            new_g = Graph(self, prune=(True, False, False))
            if hasattr(self, "_GraphView__base"):
                self._GraphView__base = new_g
            self.__graph = new_g.__graph
            self.set_vertex_filter(None)

            for pmap in pmaps:
                pname = "__tmp_purge_vertices_%d_%d" % (stamp, id(pmap))
                new_pmap = new_g.properties[(pmap.key_type(), pname)]
                pmap._PropertyMap__map = new_pmap._PropertyMap__map
                del self.properties[(pmap.key_type(), pname)]

            # update edge filter if set
            efilt = self.get_edge_filter()
            if efilt[0] is not None:
                self.set_edge_filter(efilt[0], efilt[1])

    def purge_edges(self):
        """Remove all edges of the graph which are currently being filtered out. This
        operation is not reversible.

        .. note :

           The graph will remain in a filtered state after this operation, since
           there might be vertex filters present. To return the graph to an
           unfiltered state, use :meth:`~graph_tool.Graph.clear_filters`.

        """
        self.__graph.purge_edges()
        self.set_edge_filter(None)

    def get_filter_state(self):
        """Return a copy of the filter state of the graph."""
        self.__filter_state["directed"] = self.is_directed()
        self.__filter_state["reversed"] = self.is_reversed()
        return copy.copy(self.__filter_state)

    def set_filter_state(self, state):
        """Set the filter state of the graph."""
        if libcore.graph_filtering_enabled():
            self.set_vertex_filter(state["vertex_filter"][0],
                                   state["vertex_filter"][1])
            self.set_edge_filter(state["edge_filter"][0],
                                 state["edge_filter"][1])
        self.set_directed(state["directed"])
        self.set_reversed(state["reversed"])

    # Basic graph statistics
    # ======================

    def num_vertices(self, ignore_filter=False):
        """Get the number of vertices.

        If ``ignore_filter == True``, vertex filters are ignored.

        .. note::

            If the vertices are being filtered, and ``ignore_filter == False``,
            this operation is :math:`O(V)`. Otherwise it is :math:`O(1)`.

        """
        return self.__graph.get_num_vertices(not ignore_filter)

    def __len__(self):
        """Alias to :meth:`Graph.num_vertices()`."""
        return self.num_vertices()

    def num_edges(self, ignore_filter=False):
        """Get the number of edges.

        If ``ignore_filter == True``, edge filters are ignored.

        .. note::

            If the edges are being filtered, and ``ignore_filter == False``,
            this operation is :math:`O(E)`. Otherwise it is :math:`O(1)`.

        """
        return self.__graph.get_num_edges(not ignore_filter)

    # Pickling support
    # ================

    def __getstate__(self):
        state = dict()
        sio = io.BytesIO()
        self.save(sio, "gt")
        state["blob"] = sio.getvalue()
        return state

    def __setstate__(self, state):
        Graph.__init__(self)
        blob = state["blob"]
        if blob != "":
            sio = io.BytesIO(blob)
            self.load(sio, "gt")

    def __get_base(self):
        return self
    base = property(__get_base, doc="Base graph (self).")

def load_graph(file_name, fmt="auto", ignore_vp=None, ignore_ep=None,
               ignore_gp=None):
    """Load a graph from ``file_name`` (which can be either a string or a file-like object).

    The format is guessed from ``file_name``, or can be specified by ``fmt``,
    which can be either "gt", "graphml", "xml", "dot" or "gml".  (Note that
    "graphml" and "xml" are synonyms).

    If provided, the parameters ``ignore_vp``, ``ignore_ep`` and
    ``ignore_gp``, should contain a list of property names (vertex, edge or
    graph, respectively) which should be ignored when reading the file.

    .. warning::

       The only file formats which are capable of perfectly preserving the
       internal property maps are "gt" and "graphml". Because of this,
       they should be preferred over the other formats whenever possible.

    """
    g = Graph()
    g.load(file_name, fmt, ignore_vp, ignore_ep, ignore_gp)
    return g

def load_graph_from_csv(file_name, directed=False, eprop_types=None,
                        eprop_names=None, hashed=True, hash_type="string",
                        skip_first=False, strip_whitespace=True, ecols=(0,1),
                        csv_options={"delimiter": ",", "quotechar": '"'}):
    """Load a graph from a :mod:`csv` file containing a list of edges and edge
    properties.

    Parameters
    ----------
    file_name : ``str`` or file-like object
        File in :mod:``csv`` format, with edges given in each row.
    directed : ``bool`` (optional, default: ``False``)
        Whether or not the graph is directed.
    eprop_types : list of ``str`` (optional, default: ``None``)
        List of edge property types to be read from remaining columns (if this
        is ``None``, all properties will be of type ``string``.
    eprop_names : list of ``str`` (optional, default: ``None``)
        List of edge property names to be used for the remaining columns (if
        this is ``None``, and ``skip_first`` is ``True`` their values will be
        obtained from the first line, otherwise they will be called ``c1, c2,
        ...``).
    hashed : ``bool`` (optional, default: ``True``)
        If ``True`` the vertex values in the edge list are not assumed to
        correspond to vertex indices directly. In this case they will be mapped
        to vertex indices according to the order in which they are encountered,
        and a vertex property map with the vertex values is returned.
    hash_type : ``str`` (optional, default: ``string``)
        If ``hashed == True``, this will determined the type of the vertex values.
        It can be any property map value type (see :class:`PropertyMap`).
    skip_first : ``bool`` (optional, default: ``False``)
        If ``True`` the first line of the file will be skipped.
    strip_whitespace : ``bool`` (optional, default: ``True``)
        If ``True`` whitespace will be striped from the start and end of values,
        before processing them.
    ecols : pair of ``int`` (optional, default: ``(0,1)``)
        Line columns used as source and target for the edges.
    csv_options : ``dict`` (optional, default: ``{"delimiter": ",", "quotechar": '"'}``)
        Options to be passed to the :func:`csv.reader` parser.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        The loaded graph. It will contain additional columns in the file as
        internal edge property maps. If ``hashed == True``, it will also contain
        an internal vertex property map with the vertex names.

    """
    if isinstance(file_name, str):
        if file_name.endswith(".xz"):
            try:
                file_name = lzma.open(file_name, mode="rt")
            except ImportError:
                raise NotImplementedError("lzma compression is only available in Python >= 3.3")
        elif file_name.endswith(".gz"):
            file_name = gzip.open(file_name, mode="rt")
        elif file_name.endswith(".bz2"):
            file_name = bz2.open(file_name, mode="rt")
        else:
            file_name = open(file_name, "r")
    _csv_options = {"delimiter": ",", "quotechar": '"'}
    if  "dialect" in csv_options:
        _csv_options = csv_options
    else:
        _csv_options.update(csv_options)
    r = csv.reader(file_name, **_csv_options)

    if strip_whitespace:
        def strip(r):
            for row in r:
                yield (x.strip() for x in row)
        r = strip(r)

    if skip_first:
        first_line = list(next(r))

    if ecols != (0, 1):
        def reorder(rows):
            for row in rows:
                row = list(row)
                s = row[ecols[0]]
                t = row[ecols[1]]
                del row[min(ecols)]
                del row[max(ecols)-1]
                yield [s, t] + row
        r = reorder(r)

    if not hashed:
        def conv(rows):
            for row in rows:
                row = list(row)
                row[0] = int(row[0])
                row[1] = int(row[1])
                yield row
        r = conv(r)
    elif hash_type != "string":
        tp = _python_type(hash_type)
        def conv(r):
            for row in r:
                row = list(row)
                row[0] = tp(row[0])
                row[1] = tp(row[1])
                yield row
        r = conv(r)

    line = list(next(r))
    g = Graph(directed=directed)

    if eprop_types is None:
        eprops = [g.new_ep("string") for x in line[2:]]
    else:
        eprops = [g.new_ep(t) for t in eprop_types]

    name = g.add_edge_list(itertools.chain([line], r), hashed=hashed,
                           hash_type=hash_type, eprops=eprops)

    if eprop_names is None and skip_first and len(first_line) == len(line):
        eprop_names = list(first_line)
        del eprop_names[min(ecols)]
        del eprop_names[max(ecols)-1]

    for i, p in enumerate(eprops):
        if eprop_names is not None:
            ename = eprop_names[i]
        else:
            ename = "c%d" % i
        g.ep[ename] = p

    if name is not None:
        g.vp.name = name
    return g


class GraphView(Graph):
    """A view of selected vertices or edges of another graph.

    This class uses shared data from another :class:`~graph_tool.Graph`
    instance, but allows for local filtering of vertices and/or edges, edge
    directionality or reversal. See :ref:`sec_graph_views` for more details and
    examples.

    The existence of a :class:`~graph_tool.GraphView` object does not affect the
    original graph, except if the graph view is modified (addition or removal of
    vertices or edges), in which case the modification is directly reflected in
    the original graph (and vice-versa), since they both point to the same
    underlying data. Because of this, instances of
    :class:`~graph_tool.PropertyMap` can be used interchangeably with a graph
    and its views.

    The argument ``g`` must be an instance of a :class:`~graph_tool.Graph`
    class. If specified, ``vfilt`` and ``efilt`` select which vertices and edges
    are filtered, respectively. These parameters can either be a boolean-valued
    :class:`~graph_tool.PropertyMap` or :class:`numpy.ndarray`, which specify
    which vertices/edges are selected, or an unary function that returns
    ``True`` if a given vertex/edge is to be selected, or ``False`` otherwise.

    The boolean parameter ``directed`` can be used to set the directionality of
    the graph view. If ``directed is None``, the directionality is inherited
    from ``g``.

    If ``reversed == True``, the direction of the edges is reversed.

    If ``vfilt`` or ``efilt`` is anything other than a
    :class:`~graph_tool.PropertyMap` instance, the instantiation running time is
    :math:`O(V)` and :math:`O(E)`, respectively. Otherwise, the running time is
    :math:`O(1)`.

    If either ``skip_properties``, ``skip_vfilt`` or ``skip_efilt`` is ``True``,
    then the internal properties, vertex filter or edge filter of the original
    graph are ignored, respectively.

    """

    def __init__(self, g, vfilt=None, efilt=None, directed=None,
                 reversed=False, skip_properties=False, skip_vfilt=False,
                 skip_efilt=False):
        self.__base = g.base
        Graph.__init__(self)
        # copy graph reference
        self._Graph__graph = libcore.GraphInterface(g._Graph__graph, True,
                                                    [], [],
                                                    _prop("v", g, g.vertex_index))

        if not skip_properties:
            for k, v in g.properties.items():
                self.properties[k] = self.own_property(v)

        # set already existing filters
        if not skip_efilt:
            ef = list(g.get_edge_filter())
            if ef[0] is not None:
                ef[0] = self.own_property(ef[0].copy())
        else:
            ef = [None, False]
        if not skip_vfilt:
            vf = list(g.get_vertex_filter())
            if vf[0] is not None:
                vf[0] = self.own_property(vf[0].copy())
        else:
            vf = [None, False]

        self.set_filters(ef[0], vf[0], ef[1], vf[1])

        if efilt is not None:
            if not isinstance(efilt, PropertyMap):
                emap = self.new_edge_property("bool")
                if isinstance(efilt, collections.abc.Iterable):
                    emap.fa = efilt
                else:
                    for e in g.edges():
                        emap[e] = efilt(e)
                efilt = emap
            efilt = self.own_property(efilt)
            ef = self.get_edge_filter()
            if ef[0] is not None:
                if not ef[1]:
                    ef[0].fa = efilt.fa
                else:
                    ef[0].fa = numpy.logical_not(efilt.fa)
                self.set_edge_filter(ef[0], ef[1])
            else:
                self.set_edge_filter(efilt)

        if vfilt is not None:
            if not isinstance(vfilt, PropertyMap):
                vmap = self.new_vertex_property("bool")
                if isinstance(vfilt, collections.abc.Iterable):
                    vmap.fa = vfilt
                else:
                    for v in g.vertices():
                        vmap[v] = vfilt(v)
                vfilt = vmap
            vfilt = self.own_property(vfilt)
            vf = self.get_vertex_filter()
            if vf[0] is not None:
                if not vf[1]:
                    vf[0].fa = vfilt.fa
                else:
                    vf[0].fa = numpy.logical_not(vfilt.fa)
                self.set_vertex_filter(vf[0], vf[1])
            else:
                self.set_vertex_filter(vfilt)


        if directed is not None:
            self.set_directed(directed)
        if reversed:
            self.set_reversed(not g.is_reversed())

    def __get_base(self):
        return self.__base
    base = property(__get_base, doc="Base graph.")

    # pickling support
    def __getstate__(self):
        return Graph.__getstate__(self)

    def __setstate__(self, state):
        g = Graph()
        g.__setstate__(state)
        self.__init__(g)


def value_types():
    """Return a list of possible properties value types."""
    return libcore.get_property_types()

# Vertex and Edge Types
# =====================
from .libgraph_tool_core import Vertex, Edge, VertexBase, EdgeBase

def _out_neighbors(self):
    """Return an iterator over the out-neighbors."""
    for e in self.out_edges():
        yield e.target()

def _in_neighbors(self):
    """Return an iterator over the in-neighbors."""
    for e in self.in_edges():
        yield e.source()

def _all_edges(self):
    """Return an iterator over all edges (both in or out)."""
    for e in self.out_edges():
        yield e
    for e in self.in_edges():
        yield e

def _all_neighbors(self):
    """Return an iterator over all neighbors (both in or out)."""
    for v in self.out_neighbors():
        yield v
    for v in self.in_neighbors():
        yield v

def _in_degree(self, weight=None):
    """Return the in-degree of the vertex. If provided, ``weight`` should be a
    scalar edge :class:`~graph_tool.EdgePropertyMap`, and the in-degree will
    correspond to the sum of the weights of the in-edges.
    """

    if weight is None:
        return self.__in_degree()
    else:
        return self.__weighted_in_degree(_prop("e", weight.get_graph(), weight))

def _out_degree(self, weight=None):
    """Return the out-degree of the vertex. If provided, ``weight`` should be a
    scalar edge :class:`~graph_tool.EdgePropertyMap`, and the out-degree will
    correspond to the sum of the weights of the out-edges.
    """

    if weight is None:
        return self.__out_degree()
    else:
        return self.__weighted_out_degree(_prop("e", weight.get_graph(), weight))

def _vertex_repr(self):
    if not self.is_valid():
        return "<invalid Vertex object at 0x%x>" % (id(self))
    return "<Vertex object with index '%d' at 0x%x>" % (int(self), id(self))

_vertex_doc = """Vertex descriptor.

This class represents a vertex in a :class:`~graph_tool.Graph` instance.

:class:`~graph_tool.Vertex` instances are hashable, and are convertible to
integers, corresponding to its index (see :attr:`~graph_tool.Graph.vertex_index`).
"""

def _v_eq(v1, v2):
    try:
        return int(v1) == int(v2)
    except TypeError:
        return False

def _v_ne(v1, v2):
    try:
        return int(v1) != int(v2)
    except TypeError:
        return True

def _v_lt(v1, v2):
    try:
        return int(v1) < int(v2)
    except TypeError:
        return False

def _v_gt(v1, v2):
    try:
        return int(v1) > int(v2)
    except TypeError:
        return False

def _v_le(v1, v2):
    try:
        return int(v1) <= int(v2)
    except TypeError:
        return False

def _v_ge(v1, v2):
    try:
        return int(v1) >= int(v2)
    except TypeError:
        return False

for Vertex in libcore.get_vlist():
    Vertex.__doc__ = _vertex_doc
    Vertex.out_neighbors = _out_neighbors
    Vertex.out_neighbours = _out_neighbors
    Vertex.in_neighbors = _in_neighbors
    Vertex.in_neighbours = _in_neighbors
    Vertex.all_edges = _all_edges
    Vertex.all_neighbors = _all_neighbors
    Vertex.all_neighbours = _all_neighbors
    Vertex.in_degree = _in_degree
    Vertex.out_degree = _out_degree
    try:
        Vertex.is_valid.__doc__ = "Returns ``True`` if the descriptor corresponds to an existing vertex in the graph, ``False`` otherwise."
    except AttributeError:
        pass
    Vertex.__repr__ = _vertex_repr
    Vertex.__eq__ = _v_eq
    Vertex.__ne__ = _v_ne
    Vertex.__lt__ = _v_lt
    Vertex.__gt__ = _v_gt
    Vertex.__le__ = _v_le
    Vertex.__ge__ = _v_ge

_edge_doc = """Edge descriptor.

This class represents an edge in a :class:`~graph_tool.Graph`.

:class:`~graph_tool.Edge` instances are hashable, iterable and thus are
convertible to a tuple, which contains the source and target vertices.
"""

def _edge_iter(self):
    """Iterate over the source and target"""
    for v in (self.source(), self.target()):
        yield v

def _edge_repr(self):
    if not self.is_valid():
        return "<invalid Edge object at 0x%x>" % (id(self))

    return ("<Edge object with source '%d' and target '%d'" +
            " at 0x%x>") % (int(self.source()), int(self.target()), id(self))

# There are several edge classes... me must cycle through them all to modify
# them.

for Edge in libcore.get_elist():
    Edge.__repr__ = _edge_repr
    Edge.__iter__ = _edge_iter
    Edge.__doc__ = _edge_doc
    try:
        Edge.is_valid.__doc__ = "Returns ``True`` if the descriptor corresponds to an existing edge in the graph, ``False`` otherwise."
        Edge.source.__doc__ = "Returns the source of the edge (a :class:`~graph_tool.Vertex` instance)."
        Edge.target.__doc__ = "Returns the target of the edge (a :class:`~graph_tool.Vertex` instance)."
    except AttributeError:
        pass

# some shenanigans to make it seem there is only a single edge and vertex class
EdgeBase.__doc__ = Edge.__doc__
EdgeBase.source = Edge.source
EdgeBase.target = Edge.target
EdgeBase.is_valid = Edge.is_valid
Edge = EdgeBase
Edge.__name__ = "Edge"

VertexBase.__doc__ = Vertex.__doc__
VertexBase.out_neighbors = Vertex.out_neighbors
VertexBase.in_neighbors = Vertex.in_neighbors
VertexBase.out_edges = Vertex.out_edges
VertexBase.in_edges = Vertex.in_edges
VertexBase.all_edges = Vertex.all_edges
VertexBase.all_neighbors = Vertex.all_neighbors
VertexBase.in_degree = Vertex.in_degree
VertexBase.out_degree = Vertex.out_degree
VertexBase.is_valid = Vertex.is_valid
Vertex = VertexBase
Vertex.__name__ = "Vertex"

_get_null_vertex = libcore.get_null_vertex

# Add convenience function to vector classes
def _get_array_view(self):
    return self.get_array()[:]

def _set_array_view(self, v):
    self.get_array()[:] = v

def _vt_getstate(self):
    a = self.a
    if a is None:
        return list(self)
    else:
        return a

def _vt_setstate(self, state):
    self.resize(len(state))
    if self.a is not None:
        self.a = state
    else:
        for i in range(len(state)):
            self[i] = state[i]

def _vt_copy(self):
    v = type(self)()
    v.resize(len(self))
    _vt_setstate(v, self)
    return v

def _vt_init(self, n=None, init=None):
    self.__base_init__()
    if n is not None:
        self.resize(n)
    if init is not None:
        _vt_setstate(self, init)

vector_types = [Vector_bool, Vector_int16_t, Vector_int32_t, Vector_int64_t,
                Vector_double, Vector_long_double, Vector_size_t, Vector_cdouble]

for vt in vector_types:
    if not hasattr(vt, "__base_init__"):
        vt.__base_init__ = vt.__init__
        vt.__init__ = _vt_init
    vt.a = property(_get_array_view, _set_array_view,
                    doc=r"""Shortcut to the `get_array` method as an attribute.""")
    vt.__repr__ = lambda self: self.a.__repr__()
    vt.copy = _vt_copy
    vt.__copy__ = vt.copy
    vt.__getstate__ = _vt_getstate
    vt.__setstate__ = _vt_setstate

Vector_string.a = None
Vector_string.get_array = lambda self: None
Vector_string.__repr__ = lambda self: repr(list(self))

# Global RNG

def seed_rng(seed):
    """Seed the random number generator used by graph-tool's algorithms. A value
    of ``0`` will cause the system's entropy source to be used as seed."""
    libcore.seed_rng(seed)

seed_rng(0)

def _get_rng():
    return libcore.get_rng()

from . openmp import *

if openmp_enabled() and os.environ.get("OMP_SCHEDULE") is None:
    openmp_set_schedule("static", 0)
