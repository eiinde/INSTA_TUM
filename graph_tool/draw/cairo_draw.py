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

import os
import warnings
import numpy
from collections.abc import Iterable

from .. topology import shortest_distance, is_bipartite
from .. import _check_prop_scalar, perfect_prop_hash

try:
    import cairo
except ImportError:
    msg = "Error importing cairo. Graph drawing will not work."
    warnings.warn(msg, RuntimeWarning)
    raise

default_cm = None
try:
    import matplotlib.artist
    import matplotlib.backends.backend_cairo
    import matplotlib.cm
    import matplotlib.colors
    from matplotlib.cbook import flatten
    default_clrs = list(matplotlib.cm.tab20.colors) + \
        list(matplotlib.cm.tab20b.colors)
    default_cm = matplotlib.cm.colors.ListedColormap(default_clrs)
    default_cm_bin = matplotlib.cm.colors.ListedColormap([[1, 1, 1, 1],
                                                          [0, 0, 0, 1]])
    has_draw_inline = 'inline' in matplotlib.get_backend()
    color_converter = matplotlib.colors.ColorConverter()
except ImportError:
    msg = "Error importing matplotlib module. Graph drawing will not work."
    warnings.warn(msg, RuntimeWarning)
    raise

try:
    import IPython.display
except ImportError:
    pass

import numpy as np
import gzip
import bz2
import zipfile
import copy
import io
from collections import defaultdict
from scipy.stats import rankdata

from .. import Graph, GraphView, PropertyMap, ungroup_vector_property,\
     group_vector_property, _prop, _check_prop_vector, map_property_values

from .. generation import label_parallel_edges, label_self_loops

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_draw")
try:
    from .libgraph_tool_draw import vertex_attrs, edge_attrs, vertex_shape,\
        edge_marker
except ImportError:
    msg = "Error importing cairo-based drawing library. " + \
        "Was graph-tool compiled with cairomm support?"
    warnings.warn(msg, RuntimeWarning)

from .. draw import sfdp_layout, random_layout, _avg_edge_distance, \
    radial_tree_layout, prop_to_size

from .. generation import graph_union
from .. topology import shortest_path

_vdefaults = {
    "shape": "circle",
    "color": (0.6, 0.6, 0.6, 0.8),
    "fill_color": (0.6470588235294118, 0.058823529411764705, 0.08235294117647059, 0.8),
    "size": 5,
    "aspect": 1.,
    "rotation": 0.,
    "anchor": 1,
    "pen_width": 0.8,
    "halo": 0,
    "halo_color": [0., 0., 1., 0.5],
    "halo_size": 1.5,
    "text": "",
    "text_color": [0., 0., 0., 1.],
    "text_position": -1.,
    "text_rotation": 0.,
    "text_offset": [0., 0.],
    "text_out_width": .1,
    "text_out_color": [0., 0., 0., 0.],
    "font_family": "serif",
    "font_slant": cairo.FONT_SLANT_NORMAL,
    "font_weight": cairo.FONT_WEIGHT_NORMAL,
    "font_size": 12.,
    "surface": None,
    "pie_fractions": [0.75, 0.25],
    "pie_colors": [default_cm(x) for x in range(10)]
    }

_edefaults = {
    "color": (0.1796875, 0.203125, 0.2109375, 0.8),
    "pen_width": 1,
    "start_marker": "none",
    "mid_marker": "none",
    "end_marker": "none",
    "marker_size": 4.,
    "mid_marker_pos": .5,
    "control_points": [],
    "gradient": [],
    "dash_style": [],
    "text": "",
    "text_color": (0., 0., 0., 1.),
    "text_distance": 5,
    "text_parallel": True,
    "text_out_width": .1,
    "text_out_color": [0., 0., 0., 0.],
    "font_family": "serif",
    "font_slant": cairo.FONT_SLANT_NORMAL,
    "font_weight": cairo.FONT_WEIGHT_NORMAL,
    "font_size": 12.,
    "sloppy": False,
    "seamless": False
    }

_vtypes = {
    "shape": "int",
    "color": "vector<double>",
    "fill_color": "vector<double>",
    "size": "double",
    "aspect": "double",
    "rotation": "double",
    "anchor": "double",
    "pen_width": "double",
    "halo": "bool",
    "halo_color": "vector<double>",
    "halo_size": "double",
    "text": "string",
    "text_color": "vector<double>",
    "text_position": "double",
    "text_rotation": "double",
    "text_offset": "vector<double>",
    "text_out_width": "double",
    "text_out_color": "vector<double>",
    "font_family": "string",
    "font_slant": "int",
    "font_weight": "int",
    "font_size": "double",
    "surface": "object",
    "pie_fractions": "vector<double>",
    "pie_colors": "vector<double>"
    }

_etypes = {
    "color": "vector<double>",
    "pen_width": "double",
    "start_marker": "int",
    "mid_marker": "int",
    "end_marker": "int",
    "marker_size": "double",
    "mid_marker_pos": "double",
    "control_points": "vector<double>",
    "gradient": "vector<double>",
    "dash_style": "vector<double>",
    "text": "string",
    "text_color": "vector<double>",
    "text_distance": "double",
    "text_parallel": "bool",
    "text_out_width": "double",
    "text_out_color": "vector<double>",
    "font_family": "string",
    "font_slant": "int",
    "font_weight": "int",
    "font_size": "double",
    "sloppy": "bool",
    "seamless": "bool"
    }

for k in list(_vtypes.keys()):
    _vtypes[getattr(vertex_attrs, k)] = _vtypes[k]

for k in list(_etypes.keys()):
    _etypes[getattr(edge_attrs, k)] = _etypes[k]


def shape_from_prop(shape, enum):
    if isinstance(shape, PropertyMap):
        g = shape.get_graph()
        if shape.key_type() == "v":
            prop = g.new_vertex_property("int")
            descs = g.vertices()
        else:
            descs = g.edges()
            prop = g.new_edge_property("int")
        if shape.value_type() == "string":
            def conv(x):
                return int(getattr(enum, x))
            map_property_values(shape, prop, conv)
        else:
            rg = (min(enum.values.keys()),
                  max(enum.values.keys()))
            g.copy_property(shape, prop)
            if prop.fa.min() < rg[0]:
                prop.fa += rg[0]
            prop.fa -= rg[0]
            prop.fa %= rg[1] - rg[0] + 1
            prop.fa += rg[0]
        return prop
    if isinstance(shape, str):
        return int(getattr(enum, shape))
    else:
        return shape

    raise ValueError("Invalid value for attribute %s: %s" %
                     (repr(enum), repr(shape)))

def open_file(name, mode="r"):
    name = os.path.expanduser(name)
    base, ext = os.path.splitext(name)
    if ext == ".gz":
        out = gzip.GzipFile(name, mode)
        name = base
    elif ext == ".bz2":
        out = bz2.BZ2File(name, mode)
        name = base
    elif ext == ".zip":
        out = zipfile.ZipFile(name, mode)
        name = base
    else:
        out = open(name, mode)
    fmt = os.path.splitext(name)[1].replace(".", "")
    return out, fmt

def get_file_fmt(name):
    name = os.path.expanduser(name)
    base, ext = os.path.splitext(name)
    if ext == ".gz":
        name = base
    elif ext == ".bz2":
        name = base
    elif ext == ".zip":
        name = base
    fmt = os.path.splitext(name)[1].replace(".", "")
    return fmt


def surface_from_prop(surface):
    if isinstance(surface, PropertyMap):
        if surface.key_type() == "v":
            prop = surface.get_graph().new_vertex_property("object")
            descs = surface.get_graph().vertices()
        else:
            descs = surface.get_graph().edges()
            prop = surface.get_graph().new_edge_property("object")
        surface_map = {}
        for v in descs:
            if surface.value_type() == "string":
                if surface[v] not in surface_map:
                    sfc = gen_surface(surface[v])
                    surface_map[surface[v]] = sfc
                prop[v] = surface_map[surface[v]]
            elif surface.value_type() == "python::object":
                if isinstance(surface[v], cairo.Surface):
                    prop[v] = surface[v]
                elif surface[v] is not None:
                    raise ValueError("Invalid value type for surface property: " +
                                     str(type(surface[v])))
            else:
                raise ValueError("Invalid value type for surface property: " +
                                 surface.value_type())
        return prop

    if isinstance(surface, str):
        return gen_surface(surface)
    elif isinstance(surface, cairo.Surface) or surface is None:
        return surface

    raise ValueError("Invalid value for attribute surface: " + repr(surface))

def centered_rotation(g, pos, text_pos=True):
    x, y = ungroup_vector_property(pos, [0, 1])
    cm = (x.fa.mean(), y.fa.mean())
    dx = x.fa - cm[0]
    dy = y.fa - cm[1]
    angle = g.new_vertex_property("double")
    angle.fa = numpy.arctan2(dy, dx)
    pi = numpy.pi
    angle.fa += 2 * pi
    angle.fa %= 2 * pi
    if text_pos:
        idx = (angle.a > pi / 2 ) * (angle.a < 3 * pi / 2)
        tpos = g.new_vertex_property("double")
        angle.a[idx] += pi
        tpos.a[idx] = pi
        return angle, tpos
    return angle

def choose_cm(prop, cm):
    is_bin = False
    if prop.value_type() in ["double", "long double"]:
        is_seq = True
    else:
        ph = perfect_prop_hash([prop])[0]
        if ph.fa.max() >= default_cm.N:
            is_seq = True
        else:
            is_seq = False
            if ph.fa.max() == 1:
                is_bin = True

    if cm is not None:
        return cm, is_seq

    if prop.key_type() == "e":
        if is_bin:
            cm = matplotlib.cm.RdGy
            is_seq = True
        elif not is_seq:
            cm = default_cm
        else:
            cm = matplotlib.cm.binary
    else:
        if is_bin:
            cm = default_cm_bin
        elif not is_seq:
            cm = default_cm
        else:
            cm = matplotlib.cm.magma
    return cm, is_seq

def _convert(attr, val, cmap, vnorm, pmap_default=False, g=None, k=None):
    try:
        cmap, alpha = cmap
    except TypeError:
        alpha = None

    if attr == vertex_attrs.shape:
        new_val = shape_from_prop(val, vertex_shape)
        if pmap_default and not isinstance(val, PropertyMap):
            new_val = g.new_vertex_property("int", new_val)
        return new_val
    elif attr == vertex_attrs.surface:
        new_val = surface_from_prop(val)
        if pmap_default and not isinstance(val, PropertyMap):
            new_val = g.new_vertex_property("python::object", new_val)
        return new_val
    elif attr in [edge_attrs.start_marker, edge_attrs.mid_marker,
                  edge_attrs.end_marker]:
        new_val = shape_from_prop(val, edge_marker)
        if pmap_default and not isinstance(val, PropertyMap):
            new_val = g.new_edge_property("int", new_val)
        return new_val
    elif attr in [vertex_attrs.pie_colors]:
        if isinstance(val, PropertyMap):
            if val.value_type() in ["vector<double>", "vector<long double>"]:
                return val
            if val.value_type() in ["vector<int16_t>", "vector<int32_t>",
                                    "vector<int64_t>", "vector<bool>"]:
                g = val.get_graph()
                new_val = g.new_vertex_property("vector<double>")
                rg = [numpy.inf, -numpy.inf]
                for v in g.vertices():
                    for x in val[v]:
                        rg[0] = min(x, rg[0])
                        rg[1] = max(x, rg[1])
                if rg[0] == rg[1]:
                    rg[1] = 1
                if cmap is None:
                    cmap = default_cm
                map_property_values(val, new_val,
                                    lambda y: flatten([cmap((x - rg[0]) / (rg[1] - rg[0]),
                                                            alpha=alpha) for x in y]))
                return new_val
            if val.value_type() == "vector<string>":
                g = val.get_graph()
                new_val = g.new_vertex_property("vector<double>")
                map_property_values(val, new_val,
                                    lambda y: flatten([color_converter.to_rgba(x) for x in y]))
                return new_val
            if val.value_type() == "python::object":
                try:
                    g = val.get_graph()
                    new_val = g.new_vertex_property("vector<double>")
                    def conv(y):
                        try:
                            new_val[v] = [float(x) for x in flatten(y)]
                        except ValueError:
                            new_val[v] = flatten([color_converter.to_rgba(x) for x in y])
                    map_property_values(val, new_val, conv)
                    return new_val
                except ValueError:
                    pass
        else:
            try:
                new_val = [float(x) for x in flatten(val)]
            except ValueError:
                try:
                    new_val = flatten(color_converter.to_rgba(x) for x in val)
                    new_val = list(new_val)
                except ValueError:
                    pass
            if pmap_default:
                val_a = numpy.zeros((g.num_vertices(), len(new_val)))
                for i in range(len(new_val)):
                    val_a[:, i] = new_val[i]
                return g.new_vertex_property("vector<double>", val_a)
            else:
                return new_val
    elif attr in [vertex_attrs.color, vertex_attrs.fill_color,
                  vertex_attrs.text_color, vertex_attrs.text_out_color,
                  vertex_attrs.halo_color, edge_attrs.color,
                  edge_attrs.text_color, edge_attrs.text_out_color]:
        if isinstance(val, list):
            new_val = val
        elif isinstance(val, (tuple, np.ndarray)):
            new_val = list(val)
        elif isinstance(val, str):
            new_val = list(color_converter.to_rgba(val))
        elif isinstance(val, PropertyMap):
            if val.value_type() in ["vector<double>", "vector<long double>"]:
                new_val = val
            elif val.value_type() in ["int16_t", "int32_t", "int64_t", "double",
                                      "long double", "unsigned long",
                                      "unsigned int", "bool"]:
                cmap, is_seq = choose_cm(val, cmap)
                g = val.get_graph()
                if val.value_type() in ["int16_t", "int32_t", "int64_t",
                                        "unsigned long", "unsigned int"]:
                    if not is_seq:
                        nval = val.copy()
                        nval.fa = rankdata(val.fa, method='dense') - 1
                    else:
                        nval = val
                else:
                    nval = val
                try:
                    vrange = [nval.fa.min(), nval.fa.max()]
                except (AttributeError, ValueError):
                    #vertex index
                    vrange = [int(g.vertex(0, use_index=False)),
                              int(g.vertex(g.num_vertices() - 1,
                                           use_index=False))]
                if vnorm is None:
                    if not is_seq:
                        cnorm = lambda x : x % cmap.N
                    else:
                        cnorm = matplotlib.colors.Normalize(vmin=vrange[0],
                                                            vmax=vrange[1])
                else:
                    cnorm = vnorm
                    cnorm(vrange) # for auto-scale, if needed

                g = val.get_graph()
                if val.key_type() == "v":
                    prop = g.new_vertex_property("vector<double>")
                else:
                    prop = g.new_edge_property("vector<double>")
                map_property_values(nval, prop, lambda x: cmap(cnorm(x),
                                                               alpha=alpha))
                new_val = prop
            elif val.value_type() == "string":
                g = val.get_graph()
                if val.key_type() == "v":
                    prop = g.new_vertex_property("vector<double>")
                else:
                    prop = g.new_edge_property("vector<double>")
                map_property_values(val, prop,
                                    lambda x: color_converter.to_rgba(x))
                new_val = prop
            else:
                raise ValueError("Invalid value for attribute %s: %s" %
                                 (repr(attr), repr(val)))
        if pmap_default and not isinstance(val, PropertyMap):
            if attr in [vertex_attrs.color, vertex_attrs.fill_color,
                        vertex_attrs.text_color, vertex_attrs.halo_color]:
                val_a = numpy.zeros((g.num_vertices(),len(new_val)))
                for i in range(len(new_val)):
                    val_a[:, i] = new_val[i]
                return g.new_vertex_property("vector<double>", val_a)
            else:
                val_a = numpy.zeros((g.num_edges(), len(new_val)))
                for i in range(len(new_val)):
                    val_a[:,i] = new_val[i]
                return g.new_edge_property("vector<double>", val_a)
        else:
            return new_val

    if pmap_default and not isinstance(val, PropertyMap):
        if k == "v":
            new_val = g.new_vertex_property(_vtypes[attr], val=val)
        else:
            new_val = g.new_edge_property(_etypes[attr], val=val)
        return new_val

    return val


def _attrs(attrs, d, g, cmap, vnorm):
    nattrs = {}
    defaults = {}
    for k, v in attrs.items():
        try:
            if d == "v":
                attr = getattr(vertex_attrs, k)
            else:
                attr = getattr(edge_attrs, k)
        except AttributeError:
            warnings.warn("Unknown attribute: " + str(k), UserWarning)
            continue
        if isinstance(v, PropertyMap):
            nattrs[int(attr)] = _prop(d, g, _convert(attr, v, cmap, vnorm))
        else:
            defaults[int(attr)] = _convert(attr, v, cmap, vnorm)
    return nattrs, defaults

def _convert_props(props, d, g, cmap, vnorm, pmap_default=False):
    nprops = {}
    for k, v in props.items():
        try:
            if d == "v":
                attr = getattr(vertex_attrs, k)
            else:
                attr = getattr(edge_attrs, k)
            nprops[k] = _convert(attr, v, cmap, vnorm, pmap_default=pmap_default,
                                 g=g, k=d)
        except AttributeError:
            kind = "vertex" if d == k else "edge"
            warnings.warn(f"Unknown {kind} attribute: " + str(k), UserWarning)
            continue
    return nprops


def get_attr(attr, d, attrs, defaults):
    if attr in attrs:
        p = attrs[attr]
    else:
        p = defaults[attr]
    if isinstance(p, PropertyMap):
        return p[d]
    else:
        return p


def position_parallel_edges(g, pos, loop_angle=float("nan"),
                            parallel_distance=1):
    lp = label_parallel_edges(GraphView(g, directed=False))
    ll = label_self_loops(g)
    if isinstance(loop_angle, PropertyMap):
        angle = loop_angle
    else:
        angle = g.new_vertex_property("double", float(loop_angle))

    g = GraphView(g, directed=True)
    if ((len(lp.fa) == 0 or lp.fa.max() == 0) and
        (len(ll.fa) == 0 or ll.fa.max() == 0)):
        return []
    else:
        spline = g.new_edge_property("vector<double>")
        libgraph_tool_draw.put_parallel_splines(g._Graph__graph,
                                                _prop("v", g, pos),
                                                _prop("e", g, lp),
                                                _prop("e", g, spline),
                                                _prop("v", g, angle),
                                                parallel_distance)
        return spline


def parse_props(prefix, args):
    props = {}
    others = {}
    for k, v in list(args.items()):
        if v is None:
            continue
        if k.startswith(prefix + "_"):
            props[k.replace(prefix + "_", "")] = v
        else:
            others[k] = v
    return props, others

def cairo_draw(g, pos, cr, vprops=None, eprops=None, vorder=None, eorder=None,
               nodesfirst=False, vcmap=None, vcnorm=None, ecmap=None,
               ecnorm=None, loop_angle=numpy.nan, parallel_distance=None, res=0,
               max_render_time=-1, **kwargs):
    r"""Draw a graph to a :mod:`cairo` context.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be drawn.
    pos : :class:`~graph_tool.VertexPropertyMap`
        Vector-valued vertex property map containing the x and y coordinates of
        the vertices.
    cr : :class:`~cairo.Context`
        A :class:`~cairo.Context` instance.
    vprops : dict (optional, default: ``None``)
        Dictionary with the vertex properties. Individual properties may also be
        given via the ``vertex_<prop-name>`` parameters, where ``<prop-name>`` is
        the name of the property.
    eprops : dict (optional, default: ``None``)
        Dictionary with the edge properties. Individual properties may also be
        given via the ``edge_<prop-name>`` parameters, where ``<prop-name>`` is
        the name of the property.
    vorder : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        If provided, defines the relative order in which the vertices are drawn.
    eorder : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        If provided, defines the relative order in which the edges are drawn.
    nodesfirst : bool (optional, default: ``False``)
        If ``True``, the vertices are drawn first, otherwise the edges are.
    vcmap : :class:`matplotlib.colors.Colormap` or tuple (optional, default: ``None``)
        Vertex color map. Optionally, this may be a
        (:class:`matplotlib.colors.Colormap`, alpha) tuple.
    vcnorm : :class:`matplotlib.colors.Normalize` (optional, default: ``None``)
        An object which, when called, normalizes vertex property values into the
        :math:`[0.0, 1.0]` interval, before passing to the ``vcmap`` colormap.
    ecmap : :class:`matplotlib.colors.Colormap` or tuple (optional, default: ``None``)
        Edge color map. Optionally, this may be a
        (:class:`matplotlib.colors.Colormap`, alpha) tuple.
    ecnorm : :class:`matplotlib.colors.Normalize` (optional, default: ``None``)
        An object which, when called, normalizes edge property values into the
        :math:`[0.0, 1.0]` interval, before passing to the ``ecmap`` colormap.
    loop_angle : float or :class:`~graph_tool.EdgePropertyMap` (optional, default: ``numpy.nan``)
        Angle used to draw self-loops. If ``nan`` is given, they will be placed
        radially from the center of the layout.
    parallel_distance : float (optional, default: ``None``)
        Distance used between parallel edges. If not provided, it will be
        determined automatically.
    bg_color : str or sequence (optional, default: ``None``)
        Background color. The default is transparent.
    res : float (optional, default: ``0.``):
        If shape sizes fall below this value, simplified drawing is used.
    max_render_time : int (optional, default: ``-1``):
        If nonnegative, this function will return an iterator that will perform
        part of the drawing at each step, so that each iteration takes at most
        ``max_render_time`` milliseconds.
    vertex_* : :class:`~graph_tool.VertexPropertyMap` or arbitrary types (optional, default: ``None``)
        Parameters following the pattern ``vertex_<prop-name>`` specify the
        vertex property with name ``<prop-name>``, as an alternative to the
        ``vprops`` parameter.
    edge_* : :class:`~graph_tool.EdgePropertyMap` or arbitrary types (optional, default: ``None``)
        Parameters following the pattern ``edge_<prop-name>`` specify the edge
        property with name ``<prop-name>``, as an alternative to the ``eprops``
        parameter.

    Returns
    -------
    iterator :
        If ``max_render_time`` is nonnegative, this will be an iterator that will
        perform part of the drawing at each step, so that each iteration takes
        at most ``max_render_time`` milliseconds.

    """

    if vorder is not None:
        _check_prop_scalar(vorder, name="vorder")

    vprops = {} if vprops is None else copy.copy(vprops)
    eprops = {} if eprops is None else copy.copy(eprops)

    props, kwargs = parse_props("vertex", kwargs)
    vprops.update(props)
    props, kwargs = parse_props("edge", kwargs)
    eprops.update(props)
    for k in kwargs:
        warnings.warn("Unknown parameter: " + k, UserWarning)

    cr.save()

    if "control_points" not in eprops:
        if parallel_distance is None:
            parallel_distance = vprops.get("size", _vdefaults["size"])
            if isinstance(parallel_distance, PropertyMap):
                parallel_distance = parallel_distance.fa.mean()
            parallel_distance /= 1.5
            M = cr.get_matrix()
            scale = transform_scale(M, 1,)
            parallel_distance /= scale
        eprops["control_points"] = position_parallel_edges(g, pos, loop_angle,
                                                           parallel_distance)
    if g.is_directed() and "end_marker" not in eprops:
        eprops["end_marker"] = "arrow"

    if vprops.get("text_position", None) == "centered":
        angle, tpos = centered_rotation(g, pos, text_pos=True)
        vprops["text_position"] = tpos
        vprops["text_rotation"] = angle
        toffset = vprops.get("text_offset", None)
        if toffset is not None:
            if not isinstance(toffset, PropertyMap):
                toffset = g.new_vp("vector<double>", val=toffset)
            xo, yo = ungroup_vector_property(toffset, [0, 1])
            xo.a[tpos.a == numpy.pi] *= -1
            toffset = group_vector_property([xo, yo])
            vprops["text_offset"] = toffset

    vattrs, vdefaults = _attrs(vprops, "v", g, vcmap, vcnorm)
    eattrs, edefaults = _attrs(eprops, "e", g, ecmap, ecnorm)
    vdefs = _attrs(_vdefaults, "v", g, vcmap, vcnorm)[1]
    vdefs.update(vdefaults)
    edefs = _attrs(_edefaults, "e", g, ecmap, ecnorm)[1]
    edefs.update(edefaults)

    if "control_points" not in eprops:
        if parallel_distance is None:
            parallel_distance = _defaults
        eprops["control_points"] = position_parallel_edges(g, pos, loop_angle,
                                                           parallel_distance)
    generator = libgraph_tool_draw.cairo_draw(g._Graph__graph,
                                              _prop("v", g, pos),
                                              _prop("v", g, vorder),
                                              _prop("e", g, eorder),
                                              nodesfirst, vattrs, eattrs, vdefs, edefs, res,
                                              max_render_time, cr)
    if max_render_time >= 0:
        def gen():
            for count in generator:
                yield count
            cr.restore()
        return gen()
    else:
        for count in generator:
            pass
        cr.restore()

def color_contrast(color):
    c = np.asarray(color)
    y = c[0] * .299 + c[1] * .587 + c[2] * .114
    if y < .5:
        c[:3] = 1
    else:
        c[:3] = 0
    return c

def auto_colors(g, bg, pos, back):
    if not isinstance(bg, PropertyMap):
        if isinstance(bg, str):
            bg = color_converter.to_rgba(bg)
        bg = g.new_vertex_property("vector<double>", val=bg)
    if not isinstance(pos, PropertyMap):
        if pos == "centered":
            pos = 0
        pos = g.new_vertex_property("double", pos)
    bg_a = bg.get_2d_array(range(4))
    bgc_pos = numpy.zeros((g.num_vertices(), 5))
    for i in range(4):
        bgc_pos[:, i] = bg_a[i, :]
    bgc_pos[:, 4] = pos.fa
    bgc_pos = g.new_vertex_property("vector<double>", bgc_pos)
    def conv(x):
        bgc = x[:4]
        p = x[4]
        if p < 0:
            return color_contrast(bgc)
        else:
            return color_contrast(back)
    c = g.new_vertex_property("vector<double>")
    map_property_values(bgc_pos, c, conv)
    return c



def graph_draw(g, pos=None, vprops=None, eprops=None, vorder=None, eorder=None,
               nodesfirst=False, output_size=(600, 600), fit_view=True,
               fit_view_ink=None, adjust_aspect=True, ink_scale=1,
               inline=has_draw_inline, inline_scale=2, mplfig=None,
               yflip=True, output=None, fmt="auto", bg_color=None,
               antialias=None, **kwargs):
    r"""Draw a graph to screen or to a file using :mod:`cairo`.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be drawn.
    pos : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vector-valued vertex property map containing the x and y coordinates of
        the vertices. If not given, it will be computed using :func:`sfdp_layout`.
    vprops : dict (optional, default: ``None``)
        Dictionary with the vertex properties. Individual properties may also be
        given via the ``vertex_<prop-name>`` parameters, where ``<prop-name>`` is
        the name of the property.
    eprops : dict (optional, default: ``None``)
        Dictionary with the edge properties. Individual properties may also be
        given via the ``edge_<prop-name>`` parameters, where ``<prop-name>`` is
        the name of the property.
    vorder : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        If provided, defines the relative order in which the vertices are drawn.
    eorder : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        If provided, defines the relative order in which the edges are drawn.
    nodesfirst : bool (optional, default: ``False``)
        If ``True``, the vertices are drawn first, otherwise the edges are.
    output_size : tuple of scalars (optional, default: ``(600,600)``)
        Size of the drawing canvas. The units will depend on the output format
        (pixels for the screen, points for PDF, etc).
    fit_view : bool, float or tuple (optional, default: ``True``)
        If ``True``, the layout will be scaled to fit the entire clip region.
        If a float value is given, it will be interpreted as ``True``, and in
        addition the viewport will be scaled out by that factor. If a tuple
        value is given, it should have four values ``(x, y, w, h)`` that
        specify the view in user coordinates.
    fit_view_ink : bool (optional, default: ``None``)
        If ``True``, and ``fit_view == True`` the drawing will be performed once
        to figure out the bounding box, before the actual drawing is
        made. Otherwise, only the vertex positions will be used for this
        purpose. If the value is ``None``, then it will be assumed ``True`` for
        networks of size 10,000 nodes or less, otherwise it will be assumed
        ``False``.
    adjust_aspect : bool (optional, default: ``True``)
        If ``True``, and ``fit_view == True`` the output size will be decreased
        in the width or height to remove empty spaces.
    ink_scale : float (optional, default: ``1.``)
        Scale all sizes and widths by this factor.
    inline : bool (optional, default: ``False``)
        If ``True`` and an `IPython notebook <http://ipython.org/notebook>`_  is
        being used, an inline version of the drawing will be returned.
    inline_scale : float (optional, default: ``2.``):
        Resolution scaling factor for inline images.
    mplfig : :mod:`matplotlib` container object (optional, default: ``None``)
        The ``mplfig`` object needs to have an ``add_artist()`` function. This
        can for example be a :class:`matplotlib.figure.Figure` or
        :class:`matplotlib.axes.Axes`. Only the cairo backend is supported; use
        ``switch_backend('cairo')``. If this option is used, a
        :class:`~graph_tool.draw.GraphArtist` object is returned.
    yflip : bool (optional, default: ``True``)
        If ``mplfig is not None``, and ``fit_view != False``, then the y
        direction of the axis will be flipped, reproducing the same output as
        when ``mplfig is None``.
    output : string or file object (optional, default: ``None``)
        Output file name (or object). If not given, the graph will be displayed via
        :func:`interactive_window`.
    fmt : string (default: ``"auto"``)
        Output file format. Possible values are ``"auto"``, ``"ps"``, ``"pdf"``,
        ``"svg"``, and ``"png"``. If the value is ``"auto"``, the format is
        guessed from the ``output`` parameter.
    bg_color : str or sequence (optional, default: ``None``)
        Background color. The default is transparent.
    antialias : :class:`cairo.Antialias` (optional, default: ``None``)
        If supplied, this will set the antialising mode of the cairo context.
    vertex_* : :class:`~graph_tool.VertexPropertyMap` or arbitrary types (optional, default: ``None``)
        Parameters following the pattern ``vertex_<prop-name>`` specify the
        vertex property with name ``<prop-name>``, as an alternative to the
        ``vprops`` parameter.
    edge_* : :class:`~graph_tool.EdgePropertyMap` or arbitrary types (optional, default: ``None``)
        Parameters following the pattern ``edge_<prop-name>`` specify the edge
        property with name ``<prop-name>``, as an alternative to the ``eprops``
        parameter.
    **kwargs
        Any extra parameters are passed to :func:`~graph_tool.draw.interactive_window`,
        :class:`~graph_tool.draw.GraphWindow`, :class:`~graph_tool.draw.GraphWidget`
        and :func:`~graph_tool.draw.cairo_draw`.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        Vector vertex property map with the x and y coordinates of the vertices.
    selected : :class:`~graph_tool.VertexPropertyMap` (optional, only if ``output is None``)
        Boolean-valued vertex property map marking the vertices which were
        selected interactively.

    Notes
    -----

    .. rubric:: List of vertex properties

    .. table::

        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | Name           | Description                                       | Accepted types         | Default Value                    |
        +================+===================================================+========================+==================================+
        | shape          | The vertex shape. Can be one of the following     | ``str`` or ``int``     | ``"circle"``                     |
        |                | strings: "circle", "triangle", "square",          |                        |                                  |
        |                | "pentagon", "hexagon", "heptagon", "octagon"      |                        |                                  |
        |                | "double_circle", "double_triangle",               |                        |                                  |
        |                | "double_square", "double_pentagon",               |                        |                                  |
        |                | "double_hexagon", "double_heptagon",              |                        |                                  |
        |                | "double_octagon", "pie", "none".                  |                        |                                  |
        |                | Optionally, this might take a numeric value       |                        |                                  |
        |                | corresponding to position in the list above.      |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | color          | Color used to stroke the lines of the vertex.     | ``str`` or list of     | ``[0., 0., 0., 1]``              |
        |                |                                                   | ``floats``             |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | fill_color     | Color used to fill the interior of the vertex.    | ``str`` or list of     | ``[0.640625, 0, 0, 0.9]``        |
        |                |                                                   | ``floats``             |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | size           | The size of the vertex, in the default units of   | ``float`` or ``int``   | ``5``                            |
        |                | the output format (normally either pixels or      |                        |                                  |
        |                | points).                                          |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | aspect         | The aspect ratio of the vertex.                   | ``float`` or ``int``   | ``1.0``                          |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | rotation       | Angle (in radians) to rotate the vertex.          | ``float``              | ``0.``                           |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | anchor         | Specifies how the edges anchor to the vertices.   |  ``int``               | ``1``                            |
        |                | If `0`, the anchor is at the center of the vertex,|                        |                                  |
        |                | otherwise it is at the border.                    |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | pen_width      | Width of the lines used to draw the vertex, in    | ``float`` or ``int``   | ``0.8``                          |
        |                | the default units of the output format (normally  |                        |                                  |
        |                | either pixels or points).                         |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | halo           | Whether to draw a circular halo around the        | ``bool``               | ``False``                        |
        |                | vertex.                                           |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | halo_color     | Color used to draw the halo.                      | ``str`` or list of     | ``[0., 0., 1., 0.5]``            |
        |                |                                                   | ``floats``             |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | halo_size      | Relative size of the halo.                        | ``float``              | ``1.5``                          |
        |                |                                                   |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | text           | Text to draw together with the vertex.            | ``str``                | ``""``                           |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | text_color     | Color used to draw the text. If the value is      | ``str`` or list of     | ``"auto"``                       |
        |                | ``"auto"``, it will be computed based on          | ``floats``             |                                  |
        |                | fill_color to maximize contrast.                  |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | text_position  | Position of the text relative to the vertex.      | ``float`` or ``int``   | ``-1``                           |
        |                | If the passed value is positive, it will          |  or ``"centered"``     |                                  |
        |                | correspond to an angle in radians, which will     |                        |                                  |
        |                | determine where the text will be placed outside   |                        |                                  |
        |                | the vertex. If the value is negative, the text    |                        |                                  |
        |                | will be placed inside the vertex. If the value is |                        |                                  |
        |                | ``-1``, the vertex size will be automatically     |                        |                                  |
        |                | increased to accommodate the text. The special    |                        |                                  |
        |                | value ``"centered"`` positions the texts rotated  |                        |                                  |
        |                | radially around the center of mass.               |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | text_offset    | Text position offset.                             | list of ``float``      | ``[0.0, 0.0]``                   |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | text_rotation  | Angle of rotation (in radians) for the text.      | ``float``              | ``0.0``                          |
        |                | The center of rotation is the position of the     |                        |                                  |
        |                | vertex.                                           |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | text_out_color | Color used to draw the text outline.              | ``str`` or list of     | ``[0,0,0,0]``                    |
        |                |                                                   | ``floats``             |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | text_out_width | Width of the text outline.                        | ``float``              | ``1.``                           |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | font_family    | Font family used to draw the text.                | ``str``                | ``"serif"``                      |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | font_slant     | Font slant used to draw the text.                 | ``cairo.FONT_SLANT_*`` | :data:`cairo.FONT_SLANT_NORMAL`  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | font_weight    | Font weight used to draw the text.                | ``cairo.FONT_WEIGHT_*``| :data:`cairo.FONT_WEIGHT_NORMAL` |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | font_size      | Font size used to draw the text.                  | ``float`` or ``int``   | ``12``                           |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | surface        | The cairo surface used to draw the vertex. If     | :class:`cairo.Surface` | ``None``                         |
        |                | the value passed is a string, it is interpreted   | or ``str``             |                                  |
        |                | as an image file name to be loaded.               |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | pie_fractions  | Fractions of the pie sections for the vertices if | list of ``int`` or     | ``[0.75, 0.25]``                 |
        |                | ``shape=="pie"``.                                 | ``float``              |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | pie_colors     | Colors used in the pie sections if                | list of strings or     | ``('b','g','r','c','m','y','k')``|
        |                | ``shape=="pie"``.                                 | ``float``.             |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+


    .. rubric:: List of edge properties

    .. table::

        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | Name           | Description                                       | Accepted types         | Default Value                    |
        +================+===================================================+========================+==================================+
        | color          | Color used to stroke the edge lines.              | ``str`` or list of     | ``[0.179, 0.203, 0.210, 0.8]``   |
        |                |                                                   | floats                 |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | pen_width      | Width of the line used to draw the edge, in       | ``float`` or ``int``   | ``1.0``                          |
        |                | the default units of the output format (normally  |                        |                                  |
        |                | either pixels or points).                         |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | start_marker,  | Edge markers. Can be one of "none", "arrow",      | ``str`` or ``int``     | ``none``                         |
        | mid_marker,    | "circle", "square", "diamond", or "bar".          |                        |                                  |
        | end_marker     | Optionally, this might take a numeric value       |                        |                                  |
        |                | corresponding to position in the list above.      |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | mid_marker_pos | Relative position of the middle marker.           | ``float``              | ``0.5``                          |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | marker_size    | Size of edge markers, in units appropriate to the | ``float`` or ``int``   | ``4``                            |
        |                | output format (normally either pixels or points). |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | control_points | Control points of a Bézier spline used to draw    | sequence of ``floats`` | ``[]``                           |
        |                | the edge. Each spline segment requires 6 values   |                        |                                  |
        |                | corresponding to the (x,y) coordinates of the two |                        |                                  |
        |                | intermediary control points and the final point.  |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | gradient       | Stop points of a linear gradient used to stroke   | sequence of ``floats`` | ``[]``                           |
        |                | the edge. Each group of 5 elements is interpreted |                        |                                  |
        |                | as ``[o, r, g, b, a]`` where ``o`` is the offset  |                        |                                  |
        |                | in the range [0, 1] and the remaining values      |                        |                                  |
        |                | specify the colors.                               |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | dash_style     | Dash pattern is specified by an array of positive | sequence of ``floats`` | ``[]``                           |
        |                | values. Each value provides the length of         |                        |                                  |
        |                | alternate "on" and "off" portions of the stroke.  |                        |                                  |
        |                | The last value specifies an offset into the       |                        |                                  |
        |                | pattern at which the stroke begins.               |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | text           | Text to draw next to the edges.                   | ``str``                | ``""``                           |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | text_color     | Color used to draw the text.                      | ``str`` or list of     | ``[0., 0., 0., 1.]``             |
        |                |                                                   | ``floats``             |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | text_distance  | Distance from the edge and its text.              | ``float`` or ``int``   | ``4``                            |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | text_parallel  | If ``True`` the text will be drawn parallel to    | ``bool``               | ``True``                         |
        |                | the edges.                                        |                        |                                  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | text_out_color | Color used to draw the text outline.              | ``str`` or list of     | ``[0,0,0,0]``                    |
        |                |                                                   | ``floats``             |                                  |
        +--------------- +---------------------------------------------------+------------------------+----------------------------------+
        | text_out_width | Width of the text outline.                        | ``float``              | ``1.``                           |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | font_family    | Font family used to draw the text.                | ``str``                | ``"serif"``                      |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | font_slant     | Font slant used to draw the text.                 | ``cairo.FONT_SLANT_*`` | :data:`cairo.FONT_SLANT_NORMAL`  |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | font_weight    | Font weight used to draw the text.                | ``cairo.FONT_WEIGHT_*``| :data:`cairo.FONT_WEIGHT_NORMAL` |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+
        | font_size      | Font size used to draw the text.                  | ``float`` or ``int``   | ``12``                           |
        +----------------+---------------------------------------------------+------------------------+----------------------------------+

    Examples
    --------
    .. testcode::
       :hide:

       np.random.seed(42)
       gt.seed_rng(42)
       from numpy import sqrt

    >>> g = gt.price_network(1500)
    >>> deg = g.degree_property_map("in")
    >>> deg.a = 4 * (sqrt(deg.a) * 0.5 + 0.4)
    >>> ebet = gt.betweenness(g)[1]
    >>> ebet.a /= ebet.a.max() / 10.
    >>> eorder = ebet.copy()
    >>> eorder.a *= -1
    >>> pos = gt.sfdp_layout(g)
    >>> control = g.new_edge_property("vector<double>")
    >>> for e in g.edges():
    ...     d = sqrt(sum((pos[e.source()].a - pos[e.target()].a) ** 2)) / 5
    ...     control[e] = [0.3, d, 0.7, d]
    >>> gt.graph_draw(g, pos=pos, vertex_size=deg, vertex_fill_color=deg, vorder=deg,
    ...               edge_color=ebet, eorder=eorder, edge_pen_width=ebet,
    ...               edge_control_points=control, # some curvy edges
    ...               output="graph-draw.pdf")
    <...>

    .. testcleanup::

       conv_png("graph-draw.pdf")

    .. figure:: graph-draw.png
       :align: center
       :width: 80%

       SFDP force-directed layout of a Price network with 1500 nodes. The
       vertex size and color indicate the degree, and the edge color and width
       the edge betweenness centrality.

    """

    vprops = vprops.copy() if vprops is not None else {}
    eprops = eprops.copy() if eprops is not None else {}

    props, kwargs = parse_props("vertex", kwargs)
    props = _convert_props(props, "v", g, kwargs.get("vcmap", None),
                           kwargs.get("vcnorm", None))
    vprops.update(props)
    props, kwargs = parse_props("edge", kwargs)
    props = _convert_props(props, "e", g, kwargs.get("ecmap", None),
                           kwargs.get("ecnorm", None))
    eprops.update(props)

    if pos is None:
        if (g.num_vertices() > 2 and output is None and
            not inline and kwargs.get("update_layout", True) and
            mplfig is None):
            L = np.sqrt(g.num_vertices())
            pos = random_layout(g, [L, L])
            if g.num_vertices() > 1000:
                if "multilevel" not in kwargs:
                    kwargs["multilevel"] = True
            if "layout_K" not in kwargs:
                kwargs["layout_K"] = _avg_edge_distance(g, pos) / 10
        else:
            pos = sfdp_layout(g)
    else:
        pos = g.own_property(pos)
        _check_prop_vector(pos, name="pos", floating=True)
        if output is None and not inline and mplfig is None:
            if "layout_K" not in kwargs:
                kwargs["layout_K"] = _avg_edge_distance(g, pos)
            if "update_layout" not in kwargs:
                kwargs["update_layout"] = False

    if "pen_width" in eprops and "marker_size" not in eprops:
        pw = eprops["pen_width"]
        if isinstance(pw, PropertyMap):
            pw = pw.copy("double")
            pw.fa *= 2.75
            eprops["marker_size"] = pw
        else:
            eprops["marker_size"] = pw * 2.75

    if "text" in eprops and "text_distance" not in eprops and "pen_width" in eprops:
        pw = eprops["pen_width"]
        if isinstance(pw, PropertyMap):
            pw = pw.copy("double")
            pw.fa *= 2
            eprops["text_distance"] = pw
        else:
            eprops["text_distance"] = pw * 2

    if "text" in vprops and ("text_color" not in vprops or vprops["text_color"] == "auto"):
        vcmap = kwargs.get("vcmap", None)
        bg = _convert(vertex_attrs.fill_color,
                      vprops.get("fill_color", _vdefaults["fill_color"]),
                      vcmap, kwargs.get("vcnorm", None))
        vprops["text_color"] = auto_colors(g, bg,
                                           vprops.get("text_position",
                                                      _vdefaults["text_position"]),
                                           bg_color if bg_color is not None else [1., 1., 1., 1.])

    if mplfig is not None:
        ax = None
        if isinstance(mplfig, matplotlib.figure.Figure):
            ax = mplfig.gca()
        elif isinstance(mplfig, matplotlib.axes.Axes):
            ax = mplfig
        else:
            ax = mplfig

        x, y = ungroup_vector_property(pos, [0, 1])
        l, r = x.a.min(), x.a.max()
        b, t = y.a.min(), y.a.max()

        adjust_default_sizes(g, (r - l, t - b), vprops, eprops, min_pen_width=0)

        if ink_scale != 1:
            scale_ink(ink_scale, vprops, eprops, min_pen_width=0)

        artist = GraphArtist(g, pos, ax, vprops, eprops, vorder=vorder,
                             eorder=eorder, nodesfirst=nodesfirst, **kwargs)

        ax.add_artist(artist)

        return artist

    output_file = output
    if inline and output is None:
        if fmt == "auto":
            if output is None:
                fmt = "png"
            else:
                fmt = get_file_fmt(output)
        output = io.BytesIO()

    if output is None:
        if ink_scale != 1:
            scale_ink(ink_scale, vprops, eprops)
        return interactive_window(g, pos, vprops, eprops, vorder, eorder,
                                  nodesfirst, geometry=output_size,
                                  fit_view=fit_view, bg_color=bg_color,
                                  **kwargs)
    else:
        adjust_default_sizes(g, output_size, vprops, eprops)

        if ink_scale != 1:
            scale_ink(ink_scale, vprops, eprops)

        if inline and fmt != "svg":
            output_size = [int(x * inline_scale) for x in output_size]
            scale_ink(inline_scale, vprops, eprops)

        if fit_view != False:
            try:
                x, y, w, h = fit_view
                zoom = min(output_size[0] / w, output_size[1] / h)
            except TypeError:
                pad = fit_view if fit_view is not True else 0.9
                output_size = list(output_size)
                if fit_view_ink is None:
                    fit_view_ink = g.num_vertices() <= 1000
                if fit_view_ink:
                    x, y, zoom = fit_to_view_ink(g, pos, output_size, vprops,
                                                 eprops, adjust_aspect, pad=pad)
                else:
                    x, y, zoom = fit_to_view(get_bb(g, pos), output_size,
                                             adjust_aspect=adjust_aspect,
                                             pad=pad)

        else:
            x, y, zoom = 0, 0, 1


        if isinstance(output, str):
            out, auto_fmt = open_file(output, mode="wb")
        else:
            out = output
            if fmt == "auto":
                raise ValueError("File format must be specified.")

        if fmt == "auto":
            fmt = auto_fmt
        if fmt == "pdf":
            srf = cairo.PDFSurface(out, output_size[0], output_size[1])
        elif fmt == "ps":
            srf = cairo.PSSurface(out, output_size[0], output_size[1])
        elif fmt == "eps":
            srf = cairo.PSSurface(out, output_size[0], output_size[1])
            srf.set_eps(True)
        elif fmt == "svg":
            srf = cairo.SVGSurface(out, output_size[0], output_size[1])
            srf.restrict_to_version(cairo.SVG_VERSION_1_2)
        elif fmt == "png":
            srf = cairo.ImageSurface(cairo.FORMAT_ARGB32, output_size[0],
                                     output_size[1])
        else:
            raise ValueError("Invalid format type: " + fmt)

        cr = cairo.Context(srf)
        if antialias is not None:
            cr.set_antialias(antialias)

        cr.scale(zoom, zoom)
        cr.translate(-x, -y)

        if bg_color is not None:
            if isinstance(bg_color, str):
                bg_color = matplotlib.colors.to_rgba(bg_color)
            cr.set_source_rgba(bg_color[0], bg_color[1],
                               bg_color[2], bg_color[3])
            cr.paint()

        cairo_draw(g, pos, cr, vprops, eprops, vorder, eorder,
                   nodesfirst, **kwargs)

        srf.flush()
        if fmt == "png":
            srf.write_to_png(out)
        elif fmt == "svg":
            srf.finish()

        del cr

        if inline and output_file is None:
            img = None
            if fmt == "png":
                img = IPython.display.Image(data=out.getvalue(),
                                            width=int(output_size[0]/inline_scale),
                                            height=int(output_size[1]/inline_scale))
            elif fmt == "svg":
                img = IPython.display.SVG(data=out.getvalue())
            elif img is None:
                inl_out = io.BytesIO()
                inl_srf = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                             output_size[0],
                                             output_size[1])
                inl_cr = cairo.Context(inl_srf)
                inl_cr.set_source_surface(srf, 0, 0)
                inl_cr.paint()
                inl_srf.write_to_png(inl_out)
                del inl_srf
                img = IPython.display.Image(data=inl_out.getvalue(),
                                            width=int(output_size[0]/inline_scale),
                                            height=int(output_size[1]/inline_scale))
            srf.finish()
            IPython.display.display(img)
        del srf
        return pos

def adjust_default_sizes(g, geometry, vprops, eprops, force=False,
                         min_pen_width=0.05):
    if "size" not in vprops or force:
        A = geometry[0] * geometry[1]
        N = max(g.num_vertices(), 1)
        vprops["size"] = np.sqrt(A / N) / 3.5

    if "pen_width" not in vprops or force:
        size = vprops["size"]
        if isinstance(vprops["size"], PropertyMap):
            size = vprops["size"].fa.mean()
        vprops["pen_width"] = max(size / 10, min_pen_width)
        if "pen_width" not in eprops or force:
            eprops["pen_width"] = max(size / 10, min_pen_width)
        if "marker_size" not in eprops or force:
            eprops["marker_size"] = size * 0.8

    if "font_size" not in vprops or force:
        size = vprops["size"]
        if isinstance(vprops["size"], PropertyMap):
            size = vprops["size"].fa.mean()
        vprops["font_size"] =  size * .6

    if "font_size" not in eprops or force:
        size = vprops["size"]
        if isinstance(vprops["size"], PropertyMap):
            size = vprops["size"].fa.mean()
        eprops["font_size"] =  size * .6


def scale_ink(scale, vprops, eprops, copy=True, min_pen_width=0.05):
    vink_props = ["size", "pen_width", "font_size", "text_out_width"]
    eink_props = ["marker_size", "pen_width", "font_size", "text_distance",
                  "text_out_width"]
    for ink_props, props, defaults in zip([vink_props, eink_props],
                                          [vprops, eprops],
                                          [_vdefaults, _edefaults]):
        for p in ink_props:
            if p not in props:
                props[p] = defaults[p]
            if isinstance(props[p], PropertyMap):
                if copy:
                    props[p] = props[p].copy()
                props[p].fa *= scale
                if p == "pen_width":
                    x = props[p].fa
                    x[x<min_pen_width] = min_pen_width
                    props[p].fa = x
            else:
                if copy:
                    props[p] = props[p] * scale
                else:
                    props[p] *= scale
                if p == "pen_width":
                    props[p] = max(props[p],
                                   min_pen_width)

def get_bb(g, pos):
    pos_x, pos_y = ungroup_vector_property(pos, [0, 1])
    x_range = [pos_x.fa.min(), pos_x.fa.max()]
    y_range = [pos_y.fa.min(), pos_y.fa.max()]
    return x_range[0], y_range[0], x_range[1] - x_range[0], y_range[1] - y_range[0]

def fit_to_view(rec, output_size, adjust_aspect=False, pad=.9):
    x, y, w, h = rec

    if adjust_aspect:
        if h > w:
            output_size[0] = int(round(float(output_size[1] * w / h)))
        else:
            output_size[1] = int(round(float(output_size[0] * h / w)))

    zoom = max(w / output_size[0], h / output_size[1])
    if zoom == 0:
        zoom = 1
    else:
        zoom = 1 / zoom

    x -= (output_size[0] / zoom - w) / 2
    y -= (output_size[1] / zoom - h) / 2

    zoom *= pad

    x -= (1-pad) / 2 * output_size[0] / zoom
    y -= (1-pad) / 2 * output_size[1] / zoom

    return x, y, zoom

def fit_to_view_ink(g, pos, output_size, vprops, eprops, adjust_aspect=False,
                    pad=0.9):
    x, y, zoom = fit_to_view(get_bb(g, pos), output_size, pad=pad)

    srf = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA,
                                 cairo.Rectangle(-output_size[0] * 5,
                                                 -output_size[1] * 5,
                                                 output_size[0] * 10,
                                                 output_size[1] * 10))
    cr = cairo.Context(srf)

    cr.scale(zoom, zoom)
    cr.translate(-x, -y)

    # work around cairo bug with small line widths
    def min_lw(lw):
        if isinstance(lw, PropertyMap):
            lw = lw.copy()
            x = lw.fa
            x[x < 0.05] = 0.1
            lw.fa = x
        else:
            lw = max(lw, 0.1)
        return lw

    vprops = dict(vprops, pen_width=min_lw(vprops.get("pen_width", 0.)))
    eprops = dict(eprops, pen_width=min_lw(eprops.get("pen_width", 0.)))

    cairo_draw(g, pos, cr, vprops, eprops)

    bb = list(srf.ink_extents())

    bb[0], bb[1] = cr.device_to_user(bb[0], bb[1])
    bb[2], bb[3] = cr.device_to_user_distance(bb[2], bb[3])

    x, y, zoom = fit_to_view(bb, output_size,
                             adjust_aspect=adjust_aspect, pad=pad)
    return x, y, zoom


def transform_scale(M, scale):
    p = M.transform_distance(scale / np.sqrt(2),
                             scale / np.sqrt(2))
    return np.sqrt(p[0] ** 2 + p[1] ** 2)

def get_hierarchy_control_points(g, t, tpos, beta=0.8, cts=None, is_tree=True,
                                 max_depth=None):
    r"""Return the Bézier spline control points for the edges in ``g``, given
    the hierarchical structure encoded in graph `t`.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be drawn.
    t : :class:`~graph_tool.Graph`
        Directed graph containing the hierarchy of ``g``. It must be a directed
        tree with a single root. The direction of the edges point from the root
        to the leaves, and the vertices in ``t`` with index in the range
        :math:`[0, N-1]`, with :math:`N` being the number of vertices in ``g``,
        must correspond to the respective vertex in ``g``.
    tpos : :class:`~graph_tool.VertexPropertyMap`
        Vector-valued vertex property map containing the x and y coordinates of
        the vertices in graph ``t``.
    beta : ``float`` (optional, default: ``0.8`` or :class:`~graph_tool.EdgePropertyMap`)
        Edge bundling strength. For ``beta == 0`` the edges are straight lines,
        and for ``beta == 1`` they strictly follow the hierarchy. This can be
        optionally an edge property map, which specified a different bundling
        strength for each edge.
    cts : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge property map of type ``vector<double>`` where the control points
        will be stored.
    is_tree : ``bool`` (optional, default: ``True``)
        If ``True``, ``t`` must be a directed tree, otherwise it can be any
        connected graph.
    max_depth : ``int`` (optional, default: ``None``)
        If supplied, only the first ``max_depth`` bottom levels of the hierarchy
        will be used.


    Returns
    -------

    cts : :class:`~graph_tool.EdgePropertyMap`
        Vector-valued edge property map containing the Bézier spline control
        points for the edges in ``g``.

    Notes
    -----
    This is an implementation of the edge-bundling algorithm described in
    [holten-hierarchical-2006]_.


    Examples
    --------
    .. testsetup:: nested_cts

       gt.seed_rng(42)
       np.random.seed(42)

    .. doctest:: nested_cts

       >>> g = gt.collection.data["netscience"]
       >>> g = gt.GraphView(g, vfilt=gt.label_largest_component(g))
       >>> state = gt.minimize_nested_blockmodel_dl(g)
       >>> t = gt.get_hierarchy_tree(state)[0]
       >>> tpos = pos = gt.radial_tree_layout(t, t.vertex(t.num_vertices() - 1, use_index=False), weighted=True)
       >>> cts = gt.get_hierarchy_control_points(g, t, tpos)
       >>> pos = g.own_property(tpos)
       >>> b = state.levels[0].b
       >>> shape = b.copy()
       >>> shape.a %= 14
       >>> gt.graph_draw(g, pos=pos, vertex_fill_color=b, vertex_shape=shape, edge_control_points=cts,
       ...               edge_color=[0, 0, 0, 0.3], vertex_anchor=0, output="netscience_nested_mdl.pdf")
       <...>

    .. testcleanup:: nested_cts

       conv_png("netscience_nested_mdl.pdf")

    .. figure:: netscience_nested_mdl.png
       :align: center
       :width: 80%

       Block partition of a co-authorship network, which minimizes the description
       length of the network according to the nested (degree-corrected) stochastic blockmodel.



    References
    ----------

    .. [holten-hierarchical-2006] Holten, D. "Hierarchical Edge Bundles:
       Visualization of Adjacency Relations in Hierarchical Data.", IEEE
       Transactions on Visualization and Computer Graphics 12, no. 5, 741–748
       (2006). :doi:`10.1109/TVCG.2006.147`
    """

    if cts is None:
        cts = g.new_edge_property("vector<double>")
    if cts.value_type() != "vector<double>":
        raise ValueError("cts property map must be of type 'vector<double>' not '%s' " % cts.value_type())

    u = GraphView(g, directed=True)
    tu = GraphView(t, directed=True)

    if not isinstance(beta, PropertyMap):
        beta = u.new_edge_property("double", beta)
    else:
        beta = beta.copy("double")

    if max_depth is None:
        max_depth = t.num_vertices()

    tu = GraphView(tu, skip_vfilt=True)
    tpos = tu.own_property(tpos)
    libgraph_tool_draw.get_cts(u._Graph__graph,
                               tu._Graph__graph,
                               _prop("v", tu, tpos),
                               _prop("e", u, beta),
                               _prop("e", u, cts),
                               is_tree, max_depth)
    return cts

#
# The functions and classes below depend on GTK
# =============================================
#

try:
    import gi
    try:
        gi.require_version('Gtk', '3.0')
    except ValueError:
        raise ImportError
    from gi.repository import Gtk, Gdk, GdkPixbuf
    from gi.repository import GObject as gobject
    from .gtk_draw import *
except (ImportError, RuntimeError) as e:
    msg = "Error importing Gtk module: %s; GTK+ drawing will not work." % str(e)
    warnings.warn(msg, RuntimeWarning)

def gen_surface(name):
    fobj, fmt = open_file(name)
    if fmt in ["png", "PNG"]:
        sfc = cairo.ImageSurface.create_from_png(fobj)
        return sfc
    else:
        pixbuf = GdkPixbuf.Pixbuf.new_from_file(name)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, pixbuf.get_width(),
                                     pixbuf.get_height())
        cr = cairo.Context(surface)
        Gdk.cairo_set_source_pixbuf(cr, pixbuf, 0, 0)
        cr.paint()
        return surface
#
# matplotlib
# ==========
#

class GraphArtist(matplotlib.artist.Artist):

    def __init__(self, g, pos, ax, vprops=None, eprops=None, raster=False,
                 **kwargs):
        """:class:`matplotlib.artist.Artist` specialization that draws
        :class:`~graph_tool.Graph` instances in a :mod:`matplotlib` figure.

        .. warning::

            Vector drawing is available only on cairo-based backends. For other
            backends, rasterization will be used.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
            Graph to be drawn.
        pos : :class:`~graph_tool.VertexPropertyMap`
            Vector-valued vertex property map containing the x and y coordinates
            of the vertices.
        ax : :class:`matplotlib.axes.Axes`
            A :class:`matplotlib.axes.Axes` instance where the graph will be drawn..
        vprops : dict (optional, default: ``None``)
            Dictionary with the vertex properties.
        eprops : dict (optional, default: ``None``)
            Dictionary with the edge properties.
        **kwargs : dict (optional, default: ``None``)
            Remaining options to pass to :class:`~graph_tool.draw.cairo_draw`.
        """

        matplotlib.artist.Artist.__init__(self)
        self.g = g
        self.pos = pos
        self.ax = ax
        self.vprops = vprops if vprops is not None else {}
        self.eprops = eprops if eprops is not None else {}
        self.raster = raster
        self.kwargs = kwargs.copy()

    def set_raster(self, raster=True):
        """If ``raster == True``, then the graph will be rasterized even if a
        cairo-based backend is being used."""
        self.raster = raster

    def fit_view(self, scale=1, margin=.1, yflip=False):
        """Set axis limits to fit the graph, optionally scaling it via the
        ``scale`` parameter, and adding a relative margin given by ``margin``.
        If ``yflip is True``, the y axis is flipped upside down.
        """
        x, y = ungroup_vector_property(self.pos, [0, 1])
        l, r = x.a.min(), x.a.max()
        b, t = y.a.min(), y.a.max()
        w = r - l
        h = t - b
        w *= scale
        h *= scale
        self.ax.set_xlim(l - w * margin, r + w * margin)
        if yflip:
            self.ax.set_ylim(t + h * margin, b - h * margin)
        else:
            self.ax.set_ylim(b - h * margin, t + h * margin)

    def draw_to_cairo(self, ctx, transform, mag=1):
        """Draw to a cairo context."""
        ctx.save()

        pos = self.pos.copy()
        eprops = dict(self.eprops)
        vprops = dict(self.vprops)

        # clip region
        l, r = self.ax.get_xlim()
        b, t = self.ax.get_ylim()
        l, b = transform((l, b))
        r, t = transform((r, t))
        ctx.new_path()
        ctx.rectangle(min(l, r), min(b, t), abs(r-l), abs(t-b))
        ctx.clip()

        d = (self.ax.transData.transform((1, 1)) -
             self.ax.transData.transform((0, 0)))
        scale_ink(np.mean(np.abs(d)) * mag, vprops, eprops, min_pen_width=0)

        pos = pos.t(transform)

        # transform edge control points, if present
        cp = eprops.get("control_points", None)
        if isinstance(cp, PropertyMap):
            ctx.save()
            cp = cp.copy()
            for e in self.g.edges():
                s = self.pos[e.source()].a
                t = self.pos[e.target()].a
                a = np.arctan2(t[1] - s[1],
                               t[0] - s[0])
                l = np.sqrt((t[1] - s[1]) ** 2 +
                            (t[0] - s[0]) ** 2)
                c = cp[e]
                for i in range(len(c) // 2):
                    x = c.a[i*2:(i+1)*2]
                    ctx.identity_matrix()
                    ctx.translate(s[0], s[1])
                    ctx.rotate(a)
                    ctx.scale(l, 1)
                    x = ctx.user_to_device(x[0], x[1])
                    x = transform(x)
                    c.a[i*2:(i+1)*2] = x

                s = pos[e.source()].a
                t = pos[e.target()].a
                a = np.arctan2(t[1] - s[1],
                               t[0] - s[0])
                l = np.sqrt((t[1] - s[1]) ** 2 +
                            (t[0] - s[0]) ** 2)

                for i in range(len(c) // 2):
                    x = c.a[i*2:(i+1)*2]
                    ctx.identity_matrix()
                    ctx.scale(1/l, 1)
                    ctx.rotate(-a)
                    ctx.translate(-s[0], -s[1])
                    x = ctx.user_to_device(x[0], x[1])
                    c.a[i*2:(i+1)*2] = x
            ctx.restore()
            eprops["control_points"] = cp

        cairo_draw(self.g, pos, ctx, vprops, eprops, **self.kwargs)

        ctx.restore()

    def draw(self, renderer):
        width, height = renderer.get_canvas_width_height()
        if (isinstance(renderer,
                       matplotlib.backends.backend_cairo.RendererCairo) and
            not self.raster):
            ctx = renderer.gc.ctx

            if not isinstance(ctx, cairo.Context):
                ctx = _UNSAFE_cairocffi_context_to_pycairo(ctx)

            # Cairo coordinates are flipped in the y direction
            def transform(x):
                x = self.ax.transData.transform(x)
                return (x[0], height - x[1])

            self.draw_to_cairo(ctx, transform)
        else:
            l, r = self.ax.get_xlim()
            b, t = self.ax.get_ylim()

            l, b = self.ax.transData.transform((l, b))
            r, t = self.ax.transData.transform((r, t))

            mag = renderer.get_image_magnification()
            width, height = int(abs(r-l) * mag), int(abs(t-b) * mag)

            img = cairo.ImageSurface(cairo.Format.ARGB32, width, height)
            ctx = cairo.Context(img)

            def transform(x):
                x = self.ax.transData.transform(x)
                x = ((x[0] - l) * width / (r - l),
                     (x[1] - b) * height / (t - b))
                return (x[0], height - x[1])

            self.draw_to_cairo(ctx, transform, mag)

            img.flush()

            buf = img.get_data()
            im = np.ndarray(shape=(height, width, 4), dtype=np.uint8,
                            buffer=buf)
            im[:, :, [0, 2]] = im[:, :, [2, 0]]  # fix endianess (swap R and B)
            im = im[::-1, :, :]                  # flip y direction
            gc = renderer.new_gc()
            renderer.draw_image(gc, l, b, im)

#
# Drawing hierarchies
# ===================
#

def draw_hierarchy(state, pos=None, layout="radial", beta=0.8, node_weight=None,
                   vprops=None, eprops=None, hvprops=None, heprops=None,
                   subsample_edges=None, rel_order="degree", deg_size=True,
                   vsize_scale=1, hsize_scale=1, hshortcuts=0, hide=0,
                   bip_aspect=1., empty_branches=False, **kwargs):
    r"""Draw a nested block model state in a circular hierarchy layout with edge
    bundling.

    Parameters
    ----------
    state : :class:`~graph_tool.inference.NestedBlockState`
        Nested block state to be drawn.
    pos : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        If supplied, this specifies a vertex property map with the positions of
        the vertices in the layout.
    layout : ``str`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``"radial"``)
        If ``layout == "radial"`` :func:`~graph_tool.draw.radial_tree_layout`
        will be used. If ``layout == "sfdp"``, the hierarchy tree will be
        positioned using :func:`~graph_tool.draw.sfdp_layout`. If ``layout ==
        "bipartite"`` a bipartite layout will be used. If instead a
        :class:`~graph_tool.VertexPropertyMap` is provided, it must correspond to the
        position of the hierarchy tree.
    beta : ``float`` (optional, default: ``.8``)
        Edge bundling strength.
    node_weight : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        If provided, this specifies a vertex property map with the relative angular
        section that each vertex occupies. This value is ignored if
        ``layout != "radial"``.
    vprops : dict (optional, default: ``None``)
        Dictionary with the vertex properties. Individual properties may also be
        given via the ``vertex_<prop-name>`` parameters, where ``<prop-name>`` is
        the name of the property. See :func:`~graph_tool.draw.graph_draw` for
        details.
    eprops : dict (optional, default: ``None``)
        Dictionary with the edge properties. Individual properties may also be
        given via the ``edge_<prop-name>`` parameters, where ``<prop-name>`` is
        the name of the property. See :func:`~graph_tool.draw.graph_draw` for
        details.
    hvprops : dict (optional, default: ``None``)
        Dictionary with the vertex properties for the *hierarchy tree*.
        Individual properties may also be given via the ``hvertex_<prop-name>``
        parameters, where ``<prop-name>`` is the name of the property. See
        :func:`~graph_tool.draw.graph_draw` for details.
    heprops : dict (optional, default: ``None``)
        Dictionary with the edge properties for the *hierarchy tree*. Individual
        properties may also be given via the ``hedge_<prop-name>`` parameters,
        where ``<prop-name>`` is the name of the property. See
        :func:`~graph_tool.draw.graph_draw` for details.
    subsample_edges : ``int`` or list of :class:`~graph_tool.Edge` instances (optional, default: ``None``)
        If provided, only this number of random edges will be drawn. If the
        value is a list, it should include the edges that are to be drawn.
    rel_order : ``str`` or ``None`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``"degree"``)
        If ``degree``, the vertices will be ordered according to degree inside
        each group, and the relative ordering of the hierarchy branches. If
        instead a :class:`~graph_tool.VertexPropertyMap` is provided, its value will
        be used for the relative ordering.
    deg_size : ``bool`` (optional, default: ``True``)
        If ``True``, the (total) node degrees will be used for the default
        vertex sizes..
    vsize_scale : ``float`` (optional, default: ``1.``)
        Multiplicative factor for the default vertex sizes.
    hsize_scale : ``float`` (optional, default: ``1.``)
        Multiplicative factor for the default sizes of the hierarchy nodes.
    hshortcuts : ``int`` (optional, default: ``0``)
        Include shortcuts to the number of upper layers in the hierarchy
        determined by this parameter.
    hide : ``int`` (optional, default: ``0``)
        Hide upper levels of the hierarchy.
    bip_aspect : ``float`` (optional, default: ``1.``)
        If ``layout == "bipartite"``, this will define the aspect ratio of layout.
    empty_branches : ``bool`` (optional, default: ``False``)
        If ``empty_branches == False``, dangling branches at the upper layers
        will be pruned.
    vertex_* : :class:`~graph_tool.VertexPropertyMap` or arbitrary types (optional, default: ``None``)
        Parameters following the pattern ``vertex_<prop-name>`` specify the
        vertex property with name ``<prop-name>``, as an alternative to the
        ``vprops`` parameter. See :func:`~graph_tool.draw.graph_draw` for
        details.
    edge_* : :class:`~graph_tool.EdgePropertyMap` or arbitrary types (optional, default: ``None``)
        Parameters following the pattern ``edge_<prop-name>`` specify the edge
        property with name ``<prop-name>``, as an alternative to the ``eprops``
        parameter. See :func:`~graph_tool.draw.graph_draw` for details.
    hvertex_* : :class:`~graph_tool.VertexPropertyMap` or arbitrary types (optional, default: ``None``)
        Parameters following the pattern ``hvertex_<prop-name>`` specify the
        vertex property with name ``<prop-name>``, as an alternative to the
        ``hvprops`` parameter. See :func:`~graph_tool.draw.graph_draw` for
        details.
    hedge_* : :class:`~graph_tool.EdgePropertyMap` or arbitrary types (optional, default: ``None``)
        Parameters following the pattern ``hedge_<prop-name>`` specify the edge
        property with name ``<prop-name>``, as an alternative to the ``heprops``
        parameter. See :func:`~graph_tool.draw.graph_draw` for details.
    **kwargs :
        All remaining keyword arguments will be passed to the
        :func:`~graph_tool.draw.graph_draw` function.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        This is a vertex property map with the positions of
        the vertices in the layout.
    t : :class:`~graph_tool.Graph`
        This is a the hierarchy tree used in the layout.
    tpos : :class:`~graph_tool.VertexPropertyMap`
        This is a vertex property map with the positions of
        the hierarchy tree in the layout.

    Examples
    --------
    .. testsetup:: draw_hierarchy

       gt.seed_rng(42)
       np.random.seed(42)

    .. doctest:: draw_hierarchy

       >>> g = gt.collection.data["celegansneural"]
       >>> state = gt.minimize_nested_blockmodel_dl(g)
       >>> gt.draw_hierarchy(state, output="celegansneural_nested_mdl.pdf")
       (...)

    .. testcleanup:: draw_hierarchy

       conv_png("celegansneural_nested_mdl.pdf")

    .. figure:: celegansneural_nested_mdl.png
       :align: center
       :width: 80%

       Hierarchical block partition of the C. elegans neural network, which
       minimizes the description length of the network according to the nested
       (degree-corrected) stochastic blockmodel.


    References
    ----------
    .. [holten-hierarchical-2006] Holten, D. "Hierarchical Edge Bundles:
       Visualization of Adjacency Relations in Hierarchical Data.", IEEE
       Transactions on Visualization and Computer Graphics 12, no. 5, 741–748
       (2006). :doi:`10.1109/TVCG.2006.147`

    """

    g = state.g

    overlap = state.levels[0].overlap
    if overlap:
        ostate = state.levels[0]
        bv, bcin, bcout, bc = ostate.get_overlap_blocks()
        be = ostate.get_edge_blocks()
        orig_state = state
        state = state.copy()
        b = ostate.get_majority_blocks()
        state.levels[0] = BlockState(g, b=b)
    else:
        b = state.levels[0].b

    if subsample_edges is not None:
        emask = g.new_edge_property("bool", False)
        if isinstance(subsample_edges, int):
            eidx = g.edge_index.copy("int").fa.copy()
            numpy.random.shuffle(eidx)
            emask = g.new_edge_property("bool")
            emask.a[eidx[:subsample_edges]] = True
        else:
            for e in subsample_edges:
                emask[e] = True
        g = GraphView(g, efilt=emask)

    t, tb, tvorder = get_hierarchy_tree(state,
                                        empty_branches=empty_branches)

    if layout == "radial":
        if rel_order == "degree":
            rel_order = g.degree_property_map("total")
        vorder = t.own_property(rel_order.copy())
        if pos is not None:
            x, y = ungroup_vector_property(pos, [0, 1])
            x.fa -= x.fa.mean()
            y.fa -= y.fa.mean()
            angle = g.new_vertex_property("double")
            angle.fa = (numpy.arctan2(y.fa, x.fa) + 2 * numpy.pi) % (2 * numpy.pi)
            vorder = t.own_property(angle)
        if node_weight is not None:
            node_weight = t.own_property(node_weight.copy())
            node_weight.a[node_weight.a == 0] = 1
        tpos = radial_tree_layout(t, root=t.vertex(t.num_vertices() - 1,
                                                   use_index=False),
                                  node_weight=node_weight,
                                  rel_order=vorder,
                                  rel_order_leaf=True)
    elif layout == "bipartite":
        tpos = get_bip_hierachy_pos(state, aspect=bip_aspect,
                                    node_weight=node_weight)
        tpos = t.own_property(tpos)
    elif layout == "sfdp":
        if pos is None:
            tpos = sfdp_layout(t)
        else:
            x, y = ungroup_vector_property(pos, [0, 1])
            x.fa -= x.fa.mean()
            y.fa -= y.fa.mean()
            K = numpy.sqrt(x.fa.std() + y.fa.std()) / 10
            tpos = t.new_vertex_property("vector<double>")
            for v in t.vertices():
                if int(v) < g.num_vertices(True):
                    tpos[v] = [x[v], y[v]]
                else:
                    tpos[v] = [0, 0]
            pin = t.new_vertex_property("bool")
            pin.a[:g.num_vertices(True)] = True
            tpos = sfdp_layout(t, K=K, pos=tpos, pin=pin, multilevel=False)
    else:
        tpos = t.own_property(layout)

    hvvisible = t.new_vertex_property("bool", True)
    if hide is None:
        L = len([s for s in state.levels if s.get_nonempty_B() > 0])
        hide = len(state.levels) - L

    if hide > 0:
        root = t.vertex(t.num_vertices(True) - 1)
        dist = shortest_distance(t, source=root)
        hvvisible.fa = dist.fa >= hide

    pos = g.own_property(tpos.copy())

    cts = get_hierarchy_control_points(g, t, tpos, beta,
                                       max_depth=len(state.levels) - hshortcuts)

    vprops_orig = vprops
    eprops_orig = eprops
    hvprops_orig = vprops
    heprops_orig = eprops
    kwargs_orig = kwargs

    vprops = vprops.copy() if vprops is not None else {}
    eprops = eprops.copy() if eprops is not None else {}

    props, kwargs = parse_props("vertex", kwargs)
    vprops.update(props)
    vprops.setdefault("fill_color", b)
    vprops.setdefault("color", b)
    vprops.setdefault("shape", _vdefaults["shape"] if not overlap else "pie")

    output_size = kwargs.get("output_size", (600, 600))
    if kwargs.get("mplfig", None) is not None:
        x, y = ungroup_vector_property(pos, [0, 1])
        w = x.a.max() - x.a.min()
        h = y.a.max() - y.a.min()
        output_size = (w, h)
    s = numpy.mean(output_size) / (4 * numpy.sqrt(g.num_vertices()))
    vprops.setdefault("size", prop_to_size(g.degree_property_map("total"), s/5, s))

    adjust_default_sizes(g, output_size, vprops, eprops)

    if vprops.get("text_position", None) == "centered":
        angle, text_pos = centered_rotation(g, pos, text_pos=True)
        vprops["text_position"] = text_pos
        vprops["text_rotation"] = angle
        toffset = vprops.get("text_offset", None)
        if toffset is not None:
            if not isinstance(toffset, PropertyMap):
                toffset = g.new_vp("vector<double>", val=toffset)
            xo, yo = ungroup_vector_property(toffset, [0, 1])
            xo.a[text_pos.a == numpy.pi] *= -1
            toffset = group_vector_property([xo, yo])
            vprops["text_offset"] = toffset

    self_loops = label_self_loops(g, mark_only=True)
    if self_loops.fa.max() > 0:
        parallel_distance = vprops.get("size", _vdefaults["size"])
        if isinstance(parallel_distance, PropertyMap):
            parallel_distance = parallel_distance.fa.mean()
        cts_p = position_parallel_edges(g, pos, numpy.nan,
                                        parallel_distance)
        gu = GraphView(g, efilt=self_loops)
        for e in gu.edges():
            cts[e] = cts_p[e]


    vprops = _convert_props(vprops, "v", g, kwargs.get("vcmap", None),
                            kwargs.get("vcnorm", None), pmap_default=True)

    props, kwargs = parse_props("edge", kwargs)
    eprops.update(props)
    eprops.setdefault("control_points", cts)
    eprops.setdefault("pen_width", _edefaults["pen_width"])
    eprops.setdefault("color", list(_edefaults["color"][:-1]) + [.6])
    eprops.setdefault("end_marker", "arrow" if g.is_directed() else "none")
    eprops = _convert_props(eprops, "e", g, kwargs.get("ecmap", None),
                            kwargs.get("ecnorm", None), pmap_default=True)

    hvprops = hvprops.copy() if hvprops is not None else {}
    heprops = heprops.copy() if heprops is not None else {}

    props, kwargs = parse_props("hvertex", kwargs)
    hvprops.update(props)

    blue = list(color_converter.to_rgba("#729fcf"))
    blue[-1] = .6
    hvprops.setdefault("fill_color", blue)
    hvprops.setdefault("color", [1, 1, 1, 0])
    hvprops.setdefault("shape", "square")
    hvprops.setdefault("size", s)

    if hvprops.get("text_position", None) == "centered":
        angle, text_pos = centered_rotation(t, tpos, text_pos=True)
        hvprops["text_position"] = text_pos
        hvprops["text_rotation"] = angle
        toffset = hvprops.get("text_offset", None)
        if toffset is not None:
            if not isinstance(toffset, PropertyMap):
                toffset = t.new_vp("vector<double>", val=toffset)
            xo, yo = ungroup_vector_property(toffset, [0, 1])
            xo.a[text_pos.a == numpy.pi] *= -1
            toffset = group_vector_property([xo, yo])
            hvprops["text_offset"] = toffset

    hvprops = _convert_props(hvprops, "v", t, kwargs.get("vcmap", None),
                             kwargs.get("vcnorm", None), pmap_default=True)

    props, kwargs = parse_props("hedge", kwargs)
    heprops.update(props)

    heprops.setdefault("color", blue)
    heprops.setdefault("end_marker", "arrow")
    heprops.setdefault("marker_size", s * .8)
    heprops.setdefault("pen_width", s / 10)

    heprops = _convert_props(heprops, "e", t, kwargs.get("ecmap", None),
                             kwargs.get("ecnorm", None), pmap_default=True)

    vcmap = kwargs.get("vcmap", None)
    ecmap = kwargs.get("ecmap", vcmap)

    B = state.levels[0].get_B()

    if overlap and "pie_fractions" not in vprops:
        vprops["pie_fractions"] = bc.copy("vector<double>")
        if "pie_colors" not in vprops:
            vertex_pie_colors = g.new_vertex_property("vector<double>")
            nodes = defaultdict(list)
            def conv(k):
                clrs = [vcmap(r / (B - 1) if B > 1 else 0) for r in k]
                return [item for l in clrs for item in l]
            map_property_values(bv, vertex_pie_colors, conv)
            vprops["pie_colors"] = vertex_pie_colors

    gradient = eprops.get("gradient", None)
    if gradient is None:
        gradient = g.new_edge_property("double")
        gradient = group_vector_property([gradient])
        ecolor = eprops.get("ecolor", _edefaults["color"])
        eprops["gradient"] = gradient
        if overlap:
            for e in g.edges():                       # ******** SLOW *******
                r, s = be[e]
                if not g.is_directed() and e.source() > e.target():
                    r, s = s, r
                gradient[e] = [0] + list(vcmap(r / (B - 1))) + \
                              [1] + list(vcmap(s / (B - 1)))
                if isinstance(ecolor, PropertyMap):
                    gradient[e][4] = gradient[e][9] = ecolor[e][3]
                else:
                    gradient[e][4] = gradient[e][9] = ecolor[3]


    t_orig = t
    t = GraphView(t,
                  vfilt=lambda v: int(v) >= g.num_vertices(True) and hvvisible[v])

    t_vprops = {}
    t_eprops = {}

    props = []
    for k in set(list(vprops.keys()) + list(hvprops.keys())):
        t_vprops[k] = (vprops.get(k, None), hvprops.get(k, None))
        props.append(t_vprops[k])
    for k in set(list(eprops.keys()) + list(heprops.keys())):
        t_eprops[k] = (eprops.get(k, None), heprops.get(k, None))
        props.append(t_eprops[k])

    props.append((pos, tpos))
    props.append((g.vertex_index, tb))
    props.append((b, None))
    if "eorder" in kwargs:
        eorder = kwargs["eorder"]
        props.append((eorder,
                      t.new_ep(eorder.value_type(),
                               eorder.fa.max() + 1)))

    u, props = graph_union(GraphView(g, directed=True), t, props=props)

    for k in set(list(vprops.keys()) + list(hvprops.keys())):
        t_vprops[k] = props.pop(0)
    for k in set(list(eprops.keys()) + list(heprops.keys())):
        t_eprops[k] = props.pop(0)
    pos = props.pop(0)
    tb = props.pop(0)
    b = props.pop(0)
    if "eorder" in kwargs:
        eorder = props.pop(0)

    def update_cts(widget, gg, picked, pos, vprops, eprops):
        vmask = gg.vertex_index.copy("int")
        u = GraphView(gg, directed=False, vfilt=vmask.fa < g.num_vertices(True))
        cts = eprops["control_points"]
        get_hierarchy_control_points(u, t_orig, pos, beta, cts=cts,
                                     max_depth=len(state.levels) - hshortcuts)

    def draw_branch(widget, gg, key_id, picked, pos, vprops, eprops):
        if key_id == ord('b'):
            if picked is not None and not isinstance(picked, PropertyMap) and int(picked) > g.num_vertices(True):
                p = shortest_path(t_orig, source=t_orig.vertex(t_orig.num_vertices(True) - 1),
                                  target=picked)[0]
                l = len(state.levels) - max(len(p), 1)

                bstack = state.get_bstack()
                bs = [s.vp["b"].a for s in bstack[:l+1]]
                bs[-1][:] = 0

                if not overlap:
                    b = state.project_level(l).b
                    u = GraphView(g, vfilt=b.a == tb[picked])
                    u.vp["b"] = state.levels[0].b
                    u = Graph(u, prune=True)
                    b = u.vp["b"]
                    bs[0] = b.a
                else:
                    be = orig_state.project_level(l).get_edge_blocks()
                    emask = g.new_edge_property("bool")
                    for e in g.edges():
                        rs = be[e]
                        if rs[0] == tb[picked] and rs[1] == tb[picked]:
                            emask[e] = True
                    u = GraphView(g, efilt=emask)
                    d = u.degree_property_map("total")
                    u = GraphView(u, vfilt=d.fa > 0)
                    u.ep["be"] = orig_state.levels[0].get_edge_blocks()
                    u = Graph(u, prune=True)
                    be = u.ep["be"]
                    s = OverlapBlockState(u, b=be)
                    bs[0] = s.b.a.copy()

                nstate = NestedBlockState(u, bs=bs,
                                          base_type=type(state.levels[0]),
                                          state_args=state.state_args)

                kwargs_ = kwargs_orig.copy()
                if "no_main" in kwargs_:
                    del kwargs_["no_main"]
                draw_hierarchy(nstate, beta=beta, vprops=vprops_orig,
                               eprops=eprops_orig, hvprops=hvprops_orig,
                               heprops=heprops_orig,
                               subsample_edges=subsample_edges,
                               deg_order=deg_order, empty_branches=False,
                               no_main=True, **kwargs_)

        if key_id == ord('r'):
            if layout == "radial":
                x, y = ungroup_vector_property(pos, [0, 1])
                x.fa -= x.fa.mean()
                y.fa -= y.fa.mean()
                angle = t_orig.new_vertex_property("double")
                angle.fa = (numpy.arctan2(y.fa, x.fa) + 2 * numpy.pi) % (2 * numpy.pi)
                tpos = radial_tree_layout(t_orig,
                                          root=t_orig.vertex(t_orig.num_vertices(True) - 1),
                                          rel_order=angle)
                gg.copy_property(gg.own_property(tpos), pos)

            update_cts(widget, gg, picked, pos, vprops, eprops)

            if widget.vertex_matrix is not None:
                widget.vertex_matrix.update()
            widget.picked = None
            widget.selected.fa = False

            widget.fit_to_window()
            widget.regenerate_surface(reset=True)
            widget.queue_draw()

    if ("output" not in kwargs and not kwargs.get("inline", has_draw_inline) and
        kwargs.get("mplfig", None) is None):
        kwargs["layout_callback"] = update_cts
        kwargs["key_press_callback"] = draw_branch

    if "eorder" in kwargs:
        kwargs["eorder"] = eorder

    vorder = kwargs.pop("vorder", None)
    if vorder is None:
        vorder = g.degree_property_map("total")
    tvorder = u.own_property(tvorder)
    tvorder.fa[:g.num_vertices()] = vorder.fa

    for k, v in kwargs.items():
        if isinstance(v, PropertyMap) and v.get_graph().base is not u.base:
            kwargs[k] = u.own_property(v.copy())

    ret = graph_draw(u, pos, vprops=t_vprops, eprops=t_eprops, vorder=tvorder,
                     **kwargs)
    if isinstance(ret, PropertyMap):
        ret = g.own_property(ret)
    return ret, t_orig, tpos


def get_bip_hierachy_pos(state, aspect=1., node_weight=None):

    if state.levels[0].overlap:
        g = state.g
        ostate = state.levels[0]
        bv, bcin, bcout, bc = ostate.get_overlap_blocks()
        be = ostate.get_edge_blocks()

        n_r = zeros(ostate.get_B())
        b = g.new_vertex_property("int")
        for v in g.vertices():
            i = bc[v].a.argmax()
            b[v] = bv[v][i]
            n_r[b[v]] += 1

        orphans = [r for r in range(ostate.get_B()) if n_r[r] == 0]

        for v in g.vertices():
            for r in orphans:
                b[v] = r

        orig_state = state
        state = state.copy()
        state.levels[0] = BlockState(g, b=b)

    g = state.g

    deg = g.degree_property_map("total")

    t, tb, order = get_hierarchy_tree(state)

    root = t.vertex(t.num_vertices(True) - 1)
    if root.out_degree() > 2:
        clabel = is_bipartite(g, partition=True)[1].copy("int")
        if state.levels[0].overlap:
            ostate = OverlapBlockState(g, b=clabel)
            ostate = orig_state.copy(clabel=clabel)
            bc = ostate.propagate_clabel(len(state.levels) - 2)
        else:
            state = state.copy(clabel=clabel)
            bc = state.propagate_clabel(len(state.levels) - 2)

        ps = list(root.out_neighbors())
        t.clear_vertex(root)

        p1 = t.add_vertex()
        p2 = t.add_vertex()

        t.add_edge(root, p1)
        t.add_edge(root, p2)
        for p in ps:
            if bc.a[tb[p]] == 0:
                t.add_edge(p2, p)
            else:
                t.add_edge(p1, p)

    w = t.new_vertex_property("double")
    for v in t.vertices():
        if v.in_degree() == 0:
            break
        if v.out_degree() == 0:
            w[v] = 1 if node_weight is None else node_weight[v]
        parent, = v.in_neighbors()
        w[parent] += w[v]

    pos = t.new_vertex_property("vector<double>")

    pos[root] = (0., 0.)

    p1, p2 = root.out_neighbors()

    if ((w[p1] == w[p2] and p1.out_degree() > p2.out_degree()) or
        w[p1] > w[p2]):
        p1, p2 = p2, p1

    L = len(state.levels)
    pos[p1] = (-1 / L * .5 * aspect, 0)
    pos[p2] = (+1 / L * .5 * aspect, 0)

    for i, p in enumerate([p1, p2]):
        roots = [p]
        while len(roots) > 0:
            nroots = []
            for r in roots:
                cw = pos[r][1] - w[r] / (2. * w[p])
                for v in sorted(r.out_neighbors(), key=lambda a: order[a]):
                    pos[v] = (0, 0)
                    if i == 0:
                        pos[v][0] = pos[r][0] - 1 / L * .5 * aspect
                    else:
                        pos[v][0] = pos[r][0] + 1 / L * .5 * aspect
                    pos[v][1] = cw + w[v] / (2. * w[p])
                    cw += w[v] / w[p]
                    nroots.append(v)
            roots = nroots
    return pos


# Handle cairo contexts from cairocffi

try:
    import cairocffi
    import ctypes
    pycairo_aux = ctypes.PyDLL(os.path.dirname(os.path.abspath(__file__)) + "/libgt_pycairo_aux.so")
    pycairo_aux.gt_PycairoContext_FromContext.restype = ctypes.c_void_p
    pycairo_aux.gt_PycairoContext_FromContext.argtypes = 3 * [ctypes.c_void_p]
    ctypes.pythonapi.PyList_Append.argtypes = 2 * [ctypes.c_void_p]
except ImportError:
    pass

def _UNSAFE_cairocffi_context_to_pycairo(cairocffi_context):
    # Sanity check. Continuing with another type would probably segfault.
    if not isinstance(cairocffi_context, cairocffi.Context):
        raise TypeError('Expected a cairocffi.Context, got %r'
                        % cairocffi_context)

    # Create a reference for PycairoContext_FromContext to take ownership of.
    cairocffi.cairo.cairo_reference(cairocffi_context._pointer)
    # Casting the pointer to uintptr_t (the integer type as wide as a pointer)
    # gets the context’s integer address.
    # On CPython id(cairo.Context) gives the address to the Context type,
    # as expected by PycairoContext_FromContext.
    address = pycairo_aux.gt_PycairoContext_FromContext(
        int(cairocffi.ffi.cast('uintptr_t', cairocffi_context._pointer)),
        id(cairo.Context),
        None)
    assert address
    # This trick uses Python’s C API
    # to get a reference to a Python object from its address.
    temp_list = []
    assert ctypes.pythonapi.PyList_Append(id(temp_list), address) == 0
    return temp_list[0]

# Bottom imports to avoid circular dependency issues
from .. inference import get_hierarchy_tree, NestedBlockState, BlockState, \
    OverlapBlockState
