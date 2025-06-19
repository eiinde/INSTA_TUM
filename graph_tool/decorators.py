#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
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
Some useful decorators
"""

import inspect
import functools
import types

################################################################################
# Decorators
# Some useful function decorators which will be used
################################################################################

_wraps = functools.wraps

def _attrs(**kwds):
    """Decorator which adds arbitrary attributes to methods"""
    def decorate(f):
        for k in kwds:
            setattr(f, k, kwds[k])
        return f
    return decorate


def _limit_args(allowed_vals):
    """Decorator which will limit the values of given arguments to a specified
    list of allowed values, and raise TypeError exception if the given value
    doesn't belong. 'allowed_vals' is a dict containing the allowed value list
    for each limited function argument."""
    def decorate(func):
        @_wraps(func)
        def wrap(*args, **kwargs):
            arg_names = inspect.getfullargspec(func)[0]
            arguments = list(zip(arg_names, args))
            arguments += [(k, kwargs[k]) for k in list(kwargs.keys())]
            for a in arguments:
                if a[0] in allowed_vals:
                    if a[1] not in allowed_vals[a[0]]:
                        raise TypeError("value for '%s' must be one of: %s" % \
                                         (a[0], ", ".join(allowed_vals[a[0]])))
            return func(*args, **kwargs)
        return wrap
    return decorate


def _require(arg_name, *allowed_types):
    """Decorator that lets you annotate function definitions with argument type
    requirements. These type requirements are automatically checked by the
    system at function invocation time."""
    def make_wrapper(f):
        if hasattr(f, "wrapped_args"):
            wrapped_args = f.wrapped_args
        else:
            code = f.__code__
            wrapped_args = list(code.co_varnames[:code.co_argcount])

        try:
            arg_index = wrapped_args.index(arg_name)
        except ValueError:
            raise NameError(arg_name)

        @_wraps(f)
        def wrapper(*args, **kwargs):
            if len(args) > arg_index:
                arg = args[arg_index]
                if not isinstance(arg, allowed_types):
                    type_list = " or ".join(str(allowed_type) \
                                            for allowed_type in allowed_types)
                    raise TypeError("Expected '%s' to be %s; was %s." % \
                                    (arg_name, type_list, type(arg)))
            else:
                if arg_name in kwargs:
                    arg = kwargs[arg_name]
                    if not isinstance(arg, allowed_types):
                        type_list = " or ".join(str(allowed_type) \
                                                for allowed_type in \
                                                allowed_types)
                        raise TypeError("Expected '%s' to be %s; was %s." %\
                                        (arg_name, type_list, type(arg)))

            return f(*args, **kwargs)
        wrapper.wrapped_args = wrapped_args
        return wrapper
    return make_wrapper

def _copy_func(f, name=None):
    fn = types.FunctionType(f.__code__, f.__globals__, name or f.__name__,
                            f.__defaults__, f.__closure__)
    fn.__dict__.update(f.__dict__)
    return fn

def _parallel(f):
    text = """.. admonition:: Parallel implementation.

    If enabled during compilation, this algorithm will run in parallel using
    `OpenMP <https://en.wikipedia.org/wiki/OpenMP>`_. See the :ref:`parallel
    algorithms <parallel_algorithms>` section for information about how to
    control several aspects of parallelization.
    """
    lines = f.__doc__.split("\n")
    f.__doc__ = ""
    for line in lines:
        if "@parallel@" in line:
            indent = line.replace("@parallel@", "")
            for l in text.split("\n"):
                f.__doc__ += indent + l + "\n"
        else:
            f.__doc__ += line + "\n"
    return f
