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

#ifndef GIL_RELEASE_HH
#define GIL_RELEASE_HH

#include <boost/python.hpp>
#include "openmp.hh"

namespace graph_tool
{

class GILRelease
{
public:
    GILRelease(bool release = true)
    {
        size_t tid = get_thread_num();
        if (release && tid == 0)
            _state = PyEval_SaveThread();
    }

    void restore()
    {
        if (_state != nullptr)
        {
            PyEval_RestoreThread(_state);
            _state = nullptr;
        }
    }

    ~GILRelease()
    {
        restore();
    }

private:
    PyThreadState *_state = nullptr;
};

} //graph_tool namespace

#endif
