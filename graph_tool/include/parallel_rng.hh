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

#ifndef PARALLEL_RNG_HH
#define PARALLEL_RNG_HH

#include "config.h"
#include "random.hh"
#include <vector>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "openmp.hh"

template <class RNG>
class parallel_rng
{
public:
    parallel_rng(RNG& rng):
        _rngs(get_rngs(rng))
    {
        size_t num_threads = get_num_threads();
        for (size_t i = _rngs.size(); i < num_threads - 1; ++i)
        {
            _rngs.emplace_back(rng);
            _rngs.back().set_stream(get_rng_stream());
        }
    }

    static void clear()
    {
        _trngs.clear();
    }

    RNG& get(RNG& rng)
    {
        size_t tid = get_thread_num();
        if (tid == 0)
            return rng;
        return _rngs[tid - 1];
    }

private:

    static std::vector<RNG>& get_rngs(RNG& rng)
    {
        std::lock_guard<std::mutex> lock(_init_mutex);
        return _trngs[&rng];
    }

    std::vector<RNG>& _rngs;
    static inline std::unordered_map<RNG*, std::vector<RNG>> _trngs;
    static inline std::mutex _init_mutex;
};


#endif // PARALLEL_RNG_HH
