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

#ifndef SHARED_HEAP_HH
#define SHARED_HEAP_HH

#include <vector>
#include <algorithm>

template <class T, class CMP>
class SharedHeap
{
public:

    SharedHeap(std::vector<T>& heap, size_t max_size, const CMP& cmp)
        : _heap(heap), _max_size(max_size), _cmp(cmp) {}
    ~SharedHeap()
    {
        merge();
    }

    void push(const T& x)
    {
        if (_local_heap.size() < _max_size)
        {
            _local_heap.push_back(x);
            std::push_heap(_local_heap.begin(),
                           _local_heap.end(), _cmp);
        }
        else if (_cmp(x, _local_heap.front()))
        {
            std::pop_heap(_local_heap.begin(),
                          _local_heap.end(), _cmp);
            _local_heap.back() = x;
            std::push_heap(_local_heap.begin(),
                           _local_heap.end(), _cmp);
        }
    }

    void merge()
    {
        #pragma omp critical (shared_heap)
        {
            if (_heap.empty())
            {
                _heap.swap(_local_heap);
            }
            else
            {
                for (auto& x : _local_heap)
                {
                    if (_heap.size() < _max_size)
                    {
                        _heap.push_back(x);
                        std::push_heap(_heap.begin(),
                                       _heap.end(), _cmp);
                    }
                    else if (_cmp(x, _heap.front()))
                    {
                        std::pop_heap(_heap.begin(),
                                      _heap.end(), _cmp);
                        _heap.back() = x;
                        std::push_heap(_heap.begin(),
                                       _heap.end(), _cmp);
                    }
                }
                _local_heap.clear();
            }
        }
    }

private:
    std::vector<T>& _heap;
    size_t _max_size;
    std::vector<T> _local_heap;
    const CMP& _cmp;
};

template <class T, class CMP>
SharedHeap<T, CMP> make_shared_heap(std::vector<T>& heap, size_t max_size,
                                    const CMP& cmp)
{
    return SharedHeap<T, CMP>(heap, max_size, cmp);
}


#endif // SHARED_HEAP_HH
