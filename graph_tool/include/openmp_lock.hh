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

#ifndef OPENMP_lOCK_HH
#define OPENMP_lOCK_HH
#include "config.h"

#ifdef _OPENMP
# include <omp.h>

class openmp_mutex
{
public:
    openmp_mutex()                     { omp_init_lock(&_lock); }
    openmp_mutex(const openmp_mutex& ) { omp_init_lock(&_lock); }
    ~openmp_mutex()                    { omp_destroy_lock(&_lock); }

    openmp_mutex& operator= (const openmp_mutex& ) { return *this; }

    void lock()     { omp_set_lock(&_lock); }
    void unlock()   { omp_unset_lock(&_lock); }
private:
    omp_lock_t _lock;
};

#else

class openmp_mutex
{
public:
   void lock() {}
   void unlock() {}
};
#endif

class openmp_scoped_lock
{
public:
    explicit openmp_scoped_lock(openmp_mutex& m) : _mut(m), _locked(true) { _mut.lock(); }
    ~openmp_scoped_lock() { unlock(); }
    void unlock()  { if(!_locked) return; _locked=false; _mut.unlock(); }
    void lock()    { if(_locked) return; _mut.lock(); _locked=true; }
private:
    openmp_mutex& _mut;
    bool _locked;

    // forbid copying
    void operator=(const openmp_scoped_lock&);
    openmp_scoped_lock(const openmp_scoped_lock&);
 };

#endif // OPENMP_lOCK_HH
