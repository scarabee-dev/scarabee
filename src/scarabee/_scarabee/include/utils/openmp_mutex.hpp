#ifndef SCARABEE_OPENMP_MUTEX_H
#define SCARABEE_OPENMP_MUTEX_H

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * This class was largely taken from OpenMC and then slightly modified. OpenMC
 * is distributed under the MIT license as follows:
 *
 * Copyright (c) 2011-2025 Massachusetts Institute of Technology, UChicago Argonne
 * LLC, and OpenMC contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

namespace scarabee {

  class OpenMPMutex {
    public:
      OpenMPMutex() { init(); }

      ~OpenMPMutex()
      {
#ifdef _OPENMP
        omp_destroy_lock(&mutex_);
#endif
      }

      // omp_lock_t objects cannot be deep copied, they can only be shallow
      // copied. Thus, while shallow copying of an omp_lock_t object is
      // completely valid (provided no race conditions exist), true copying
      // of an OpenMPMutex object is not valid due to the action of the
      // destructor. However, since locks are fungible, we can simply replace
      // copying operations with default construction. This allows storage of
      // OpenMPMutex objects within containers that may need to move/copy them
      // (e.g., std::vector). It is left to the caller to understand that
      // copying of OpenMPMutex does not produce two handles to the same mutex,
      // rather, it produces two different mutexes.

      // Copy constructor
      OpenMPMutex(const OpenMPMutex&) { init(); }

      // Copy assignment operator
      OpenMPMutex& operator=(const OpenMPMutex&) { return *this; }

      //! Lock the mutex.
      //
      //! This function blocks execution until the lock succeeds.
      void lock()
      {
#ifdef _OPENMP
        omp_set_lock(&mutex_);
#endif
      }

      //! Try to lock the mutex and indicate success.
      //
      //! This function does not block.  It returns immediately and gives false if
      //! the lock is unavailable.
      bool try_lock() noexcept
      {
#ifdef _OPENMP
        return omp_test_lock(&mutex_);
#else
        return true;
#endif
      }

      //! Unlock the mutex.
      void unlock() noexcept
      {
#ifdef _OPENMP
        omp_unset_lock(&mutex_);
#endif
      }

    private:
#ifdef _OPENMP
      omp_lock_t mutex_;
#endif

      void init()
      {
#ifdef _OPENMP
        omp_init_lock(&mutex_);
#endif
      }
  };

}

#endif
