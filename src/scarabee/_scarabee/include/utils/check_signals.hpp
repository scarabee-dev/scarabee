#ifndef SCARABEE_CHECK_SIGNALS_H
#define SCARABEE_CHECK_SIGNALS_H

#include <utils/logging.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace scarabee {

inline void check_for_signals() {
  // Unfortunately, we must acquire the GIL before checking for signals
  py::gil_scoped_acquire acquire;

  if (PyErr_CheckSignals() != 0) {
    spdlog::error("Received interupt signal...");
    throw py::error_already_set();
  }
}

}  // namespace scarabee

#endif
