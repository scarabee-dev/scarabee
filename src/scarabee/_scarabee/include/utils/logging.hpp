#ifndef SCARABEE_LOGGING_H
#define SCARABEE_LOGGING_H

#include <spdlog/spdlog.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/details/null_mutex.h>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>
#include <mutex>
#include <string>

namespace scarabee {

using LogLevel = spdlog::level::level_enum;

void set_logging_level(LogLevel level);

void set_output_file(const std::string& fname);

template <typename Mutex>
class PythonSink : public spdlog::sinks::base_sink<Mutex> {
 public:
  PythonSink() = default;

 protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    spdlog::memory_buf_t formatted;
    spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);

    // Make sure we acquire the GIL before printing to Python !
    py::gil_scoped_acquire gil;

    py::print(fmt::to_string(formatted), py::arg("end") = "",
              py::arg("flush") = true);
  }

  void flush_() override {
    // Make sure we acquire the GIL before printing to Python !
    py::gil_scoped_acquire gil;
    py::print("", py::arg("end") = "", py::arg("flush") = true);
  }
};

// Mutli-Threaded Sink
using PythonSinkMT = PythonSink<std::mutex>;

// Single-Threaded Sink
using PythonSinkST = PythonSink<spdlog::details::null_mutex>;

}  // namespace scarabee

#endif
