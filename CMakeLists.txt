cmake_minimum_required(VERSION 3.29)
project(${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX)

option(SCARABEE_USE_OMP "Compile Scarabée with OpenMP for shared memory parallelism" ON)

# Get FetchContent for downloading dependencies
include(FetchContent)

#===============================================================================
# Get Eigen
message(STATUS "Downloading Eigen3 v5.0.1")
FetchContent_Declare(eigen
  URL      https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
  URL_HASH MD5=294f188b9cd8ff95650ebce53b8d3f1d
  EXCLUDE_FROM_ALL
)
set(EIGEN_BUILD_TESTING OFF)
FetchContent_MakeAvailable(eigen)

#===============================================================================
# Get SPDLOG
message(STATUS "Downloading spdlog v1.16.0")
FetchContent_Declare(spdlog
  URL      https://github.com/gabime/spdlog/archive/refs/tags/v1.16.0.tar.gz
  URL_HASH MD5=7e963629c9075b628e765896e057521f
  EXCLUDE_FROM_ALL
)
set(SPDLOG_BUILD_PIC ON CACHE BOOL "Build position independent code (-fPIC)")
FetchContent_MakeAvailable(spdlog)

#===============================================================================
# Get HTL
message(STATUS "Downloading HTL")
FetchContent_Declare(htl
  URL      https://github.com/HunterBelanger/htl/archive/8d89b5ed90f499669a720a55b46f529495bcac0a.tar.gz # Master on Mar. 11 2025
  URL_HASH MD5=1e8330327f6fe11cabd5fab30933283d
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(htl)

#===============================================================================
# Get XTL
message(STATUS "Downloading xtl v0.8.1")
FetchContent_Declare(xtl
  URL      https://github.com/xtensor-stack/xtl/archive/refs/tags/0.8.1.tar.gz
  URL_HASH MD5=fe52514ffeb7e54a031b9d4f57090d48
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(xtl)

#===============================================================================
# Get XSIMD
message(STATUS "Downloading xsimd v14.0.0")
FetchContent_Declare(xsimd
  URL      https://github.com/xtensor-stack/xsimd/archive/refs/tags/14.0.0.tar.gz
  URL_HASH MD5=75c0d34cf7011924ba19978076c76dc1
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(xsimd)

#===============================================================================
# Get XTENSOR
message(STATUS "Downloading xtensor v0.27.1")
FetchContent_Declare(xtensor
  URL https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.27.1.tar.gz
  URL_HASH MD5=ef143422b31b94dd0f5b95b69388cd48
  EXCLUDE_FROM_ALL
)
set(XTENSOR_USE_XSIMD ON CACHE BOOL "Enable SIMD acceleration.")
FetchContent_MakeAvailable(xtensor)

#===============================================================================
# Find Pybind11 from Scikit-Build-Core
set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
  set(PYBIND11_FINDPYTHON ON)
  message(STATUS "Downloading pybind11 v3.0.1")
  FetchContent_Declare(pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.tar.gz
    URL_HASH MD5=81399a5277559163b3ee912b41de1b76
  )
  FetchContent_MakeAvailable(pybind11)
  if (NOT pybind11_VERSION)
    set(pybind11_VERSION 3.0.1)
  endif()
endif()

#===============================================================================
# Get XTENSOR-PYTHON
message(STATUS "Downloading xtensor-python v0.29.0 with Hunter's NumPy fix")
FetchContent_Declare(xtensor-python
  URL      https://github.com/HunterBelanger/xtensor-python/archive/b5bf5773771c70a331a7a5f0ca2e77e16243a6c3.tar.gz
  URL_HASH MD5=7aeb5606b8373ff0cede3a6c872c434a
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(xtensor-python)

#===============================================================================
# Get HDF5
message(STATUS "Downloading HDF5 v2.0.0")
set(HDF5_EXTERNALLY_CONFIGURED  1)
set(HDF5_ENABLE_SZIP_SUPPORT  OFF)
set(HDF5_ENABLE_ZLIB_SUPPORT OFF)
set(HDF5_BUILD_EXAMPLES       OFF)
set(HDF5_BUILD_TOOLS          OFF)
set(HDF5_BUILD_UTILS          OFF)
set(HDF5_BUILD_HL_LIB         OFF)
set(BUILD_SHARED_LIBS         OFF)
FetchContent_Declare(hdf5
  URL      https://github.com/HDFGroup/hdf5/releases/download/2.0.0/hdf5-2.0.0.tar.gz
  URL_HASH MD5=801597eccdf74d7304444fdc18652b75
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(hdf5)

#===============================================================================
# Get HighFive
message(STATUS "Downloading HighFive")
set(HIGHFIVE_UNIT_TESTS OFF)
set(HIGHFIVE_TEST_SPAN OFF)
set(HIGHFIVE_EXAMPLES OFF)
set(HIGHFIVE_BUILD_DOCS OFF)
set(HIGHFIVE_FIND_HDF5 OFF)
FetchContent_Declare(HighFive
  URL      https://github.com/highfive-devs/highfive/archive/6ea1c7b8b38db5a58acf33f011f442e0c28f0aa7.tar.gz # Dec. 16 2025
  URL_HASH MD5=6fdc23148460812548f67b07019b209c
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(HighFive)

#===============================================================================
# Get ImApp
message(STATUS "Downloading ImApp")
FetchContent_Declare(ImApp
  URL      https://github.com/HunterBelanger/ImApp/archive/7bcee00689165944ca608266ce31ee9ea488ef3f.tar.gz
  URL_HASH MD5=b9ef619c5550bfb8435ecd7328cfdb21
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(ImApp)

#===============================================================================
# Get Cereal
message(STATUS "Downloading Cereal")
set(BUILD_DOC OFF)
set(BUILD_SANDBOX OFF)
set(SKIP_PERFORMANCE_COMPARISON ON)
FetchContent_Declare(cereal
  URL      https://github.com/USCiLab/cereal/archive/a56bad8bbb770ee266e930c95d37fff2a5be7fea.tar.gz
  URL_HASH MD5=5ac5bf09b9298f8d7210af301dc3cea9
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(cereal)

#===============================================================================
# Make python library
pybind11_add_module(_scarabee src/scarabee/_scarabee/constants.cpp
                              src/scarabee/_scarabee/nuclide_names.cpp
                              src/scarabee/_scarabee/water.cpp
                              src/scarabee/_scarabee/logging.cpp
                              src/scarabee/_scarabee/gauss_legendre.cpp
                              src/scarabee/_scarabee/gauss_kronrod.cpp
                              src/scarabee/_scarabee/chebyshev.cpp
                              src/scarabee/_scarabee/math.cpp
                              src/scarabee/_scarabee/homogenization.cpp
                              src/scarabee/_scarabee/cross_section.cpp
                              src/scarabee/_scarabee/diffusion_cross_section.cpp
                              src/scarabee/_scarabee/material.cpp
                              src/scarabee/_scarabee/nd_library.cpp
                              src/scarabee/_scarabee/flux_calculator.cpp
                              src/scarabee/_scarabee/cylindrical_cell.cpp
                              src/scarabee/_scarabee/cylindrical_flux_solver.cpp
                              src/scarabee/_scarabee/surface.cpp
                              src/scarabee/_scarabee/flat_source_region.cpp
                              src/scarabee/_scarabee/cell.cpp
                              src/scarabee/_scarabee/empty_cell.cpp
                              src/scarabee/_scarabee/simple_pin_cell.cpp
                              src/scarabee/_scarabee/pin_cell.cpp
                              src/scarabee/_scarabee/simple_bwr_corner_pin_cell.cpp
                              src/scarabee/_scarabee/bwr_corner_pin_cell.cpp
                              src/scarabee/_scarabee/cartesian_2d.cpp
                              src/scarabee/_scarabee/cmfd.cpp
                              src/scarabee/_scarabee/track.cpp
                              src/scarabee/_scarabee/legendre.cpp
                              src/scarabee/_scarabee/yamamoto_tabuchi.cpp
                              src/scarabee/_scarabee/moc_driver.cpp
                              src/scarabee/_scarabee/moc_plotter.cpp
                              src/scarabee/_scarabee/criticality_spectrum.cpp
                              src/scarabee/_scarabee/diffusion_data.cpp
                              src/scarabee/_scarabee/diffusion_geometry.cpp
                              src/scarabee/_scarabee/fd_diffusion_driver.cpp
                              src/scarabee/_scarabee/nem_diffusion_driver.cpp
                              src/scarabee/_scarabee/reflector_sn.cpp
                              src/scarabee/_scarabee/spherical_harmonics.cpp
                              src/scarabee/_scarabee/depletion_chain.cpp
                              src/scarabee/_scarabee/depletion_matrix.cpp
                              #=================================================
                              src/scarabee/_scarabee/python/scarabee.cpp
                              src/scarabee/_scarabee/python/nuclide_names.cpp
                              src/scarabee/_scarabee/python/vector.cpp
                              src/scarabee/_scarabee/python/direction.cpp
                              src/scarabee/_scarabee/python/logging.cpp
                              src/scarabee/_scarabee/python/xs1d.cpp
                              src/scarabee/_scarabee/python/xs2d.cpp
                              src/scarabee/_scarabee/python/cross_section.cpp
                              src/scarabee/_scarabee/python/diffusion_cross_section.cpp
                              src/scarabee/_scarabee/python/micro_cross_sections.cpp
                              src/scarabee/_scarabee/python/material.cpp
                              src/scarabee/_scarabee/python/nd_library.cpp
                              src/scarabee/_scarabee/python/flux_calculator.cpp
                              src/scarabee/_scarabee/python/cylindrical_cell.cpp
                              src/scarabee/_scarabee/python/cylindrical_flux_solver.cpp
                              src/scarabee/_scarabee/python/polar_quadrature.cpp
                              src/scarabee/_scarabee/python/boundary_condition.cpp
                              src/scarabee/_scarabee/python/simulation_mode.cpp
                              src/scarabee/_scarabee/python/track.cpp 
                              src/scarabee/_scarabee/python/cell.cpp
                              src/scarabee/_scarabee/python/empty_cell.cpp
                              src/scarabee/_scarabee/python/pin_cell_type.cpp
                              src/scarabee/_scarabee/python/simple_pin_cell.cpp
                              src/scarabee/_scarabee/python/pin_cell.cpp
                              src/scarabee/_scarabee/python/simple_bwr_corner_pin_cell.cpp
                              src/scarabee/_scarabee/python/bwr_corner_pin_cell.cpp
                              src/scarabee/_scarabee/python/cartesian_2d.cpp
                              src/scarabee/_scarabee/python/cmfd.cpp
                              src/scarabee/_scarabee/python/moc_driver.cpp
                              src/scarabee/_scarabee/python/criticality_spectrum.cpp
                              src/scarabee/_scarabee/python/diffusion_data.cpp
                              src/scarabee/_scarabee/python/diffusion_geometry.cpp
                              src/scarabee/_scarabee/python/fd_diffusion_driver.cpp
                              src/scarabee/_scarabee/python/nem_diffusion_driver.cpp
                              src/scarabee/_scarabee/python/reflector_sn.cpp
                              src/scarabee/_scarabee/python/water.cpp
                              src/scarabee/_scarabee/python/depletion_chain.cpp
                              src/scarabee/_scarabee/python/depletion_matrix.cpp
                            )

target_include_directories(_scarabee PRIVATE include)
target_compile_features(_scarabee PRIVATE cxx_std_20)
target_link_libraries(_scarabee PUBLIC xtl xtensor xtensor-python htl HighFive hdf5-static Eigen3::Eigen spdlog::spdlog ImApp::ImApp cereal::cereal)
target_include_directories(_scarabee PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/scarabee/_scarabee/include>
)
target_compile_definitions(_scarabee PRIVATE SCARABEE_MAJOR_VERSION=${PROJECT_VERSION_MAJOR})
target_compile_definitions(_scarabee PRIVATE SCARABEE_MINOR_VERSION=${PROJECT_VERSION_MINOR})
target_compile_definitions(_scarabee PRIVATE SCARABEE_PATCH_VERSION=${PROJECT_VERSION_PATCH})

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") # Comile options for Windows
  target_compile_options(_scarabee PRIVATE /W3)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU") # Compile options for GCC
  target_compile_options(_scarabee PRIVATE -W -Wall -Wextra -Wconversion -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang") # Compile options for Clang
  target_compile_options(_scarabee PRIVATE -W -Wall -Wextra -Wconversion -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel") # Compile options for Intel
  target_compile_options(_scarabee PRIVATE -W -Wall -Wextra -Wconversion -Wpedantic)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") # Comile options for Windows
    target_compile_options(_scarabee PRIVATE /O2)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU") # Compile options for GCC
    target_compile_options(_scarabee PRIVATE -O3)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang") # Compile options for Clang
    target_compile_options(_scarabee PRIVATE -O3)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel") # Compile options for Intel
    target_compile_options(_scarabee PRIVATE -O3)
  endif()
endif()

# Find OpenMP if desired
if(SCARABEE_USE_OMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(_scarabee PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(_scarabee PUBLIC SCARABEE_USE_OMP)
  endif()
endif()

if (SKBUILD_PROJECT_NAME)
  # Generate stub file for type completion
  add_custom_command(TARGET _scarabee POST_BUILD
      COMMAND stubgen --include-docstrings -m _scarabee -o ${SKBUILD_PLATLIB_DIR}/${SKBUILD_PROJECT_NAME}
      WORKING_DIRECTORY $<TARGET_FILE_DIR:_scarabee> 
      COMMENT "Generating Python stub file for _scarabee"
  )

  # The install directory is the output (wheel) directory
  install(TARGETS _scarabee LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME} ) 
endif()
