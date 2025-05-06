# SPDX-FileCopyrightText: 2021-2025 The Ikarus Developers ikarus@ibb.uni-stuttgart.de
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.13)
project(
  ikarus-examples
  VERSION 0.2
  LANGUAGES C CXX
)

cmake_policy(SET CMP0087 NEW)

add_definitions(-DDUNE_LOCALFEFUNCTIONS_USE_EIGEN=1)

if(NOT
   (dune-common_DIR
    OR dune-common_ROOT
    OR "${CMAKE_PREFIX_PATH}" MATCHES ".*dune-common.*")
)
  string(REPLACE ${PROJECT_NAME} dune-common dune-common_DIR ${PROJECT_BINARY_DIR})
endif()

# find dune-common and set the module path
find_package(dune-common REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules" ${dune-common_MODULE_PATH})

# include the dune macros
include(DuneMacros)

# start a dune project with information from dune.module
dune_project()

dune_enable_all_packages()

add_subdirectory(src)
add_subdirectory(cmake/modules)

# Add format target from Ikarus
add_subdirectory(cmake/FormatTarget)

# finalize the dune project, e.g. generating config.h etc.
finalize_dune_project()
