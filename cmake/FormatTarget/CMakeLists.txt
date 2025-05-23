# SPDX-FileCopyrightText: 2021-2025 The Ikarus Developers ikarus@ibb.uni-stuttgart.de
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(IkarusFormat)

if(ADD_FORMATTARGET)
  include(../CPM.cmake)
  CPMAddPackage(
    NAME Format.cmake
    VERSION 1.8.1
    GITHUB_REPOSITORY TheLartians/Format.cmake
    OPTIONS "FORMAT_SKIP_CMAKE NO" "CMAKE_FORMAT_EXCLUDE config.h.cmake"
  )
endif()

if(ADD_CODESPELLTARGET)
  find_package(Python)
  if(Python_FOUND)
    message("Adding codespell targets")
    message("${CMAKE_SOURCE_DIR}")
    message(${CMAKE_SOURCE_DIR})
    set(CODESPELLCOMAND
        codespell
        -c
        -s
        -S
        literature.bib
        --count
        ${CMAKE_SOURCE_DIR}/src
        README.md
    )
    add_custom_target(
      codespell
      COMMAND ${CODESPELLCOMAND}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_custom_target(
      fix-codespell
      COMMAND ${CODESPELLCOMAND} -w
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
  else()
    message(STATUS "python not found, codespell target not available")

  endif()

endif()
