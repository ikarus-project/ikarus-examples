# SPDX-FileCopyrightText: 2021-2025 The Ikarus Developers ikarus@ibb.uni-stuttgart.de
# SPDX-License-Identifier: LGPL-3.0-or-later

file(GLOB programSourceFiles CONFIGURE_DEPENDS *.cpp)
include_directories(../libAddons/eigen/eigenAddons)
add_custom_target(AllExamples) # Create target that includes all examples

find_dependency(spdlog)
find_dependency(Matplot++)
find_dependency(Eigen3)

foreach(programSourceFile ${programSourceFiles})
  get_filename_component(programName ${programSourceFile} NAME_WLE)
  add_executable(${programName} ${programSourceFile})
  set_target_properties(${programName} PROPERTIES CXX_STANDARD 20 OUTPUT_NAME ${programName})
  add_dune_pythonlibs_flags(${programName})
  target_compile_options(
    ${programName} PUBLIC -Wno-unused-variable -Wno-unused-parameter -Wno-unused-but-set-variable
                          -Wno-deprecated-copy -ftemplate-backtrace-limit=0
  )

  target_link_libraries(${programName} PRIVATE ikarus)
  add_dependencies(AllExamples ${programName})
  find_package(PythonLibs REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIRS})
  target_link_libraries(${programName} PUBLIC ${PYTHON_LIBRARIES})
endforeach()

# Copy the example grid used for testing into the build dir
file(COPY auxiliaryFiles/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/auxiliaryFiles)
