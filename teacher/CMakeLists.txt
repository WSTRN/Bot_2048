cmake_minimum_required(VERSION 3.15...4.0)
project(teacher_2048 LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYBIND11_FINDPATH ON)
find_package(pybind11 CONFIG REQUIRED)

enable_testing()
add_library(catch2_main lib/catch2_main.cpp)
target_include_directories(catch2_main PUBLIC lib)
link_libraries(catch2_main)

include_directories(${PROJECT_SOURCE_DIR}/include)

pybind11_add_module(teacher_2048 teacher_2048.cpp)
add_executable(teacher_2048_test_exe teacher_2048.test.cpp)
add_test(teacher_2048_test teacher_2048_test_exe)

add_custom_command(TARGET teacher_2048 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:teacher_2048> ../../
)