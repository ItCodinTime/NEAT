cmake_minimum_required(VERSION 3.20)
project(neat_core LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_neat_core
  neat_core/src/bindings.cpp
  neat_core/src/step.cpp
)

target_include_directories(_neat_core PRIVATE neat_core/include)
target_compile_features(_neat_core PRIVATE cxx_std_17)

install(TARGETS _neat_core DESTINATION neat_optim)
