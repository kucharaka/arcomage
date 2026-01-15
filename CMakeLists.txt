cmake_minimum_required(VERSION 3.20)
project(arcomage_rl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- SFML 3 (brew) ----
find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)

# ---- pybind11 (FetchContent) ----
include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# ---- onnxruntime (brew) ----
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS /opt/homebrew/include/onnxruntime /usr/local/include/onnxruntime
)
find_library(ONNXRUNTIME_LIB onnxruntime
        PATHS /opt/homebrew/lib /usr/local/lib
)

if (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "onnxruntime not found. Try: brew install onnxruntime")
endif()

# ---- Engine library ----
add_library(arcomage_engine STATIC
        src/engine.cpp
)
target_include_directories(arcomage_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- Python module (pybind11) ----
pybind11_add_module(arkomag_cpp src/bindings.cpp)
target_link_libraries(arkomag_cpp PRIVATE arcomage_engine)

# put module into build/ so: PYTHONPATH=build python ...
set_target_properties(arkomag_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ---- Game executable ----
add_executable(arcomage_game
        src/main.cpp
)
target_link_libraries(arcomage_game PRIVATE
        arcomage_engine
        SFML::Graphics SFML::Window SFML::System
        ${ONNXRUNTIME_LIB}
)
target_include_directories(arcomage_game PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
