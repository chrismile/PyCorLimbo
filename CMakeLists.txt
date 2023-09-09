cmake_minimum_required(VERSION 3.11)
project(PyCorLimbo)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES src/*.cpp src/*.c src/*.h src/*.hpp)

include_directories(src)

if(MSVC)
    set(CMAKE_CXX_FLAGS "-W3 /EHsc")
elseif(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-Wall")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-Wall")
endif()

#add_library(pycorlimbo SHARED ${SOURCES})
add_executable(pycorlimbo ${SOURCES})

cmake_policy(SET CMP0012 NEW)
find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)

find_package(Python3 COMPONENTS Development REQUIRED)
target_link_libraries(pycorlimbo PRIVATE Python3::Python)

target_link_libraries(pycorlimbo PRIVATE ${TORCH_LIBRARIES})
target_include_directories(pycorlimbo PUBLIC ${TORCH_INCLUDE_DIR})

find_package(Boost COMPONENTS system filesystem REQUIRED)
target_link_libraries(pycorlimbo PRIVATE ${Boost_LIBRARIES})
target_include_directories(pycorlimbo PRIVATE ${Boost_INCLUDE_DIR})

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Eigen3 includes.
if(VCPKG_TOOLCHAIN)
    find_package(Eigen3 CONFIG REQUIRED)
else()
    find_package(Eigen3 REQUIRED)
endif()
if (NOT (EIGEN3_VERSION_STRING LESS 3.3))
    target_link_libraries(pycorlimbo PRIVATE Eigen3::Eigen)
else()
    target_include_directories(pycorlimbo PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

# Include directories and link libraries for Limbo.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/limbo" AND Eigen3_FOUND)
    target_include_directories(pycorlimbo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/limbo/src")
    if (VCPKG_TOOLCHAIN)
        find_package(NLopt CONFIG REQUIRED)
        target_link_libraries(pycorlimbo PRIVATE NLopt::nlopt)
    else()
        target_link_libraries(pycorlimbo PRIVATE nlopt)
    endif()
    if (MSVC)
        target_link_libraries(pycorlimbo PRIVATE ws2_32)
    endif()
    target_compile_definitions(pycorlimbo PRIVATE USE_LIMBO)
    MESSAGE(STATUS "Found Eigen ${EIGEN3_VERSION_STRING} and Limbo. Enabling Bayesian optimization support.")
else()
    MESSAGE(STATUS "Eigen 3 or Limbo couldn't be found. Switching off Bayesian optimization support.")
endif()


install(
        TARGETS pycorlimbo
        ARCHIVE DESTINATION modules
        LIBRARY DESTINATION modules
        RUNTIME DESTINATION bin
)
