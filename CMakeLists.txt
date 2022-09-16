cmake_minimum_required(VERSION 3.22)

project(grassland)

set(CMAKE_CXX_STANDARD 17)

set(GRASSLAND_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GRASSLAND_INCLUDE_DIR ${GRASSLAND_SOURCE_DIR}/include)

add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(demo)
