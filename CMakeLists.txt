cmake_minimum_required(VERSION 3.22)

project(grassland)

set(CMAKE_CXX_STANDARD 17)

set(GRASSLAND_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GRASSLAND_INCLUDE_DIR ${GRASSLAND_SOURCE_DIR}/include)

find_package(Vulkan REQUIRED)
if (Vulkan_FOUND)
    message("Vulkan found: ${Vulkan_VERSION}")
else()
    message(FATAL_ERROR "Vulkan SDK not found!")
endif()

add_subdirectory(external)
list(APPEND GRASSLAND_EXTERNAL_INCLUDE_DIRS ${GLFW_SOURCE_DIR}/include)
list(APPEND GRASSLAND_EXTERNAL_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/external/glm)
list(APPEND GRASSLAND_EXTERNAL_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/external/spdlog/include)
list(APPEND GRASSLAND_EXTERNAL_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/external/freetype/include)

add_subdirectory(src)
add_subdirectory(demo)
