cmake_minimum_required(VERSION 3.22)

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake)

project(grassland)

if (CMAKE_TOOLCHAIN_FILE)
    message("Using vcpkg: ${CMAKE_TOOLCHAIN_FILE}")
else ()
    message("Please add -DCMAKE_TOOLCHAIN_FILE={current_dir_path}/external/vcpkg/scripts/buildsystems/vcpkg.cmake to cmake options.")
endif ()

find_package(CUDA)

if (CUDA_FOUND)
    message("CUDA Found.")

    if (NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
        set(CMAKE_CUDA_ARCHITECTURES 52 61 75)
    endif ()

    project(grassland CUDA CXX)
    add_compile_definitions(GRASSLAND_CUDA_ENABLE)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif ()

set(CMAKE_CXX_STANDARD 17)

if (MSVC AND ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    message("Using msvc-clang compiler")
    add_compile_options(/EHsc)
endif ()

set(GRASSLAND_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GRASSLAND_INCLUDE_DIR ${GRASSLAND_SOURCE_DIR}/src)
set(GRASSLAND_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

find_package(Vulkan REQUIRED)
if (Vulkan_FOUND)
    message("Vulkan found: ${Vulkan_VERSION}")
    message("Vulkan include dir: ${Vulkan_INCLUDE_DIRS}")
    message("Vulkan libraries: ${Vulkan_LIBRARIES}")
    list(APPEND GRASSLAND_EXTERNAL_INCLUDE_DIRS ${Vulkan_INCLUDE_DIRS})
else ()
    message(FATAL_ERROR "Vulkan SDK not found!")
endif ()

set(FT_DISABLE_HARFBUZZ ON CACHE INTERNAL "" FORCE)

# add_subdirectory(external)
find_package(glm CONFIG REQUIRED)
find_package(freetype CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui REQUIRED)
find_package(spdlog REQUIRED)
find_package(SFML COMPONENTS system window graphics audio network REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(glslang CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

message("${VCPKG_INCLUDE_DIR}")

include(scripts/pack_shaders.cmake)

find_path(VCPKG_INCLUDE_DIR "glm/glm.hpp")
message("Vcpkg include path: ${VCPKG_INCLUDE_DIR}")
list(APPEND GRASSLAND_EXTERNAL_INCLUDE_DIRS ${VCPKG_INCLUDE_DIR})


#list(APPEND GRASSLAND_EXTERNAL_INCLUDE_DIRS ${glfw_DIR})
#list(APPEND GRASSLAND_EXTERNAL_INCLUDE_DIRS ${glm_DIR})
#list(APPEND GRASSLAND_EXTERNAL_INCLUDE_DIRS ${spdlog_INCLUDE_DIRS})
#list(APPEND GRASSLAND_EXTERNAL_INCLUDE_DIRS ${freetype_DIR}/include)
#list(APPEND GRASSLAND_EXTERNAL_INCLUDE_DIRS ${stb_DIR})
#list(APPEND GRASSLAND_EXTERNAL_INCLUDE_DIRS ${imgui_DIR})
# message("${GRASSLAND_EXTERNAL_INCLUDE_DIRS}")

add_subdirectory(src)

if (GRASSLAND_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    add_subdirectory(assets)
    add_subdirectory(demo)
else ()
    set(GRASSLAND_INCLUDE_DIRS ${GRASSLAND_INCLUDE_DIR} ${GRASSLAND_EXTERNAL_INCLUDE_DIRS} PARENT_SCOPE)
    message("Grassland is included as subdirectory, not building demos.")
endif ()
