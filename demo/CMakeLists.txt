
function(add_demo demo_name)
    set(exe_name ${demo_name}_demo)
    add_executable(${exe_name})
    file(GLOB demo_sources ${demo_name}/*.cpp ${demo_name}/*.h)
    foreach(demo_source ${demo_sources})
        target_sources(${exe_name} PRIVATE ${demo_source})
    endforeach()
    target_include_directories(${exe_name} PRIVATE ${GRASSLAND_INCLUDE_DIR} ${GRASSLAND_EXTERNAL_INCLUDE_DIRS})
    target_link_libraries(${exe_name} PRIVATE grassland)
    add_dependencies(${exe_name} compiled_shaders_target)
endfunction()

add_demo(rotating_cube)
add_demo(rotating_cube_headless)
add_demo(framework)
add_demo(blend_test)
add_demo(font)
add_demo(imgui)
add_demo(raytracing)
