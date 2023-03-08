
function(add_demo demo_name)
    if (${demo_name} MATCHES "cuda$")
        if(CUDA_FOUND)
            file(GLOB demo_sources ${demo_name}/*.cpp ${demo_name}/*.hpp ${demo_name}/*.c ${demo_name}/*.h ${demo_name}/*.cu ${demo_name}/*.cuh)
        else()
            message("CUDA Not Found: ignoring ${demo_name}")
            return()
        endif()
    else()
        file(GLOB demo_sources ${demo_name}/*.cpp ${demo_name}/*.hpp ${demo_name}/*.c ${demo_name}/*.h)
    endif()

    set(exe_name ${demo_name}_demo)
    add_executable(${exe_name})
    set_target_properties(${exe_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${demo_name})

    foreach(demo_source ${demo_sources})
        target_sources(${exe_name} PRIVATE ${demo_source})
    endforeach()
    target_include_directories(${exe_name} PRIVATE ${GRASSLAND_INCLUDE_DIR} ${GRASSLAND_EXTERNAL_INCLUDE_DIRS})
    target_link_libraries(${exe_name} PRIVATE grassland)

    file(GLOB shader_extra_files ${demo_name}/*.glsl)

    if (APPLE)
        file(GLOB shader_files ${demo_name}/*.vert ${demo_name}/*.frag ${demo_name}/*.comp)
        set(spv_version vulkan1.1)
    else()
        file(GLOB shader_files ${demo_name}/*.vert ${demo_name}/*.frag ${demo_name}/*.comp ${demo_name}/*.rgen ${demo_name}/*.rchit ${demo_name}/*.rint ${demo_name}/*.rmiss)
        set(spv_version vulkan1.2)
    endif()

    if (shader_files)
        message("[${demo_name}] Shader file detected.")
        foreach (shader ${shader_files})
            get_filename_component(file_name ${shader} NAME)
            get_filename_component(full_path ${shader} ABSOLUTE)
            set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/${demo_name})
            set(output_file ${output_dir}/${file_name}.spv)
            set(compiled_shaders ${compiled_shaders} ${output_file})
            set_source_files_properties(${shader} PROPERTIES HEADER_FILE_ONLY TRUE)
            add_custom_command(
                    OUTPUT ${output_file}
                    COMMAND ${CMAKE_COMMAND} -E make_directory ${output_dir}
                    COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} --target-env ${spv_version} -V ${full_path} -o ${output_file}
                    DEPENDS ${full_path}
            )
        endforeach ()

        source_group(${demo_name}/shaders FILES ${shader_files} ${shader_extra_files})

        add_custom_target(
                ${demo_name}_compiled_shaders_target
                DEPENDS ${compiled_shaders}
                SOURCES ${shader_files} ${shader_extra_files})

        # Before we can correctly detect shader file dependency, let's simply recompile all shader files everytime we build.
        add_custom_command(TARGET ${demo_name}_compiled_shaders_target
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E touch_nocreate ${shader_files})
        add_dependencies(${exe_name} ${demo_name}_compiled_shaders_target)
    endif()
endfunction()

file(GLOB demo_dirs *)

foreach(demo_dir ${demo_dirs})
    if (IS_DIRECTORY ${demo_dir})
        file(RELATIVE_PATH demo_name ${CMAKE_CURRENT_SOURCE_DIR} ${demo_dir})
        message("Loading demo: ${demo_name}")
        add_demo(${demo_name})
    endif()
endforeach()
