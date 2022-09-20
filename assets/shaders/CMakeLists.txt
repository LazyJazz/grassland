
file(GLOB shader_files ./*.vert ./*.frag ./*.comp ./*.rgen ./*.rchit ./*.rint ./*.rmiss)

file(GLOB shader_extra_files ./*.glsl)

foreach (shader ${shader_files})
    get_filename_component(file_name ${shader} NAME)
    get_filename_component(full_path ${shader} ABSOLUTE)
    set(output_dir ${CMAKE_BINARY_DIR}/shaders)
    set(output_file ${output_dir}/${file_name}.spv)
    set(compiled_shaders ${compiled_shaders} ${output_file})
    set(compiled_shaders ${compiled_shaders} PARENT_SCOPE)
    message("output spv shader: ${output_file}")
    set_source_files_properties(${shader} PROPERTIES HEADER_FILE_ONLY TRUE)
    add_custom_command(
            OUTPUT ${output_file}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${output_dir}
            COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} --target-env vulkan1.2 -V ${full_path} -o ${output_file}
            DEPENDS ${full_path}
    )
endforeach ()


source_group("Shaders" FILES ${shader_files} ${shader_extra_files})

add_custom_target(
        compiled_shaders_target
        DEPENDS ${compiled_shaders}
        SOURCES ${shader_files} ${shader_extra_files})

# Before we can correctly detect shader file dependency, let's simply recompile all shader files everytime we build.
add_custom_command(TARGET compiled_shaders_target
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E touch_nocreate ${shader_files})
