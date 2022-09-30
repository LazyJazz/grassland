file(GLOB textures *.png *.jpg *.hdr *.bmp *.jpeg)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/textures)

foreach(texture ${textures})
    file(COPY ${texture} DESTINATION ${CMAKE_BINARY_DIR}/textures)
endforeach()
