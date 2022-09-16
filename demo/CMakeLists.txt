add_executable(grassland-demo)

target_sources(grassland-demo PRIVATE main.cpp)
target_include_directories(grassland-demo PRIVATE ${GRASSLAND_INCLUDE_DIR})
target_link_libraries(grassland-demo PRIVATE grassland)
