cmake_minimum_required(VERSION 3.0)

project(MyProgram)

# add source file
add_executable(myprogram client.cpp)

# add library directory 
link_directories(/home/libtriton/lib)

# search all .a and .so files within LIBRARIES folder
file(GLOB LIBRARIES /path/to/your/lib/*.a /path/to/your/lib/*.so)

# link library
target_link_libraries(myprogram PRIVATE ${LIBRARIES})
