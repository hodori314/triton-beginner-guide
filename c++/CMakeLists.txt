cmake_minimum_required(VERSION 3.0)

project(MyProgram)

# add source file
add_executable(myprogram client.cpp)

# add library directory 
link_directories(/path/to/your/triton/lib)

# search all .a and .so files within LIBRARIES folder
file(GLOB LIBRARIES /path/to/your/triton/lib/*.a /path/to/your/triton/lib/*.so)

# link library
target_link_libraries(myprogram PRIVATE ${LIBRARIES})
