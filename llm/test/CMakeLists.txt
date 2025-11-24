cmake_minimum_required(VERSION 3.10)
project(llm_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_FILES_1 llm_test.cpp)
add_executable(llm_test ${SOURCE_FILES_1})

include_directories(${RKLLM_API_PATH}/include)

target_link_libraries(llm_test    
    /usr/local/lib/libzmq_component.so  
    zmq )

