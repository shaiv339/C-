cmake_minimum_required(VERSION 3.10)
project(LlamaWrapperExample)

set(CMAKE_CXX_STANDARD 17)

# llama.cpp (you must clone it manually)
add_subdirectory(../llama.cpp build_llama EXCLUDE_FROM_ALL)

include_directories(../llama.cpp)

add_executable(main
    main.cpp
    llama_wrapper.cpp
)

target_link_libraries(main PRIVATE llama)