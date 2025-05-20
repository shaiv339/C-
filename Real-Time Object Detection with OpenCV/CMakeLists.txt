cmake_minimum_required(VERSION 3.10)
project(RealTimeObjectDetection)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(rt_detect
    src/main.cpp
    src/detector.cpp
)

target_link_libraries(rt_detect ${OpenCV_LIBS})