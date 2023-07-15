cmake_minimum_required(VERSION 3.16)

project(ilqr)

set(CMAKE_CXX_STANDARD 17)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(autodiff REQUIRED)

find_package(Eigen3 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include/)

aux_source_directory(src DIR_SRCS)

add_executable(simple_ilqr main.cpp ${DIR_SRCS})

target_link_libraries(simple_ilqr autodiff::autodiff Eigen3::Eigen)
