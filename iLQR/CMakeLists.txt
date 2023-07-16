cmake_minimum_required(VERSION 3.16)

project(ilqr)

set(CMAKE_CXX_STANDARD 17)

find_package(autodiff COMPONENTS REQUIRED)

find_package(Eigen3 COMPONENTS REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include/
                    ${Python_INCLUDE_DIRS})

aux_source_directory(src DIR_SRCS)

add_executable(simple_ilqr main.cpp ${DIR_SRCS})

target_link_libraries(simple_ilqr autodiff::autodiff Eigen3::Eigen
                      Python3::Python Python3::Module Python3::NumPy)
