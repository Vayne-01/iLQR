cmake_minimum_required(VERSION 3.16)

project(ilqr)

set(CMAKE_CXX_STANDARD 17)

find_package(autodiff COMPONENTS REQUIRED)

find_package(Eigen3 COMPONENTS REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include/
                    ${Python_INCLUDE_DIRS})

aux_source_directory(src DIR_SRCS)

add_executable(multivehicle multivehicle_exam.cpp ${DIR_SRCS})

target_link_libraries(multivehicle autodiff::autodiff Eigen3::Eigen
                      Python3::Python Python3::Module Python3::NumPy)

add_executable(MassPointPlanning mass_point_planning.cpp ${DIR_SRCS})

target_link_libraries(MassPointPlanning autodiff::autodiff Eigen3::Eigen
                      Python3::Python Python3::Module Python3::NumPy)
