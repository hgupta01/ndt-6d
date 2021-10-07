cmake_minimum_required(VERSION 3.5)
project(ndt_registration_knn)

set(CMAKE_SHARED_MODULE_PREFIX "")


find_package(PkgConfig)
pkg_check_modules(PC_FLANN flann)

find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 3.3.4 REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set( OpenCV_DIR "/home/onepiece/.local/lib/cmake/opencv4" )
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
)

if(UNIX)
  set(LZ4_LIBRARIES "/usr/lib/x86_64-linux-gnu/liblz4.so")
endif()

# add_library(ndt_registration_knn MODULE src/ndt_registration_6d.cpp src/utils.cpp)
add_executable(main src/main.cpp)
target_link_libraries(main
  ${OpenMP_LIBS}
  ${CERES_LIBRARIES}
  ${LZ4_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${PCL_LIBRARIES} 
  ${OpenCV_LIBS}
)