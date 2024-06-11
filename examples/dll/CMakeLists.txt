cmake_minimum_required(VERSION 3.19.2)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.11" CACHE STRING "Minimum OS X deployment version")
set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "OS X universal binaries")

project(hwpv)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(WIN32)
  set(CMAKE_CXX_FLAGS "/O2 -D_WINDOWS")
endif()
if(UNIX)
  set(CMAKE_CXX_FLAGS "-O3 -fPIC")
endif()
if(APPLE)
  set(CMAKE_CXX_FLAGS "-O3 -fPIC --std=c++11")
endif()

#include_directories(Include)
#
## shared library for OpenDSS
#add_subdirectory(AMD)

# demonstration program for local testing
add_executable(hwpv hwpv.cpp)
#target_compile_options(KLDemo PUBLIC "-DREAD_KUNDERT")
#target_link_libraries(KLDemo PRIVATE
# KLU
# AMD
# BTF
# COLAMD
# CSparse
# CZSparse)
#if(UNIX)
#  target_link_libraries(KLDemo PRIVATE m)
#endif()
#
## install for testing
#install(TARGETS klusolve DESTINATION "${CMAKE_SOURCE_DIR}/Lib")
#install(TARGETS KLDemo DESTINATION "${CMAKE_SOURCE_DIR}/Test")
##install for deployment
#install(TARGETS klusolve
#  RUNTIME DESTINATION bin
#  LIBRARY DESTINATION lib
#  ARCHIVE DESTINATION lib)
#install(FILES KLUSolve/KLUSolve.h DESTINATION include)

