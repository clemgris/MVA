cmake_minimum_required(VERSION 3.4)
set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "") #Apple Silicon
project(Panorama)
list(APPEND CMAKE_FRAMEWORK_PATH /Library/Frameworks) #Mac, why not auto?
find_package(Imagine REQUIRED)

add_executable(Panorama Panorama.cpp)
ImagineUseModules(Panorama LinAlg Images)
