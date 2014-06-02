# this project is used to wrap YacuDecu cuda deconvolution by Bob Pepin
project(YacuDecuWrapperProject)

# Find Swig
FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

# Find JNI
find_package(JNI REQUIRED)

# Find Cuda
find_package(CUDA REQUIRED)

# Find Java
find_package(Java REQUIRED)
INCLUDE(UseJava)

include_directories(${JNI_INCLUDE_DIRS})

# add the cuda library
cuda_add_library(YacuDecu SHARED deconv.cu)
CUDA_ADD_CUFFT_TO_TARGET( YacuDecu )

# tell swig to put the .java files into javasrc directory
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR}/javasrc)

# tell swig to use com.deconware.wrappers namespace
SET(CMAKE_SWIG_FLAGS -package com.deconware.wrappers)

# add the swig java wrapper module
swig_add_module(YacuDecuJavaSwig java YacuDecuWrapper.i "deconv.cu")
# link swig module with the YacuDecuWrapper library
swig_link_libraries(YacuDecuJavaSwig YacuDecu)

# glob all the java files
FILE(GLOB JAVA_SOURCE ${CMAKE_SWIG_OUTDIR}/*.java)
set(CMAKE_JAVA_COMPILE_FLAGS "-source" "1.6" "-target" "1.6")
# make the jar wrapper
add_jar(YacuDecuJavaWrapper ${JAVA_SOURCE} OUTPUT_DIR ${JAR_PATH})

add_dependencies(YacuDecuJavaWrapper YacuDecuJavaSwig)

