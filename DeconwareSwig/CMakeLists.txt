# this project is used to wrap YacuDecu cuda deconvolution by Bob Pepin
project(DeconwareSwigProject)

# Find Swig
FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

# Find JNI
find_package(JNI REQUIRED)

# Find Java
find_package(Java REQUIRED)
INCLUDE(UseJava)

include_directories(${JNI_INCLUDE_DIRS})

# tell swig to put the .java files into javasrc directory
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR}/javasrc)

# tell swig to use com.deconware.wrappers namespace
SET(CMAKE_SWIG_FLAGS -package com.deconware.wrappers)

# add the swig java wrapper module
swig_add_module(DeconwareSwig java DeconwareSwig.i)

set(LIB_PATH ${CMAKE_BINARY_DIR}/lib)

# link swig module with the libraries
swig_link_libraries(DeconwareSwig deconwareGPU)

# glob all the java files
FILE(GLOB JAVA_SOURCE ${CMAKE_SWIG_OUTDIR}/*.java)
set(CMAKE_JAVA_COMPILE_FLAGS "-source" "1.6" "-target" "1.6")
# make the jar wrapper
add_jar(DeconwareJavaWrapper ${JAVA_SOURCE} OUTPUT_DIR ${JAR_PATH} VERSION 1.0)


#INSTALL(TARGETS DeconwareSwig DESTINATION ${LIB_PATH})






