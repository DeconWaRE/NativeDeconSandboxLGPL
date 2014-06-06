# this project implements GPU functionality 
project(deconwareGPUProject)

# Find Cuda
find_package(CUDA REQUIRED)

# add files to the library
cuda_add_library(deconwareGPU SHARED dot.cu)
CUDA_ADD_CUFFT_TO_TARGET( deconwareGPU )

set(LIB_PATH ${CMAKE_BINARY_DIR}/lib)

INSTALL(TARGETS deconwareGPU DESTINATION ${LIB_PATH})

