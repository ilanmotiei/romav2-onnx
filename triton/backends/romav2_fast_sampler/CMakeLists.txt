cmake_minimum_required(VERSION 3.24)

project(romav2_fast_sampler_backend LANGUAGES CXX CUDA)

option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_COMMON_REPO_TAG "r25.07" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r25.07" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r25.07" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

find_package(CUDAToolkit REQUIRED)

configure_file(src/libtriton_romav2_fast_sampler.ldscript libtriton_romav2_fast_sampler.ldscript COPYONLY)

add_library(
  triton-romav2-fast-sampler-backend SHARED
  src/romav2_fast_sampler.cc
)

target_compile_features(
  triton-romav2-fast-sampler-backend PRIVATE
  cxx_std_${TRITON_MIN_CXX_STANDARD}
)

target_compile_options(
  triton-romav2-fast-sampler-backend PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits>
)

target_link_libraries(
  triton-romav2-fast-sampler-backend
  PRIVATE
    CUDA::cudart
    triton-core-serverapi
    triton-core-backendapi
    triton-core-serverstub
    triton-backend-utils
)

set_target_properties(
  triton-romav2-fast-sampler-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_romav2_fast_sampler
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_romav2_fast_sampler.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_romav2_fast_sampler.ldscript"
)

include(GNUInstallDirs)
install(
  TARGETS triton-romav2-fast-sampler-backend
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/romav2_fast_sampler
)
