cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test_torch)

include_directories(/software/CMake-hdf5-1.10.1/hdf5-1.10.1/src
	/software/CMake-hdf5-1.10.1/hdf5-1.10.1/c++/src
	/software/CMake-hdf5-1.10.1/hdf5-1.10.1/bin
        /software/CMake-hdf5-1.10.1/build)
set(_hdf5_libs
	/software/CMake-hdf5-1.10.1/build/bin/libhdf5_cpp.a
	/software/CMake-hdf5-1.10.1/build/bin/libhdf5.a -ldl -lz -lsz)


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(test_torch test_torch.cpp)

target_link_libraries(test_torch "${TORCH_LIBRARIES}" ${_hdf5_libs})

set_property(TARGET test_torch PROPERTY CXX_STANDARD 14)
