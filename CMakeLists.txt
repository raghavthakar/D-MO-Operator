cmake_minimum_required(VERSION 3.10)
project(MiningDomain)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_PREFIX_PATH libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(include)
include_directories("libtorch/include")

# Source files
set(SOURCES 
    src/environment.cpp
    src/policy.cpp
    src/team.cpp
    src/evolutionary.cpp
    src/main.cpp
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link the yaml-cpp library
target_link_libraries(${PROJECT_NAME} yaml-cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")