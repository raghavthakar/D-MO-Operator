cmake_minimum_required(VERSION 3.12)
project(MOD)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Optimization flags
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto -fopenmp -march=native -fvisibility-inlines-hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_PREFIX_PATH libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Look for an installation of pagmo in the system.
find_package(Pagmo REQUIRED)

# Include directories
include_directories(include)
include_directories("libtorch/include")

# Source files
set(SOURCES
    src/MOREPBaseAgent.cpp
    src/MOREPPolicy.cpp
    src/MOREPDomain.cpp
    src/MOBPBaseAgent.cpp
    src/MOBPDomain.cpp
    src/environment.cpp
    src/team.cpp
    src/evolutionary_utils.cpp
    src/MOD.cpp
    src/MOD_ablated.cpp
    src/MOD_team_ablated.cpp
    src/NSGA_II.cpp
    src/main.cpp
    src/DNSGA.cpp
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} yaml-cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} Pagmo::pagmo)
target_link_libraries(${PROJECT_NAME} tbb)
