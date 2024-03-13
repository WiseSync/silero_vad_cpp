
cmake_minimum_required( VERSION 3.5 )
project( silero_vad_cpp VERSION 0.2 LANGUAGES CXX )

set (BUILD_EXAMPLES ON)
set( SRC_DIR src )
set( INCLUDE_DIRS include )

find_package(onnxruntime REQUIRED)



file(GLOB_RECURSE SOURCE_FILES
    "${SRC_DIR}/*.cpp"
    "${SRC_DIR}/*.cc"
)

add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC onnxruntime::onnxruntime)
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRS})
file(COPY model DESTINATION ${CMAKE_BINARY_DIR} )

if(${BUILD_EXAMPLES})
    message(STATUS "BUILDING ${PROJECT_NAME} EXAMPLES")
    find_package(SDL2 REQUIRED)
    if(SDL2_FOUND)
        message(STATUS "[SDL2_INCLUDES: ${SDL2_INCLUDE_DIRS}]")
        message(STATUS "[SDL2_LIBRARIES: ${SDL2_LIBRARIES}]")
    endif()
    
    add_executable(${PROJECT_NAME}_example example.cpp)
    target_include_directories(${PROJECT_NAME}_example PRIVATE ${SDL2_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME}_example PRIVATE ${PROJECT_NAME} ${SDL2_LIBRARIES})
endif()


