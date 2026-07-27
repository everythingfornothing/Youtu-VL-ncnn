cmake_minimum_required(VERSION 3.20)

project(YoutuVLNcnn VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(YOUTU_VL_BUILD_TESTS "Build lightweight validation tools" ON)
set(
    YOUTU_VL_TEST_MODEL_ROOT
    ""
    CACHE PATH
    "Optional model root used by local ncnn lifecycle tests"
)
find_package(ncnn CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(JPEG REQUIRED)

add_executable(
    youtu_vl_ncnn
    src/youtu_vl_cli.cpp
    src/frontend.cpp
)
target_include_directories(
    youtu_vl_ncnn
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(
    youtu_vl_ncnn
    PRIVATE
        ncnn
        OpenMP::OpenMP_CXX
        JPEG::JPEG
)
if(WIN32)
    target_link_libraries(youtu_vl_ncnn PRIVATE shell32)
endif()

if(MSVC)
    target_compile_options(youtu_vl_ncnn PRIVATE /W4 /permissive-)
else()
    target_compile_options(youtu_vl_ncnn PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS youtu_vl_ncnn RUNTIME DESTINATION bin)

if(YOUTU_VL_BUILD_TESTS)
    enable_testing()
    add_test(NAME youtu_vl_cli_usage COMMAND youtu_vl_ncnn)
    set_tests_properties(youtu_vl_cli_usage PROPERTIES WILL_FAIL TRUE)

    add_executable(test_portable_gemm tests/test_portable_gemm.cpp)
    target_include_directories(
        test_portable_gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(test_portable_gemm PRIVATE OpenMP::OpenMP_CXX)
    add_test(NAME youtu_vl_portable_gemm COMMAND test_portable_gemm)

    add_executable(
        test_persistent_module
        tests/test_persistent_module.cpp
    )
    target_include_directories(
        test_persistent_module
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(
        test_persistent_module
        PRIVATE
            ncnn
            OpenMP::OpenMP_CXX
    )
    if(MSVC)
        target_compile_options(test_persistent_module PRIVATE /W4 /permissive-)
    else()
        target_compile_options(
            test_persistent_module PRIVATE -Wall -Wextra -Wpedantic
        )
    endif()

    if(
        YOUTU_VL_TEST_MODEL_ROOT
        AND EXISTS "${YOUTU_VL_TEST_MODEL_ROOT}/models/youtu_merger.ncnn.bin"
        AND EXISTS "${YOUTU_VL_TEST_MODEL_ROOT}/vision_hidden_ncnn_cpp.npy"
    )
        add_test(
            NAME youtu_vl_persistent_module
            COMMAND
                test_persistent_module
                "${YOUTU_VL_TEST_MODEL_ROOT}/models"
                "${YOUTU_VL_TEST_MODEL_ROOT}/vision_hidden_ncnn_cpp.npy"
        )
    endif()
endif()
