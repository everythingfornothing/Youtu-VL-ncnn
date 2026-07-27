#include "youtu_vl/portable_gemm.hpp"

#include <cmath>
#include <iostream>

namespace {

bool close(float actual, float expected) {
    return std::fabs(actual - expected) <= 1e-6f;
}

}  // namespace

int main() {
    const float left[] = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 4.0f};
    const float right_nn[] = {2.0f, -1.0f, 0.5f, 3.0f, -2.0f, 4.0f};
    float output_nn[4]{};
    youtu_vl::matmul_nn(left, right_nn, output_nn, 2, 2, 3);
    const float expected_nn[] = {-3.0f, 17.0f, -9.75f, 18.5f};
    for (int index = 0; index < 4; ++index) {
        if (!close(output_nn[index], expected_nn[index])) {
            std::cerr << "matmul_nn mismatch at " << index << "\n";
            return 1;
        }
    }

    const float right_nt[] = {1.0f, 0.0f, 2.0f, -1.0f, 3.0f, 0.5f};
    float output_nt[4]{};
    youtu_vl::matmul_nt(left, right_nt, output_nt, 2, 2, 3, 0.5f);
    const float expected_nt[] = {3.5f, 3.25f, 3.5f, 2.25f};
    for (int index = 0; index < 4; ++index) {
        if (!close(output_nt[index], expected_nt[index])) {
            std::cerr << "matmul_nt mismatch at " << index << "\n";
            return 1;
        }
    }
    std::cout << "portable GEMM: PASS\n";
    return 0;
}
