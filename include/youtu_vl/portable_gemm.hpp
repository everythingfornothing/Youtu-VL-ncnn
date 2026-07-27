#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace youtu_vl {

// Portable row-major matrix multiplication used by the attention bridge.
// Accumulation follows increasing K order for deterministic FP32 behavior.
inline void matmul_nt(
    const float* left,
    const float* right,
    float* output,
    int rows,
    int columns,
    int shared,
    float scale) {
    if (rows < 0 || columns < 0 || shared < 0) {
        throw std::invalid_argument("matmul_nt received a negative dimension");
    }
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; ++row) {
        const float* left_row = left + static_cast<size_t>(row) * shared;
        float* output_row = output + static_cast<size_t>(row) * columns;
        for (int column = 0; column < columns; ++column) {
            const float* right_row =
                right + static_cast<size_t>(column) * shared;
            float sum = 0.0f;
            for (int index = 0; index < shared; ++index) {
                sum += left_row[index] * right_row[index];
            }
            output_row[column] = sum * scale;
        }
    }
}

inline void matmul_nn(
    const float* left,
    const float* right,
    float* output,
    int rows,
    int columns,
    int shared) {
    if (rows < 0 || columns < 0 || shared < 0) {
        throw std::invalid_argument("matmul_nn received a negative dimension");
    }
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; ++row) {
        const float* left_row = left + static_cast<size_t>(row) * shared;
        float* output_row = output + static_cast<size_t>(row) * columns;
        std::fill(output_row, output_row + columns, 0.0f);
        for (int index = 0; index < shared; ++index) {
            const float factor = left_row[index];
            const float* right_row =
                right + static_cast<size_t>(index) * columns;
            for (int column = 0; column < columns; ++column) {
                output_row[column] += factor * right_row[column];
            }
        }
    }
}

}  // namespace youtu_vl
