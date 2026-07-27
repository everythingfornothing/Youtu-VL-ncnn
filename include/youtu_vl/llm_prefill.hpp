#pragma once

#include "youtu_vl/llm_decoder_layer.hpp"
#include "youtu_vl/portable_gemm.hpp"

#include <filesystem>
#include <vector>

namespace youtu_vl {

struct PrefillOutput {
    Array layer_output;
    Array key_cache;
    Array value_cache;
};

BridgeOutput prefill_bridge(
    const Array& q_flat,
    const Array& kv_b_flat,
    const Array& k_rot_flat,
    const Array& cos,
    const Array& sin) {
    const int sequence_length = q_flat.shape[0];
    Array attention{
        {sequence_length, kAttentionOutputSize},
        std::vector<float>(
            static_cast<size_t>(sequence_length) * kAttentionOutputSize),
    };
    std::vector<float> weights(
        static_cast<size_t>(sequence_length) * sequence_length);
    Array query_states{
        {kNumHeads, sequence_length, kQkHeadDim},
        std::vector<float>(
            static_cast<size_t>(kNumHeads) * sequence_length * kQkHeadDim),
    };
    Array key_cache{
        {kNumHeads, sequence_length, kQkHeadDim},
        std::vector<float>(
            static_cast<size_t>(kNumHeads) * sequence_length * kQkHeadDim),
    };
    Array value_cache{
        {kNumHeads, sequence_length, kValueHeadDim},
        std::vector<float>(
            static_cast<size_t>(kNumHeads) * sequence_length * kValueHeadDim),
    };
    for (int token = 0; token < sequence_length; ++token) {
        const float* key_rope_source =
            k_rot_flat.data.data() +
            static_cast<size_t>(token) * kQkRopeHeadDim;
        for (int head = 0; head < kNumHeads; ++head) {
            const float* q_source =
                q_flat.data.data() +
                static_cast<size_t>(token) * kNumHeads * kQkHeadDim +
                head * kQkHeadDim;
            const float* kv_source =
                kv_b_flat.data.data() +
                static_cast<size_t>(token) *
                    kNumHeads * (kQkNopeHeadDim + kValueHeadDim) +
                head * (kQkNopeHeadDim + kValueHeadDim);
            for (int dimension = 0; dimension < kQkNopeHeadDim; ++dimension) {
                query_states.data[offset3(
                    head,
                    token,
                    dimension,
                    sequence_length,
                    kQkHeadDim)] = q_source[dimension];
                key_cache.data[offset3(
                    head,
                    token,
                    dimension,
                    sequence_length,
                    kQkHeadDim)] = kv_source[dimension];
                value_cache.data[offset3(
                    head,
                    token,
                    dimension,
                    sequence_length,
                    kValueHeadDim)] =
                    kv_source[kQkNopeHeadDim + dimension];
            }
            for (int dimension = 0; dimension < kQkRopeHeadDim; ++dimension) {
                const int rotated_dimension =
                    dimension < kQkRopeHeadDim / 2
                        ? dimension + kQkRopeHeadDim / 2
                        : dimension - kQkRopeHeadDim / 2;
                const float source =
                    interleaved_value(
                        q_source + kQkNopeHeadDim,
                        dimension);
                const float rotated =
                    dimension < kQkRopeHeadDim / 2
                        ? -interleaved_value(
                              q_source + kQkNopeHeadDim,
                              rotated_dimension)
                        : interleaved_value(
                              q_source + kQkNopeHeadDim,
                              rotated_dimension);
                const float rope_cos =
                    cos.data[static_cast<size_t>(token) * kQkRopeHeadDim +
                             dimension];
                const float rope_sin =
                    sin.data[static_cast<size_t>(token) * kQkRopeHeadDim +
                             dimension];
                query_states.data[offset3(
                    head,
                    token,
                    kQkNopeHeadDim + dimension,
                    sequence_length,
                    kQkHeadDim)] =
                    source * rope_cos + rotated * rope_sin;
                const float key_source =
                    interleaved_value(key_rope_source, dimension);
                const float key_rotated =
                    dimension < kQkRopeHeadDim / 2
                        ? -interleaved_value(
                              key_rope_source,
                              rotated_dimension)
                        : interleaved_value(
                              key_rope_source,
                              rotated_dimension);
                key_cache.data[offset3(
                    head,
                    token,
                    kQkNopeHeadDim + dimension,
                    sequence_length,
                    kQkHeadDim)] =
                    key_source * rope_cos + key_rotated * rope_sin;
            }
        }
    }

    for (int head = 0; head < kNumHeads; ++head) {
        const float* query =
            query_states.data.data() +
            static_cast<size_t>(head) * sequence_length * kQkHeadDim;
        const float* key =
            key_cache.data.data() +
            static_cast<size_t>(head) * sequence_length * kQkHeadDim;
        matmul_nt(
            query,
            key,
            weights.data(),
            sequence_length,
            sequence_length,
            kQkHeadDim,
            kScaling);

        for (int token = 0; token < sequence_length; ++token) {
            float* row =
                weights.data() + static_cast<size_t>(token) * sequence_length;
            float maximum = -std::numeric_limits<float>::infinity();
            for (int index = 0; index <= token; ++index) {
                maximum = std::max(maximum, row[index]);
            }
            float denominator = 0.0f;
            for (int index = 0; index <= token; ++index) {
                row[index] = std::exp(row[index] - maximum);
                denominator += row[index];
            }
            for (int index = 0; index <= token; ++index) {
                row[index] /= denominator;
            }
            std::fill(
                row + token + 1,
                row + sequence_length,
                0.0f);
        }

        const float* value =
            value_cache.data.data() +
            static_cast<size_t>(head) * sequence_length * kValueHeadDim;
        std::vector<float> head_output(
            static_cast<size_t>(sequence_length) * kValueHeadDim);
        matmul_nn(
            weights.data(),
            value,
            head_output.data(),
            sequence_length,
            kValueHeadDim,
            sequence_length);
        for (int token = 0; token < sequence_length; ++token) {
            std::copy_n(
                head_output.data() +
                    static_cast<size_t>(token) * kValueHeadDim,
                kValueHeadDim,
                attention.data.begin() +
                    static_cast<size_t>(token) * kAttentionOutputSize +
                    head * kValueHeadDim);
        }
    }
    return {
        std::move(attention),
        {},
        {},
        std::move(key_cache),
        std::move(value_cache),
    };
}

PrefillOutput run_decoder_prefill_layer(
    const DecoderLayer& layer,
    const Array& hidden_states,
    const Array& cos,
    const Array& sin) {
    const auto part_a = layer.attention_input(hidden_states);
    BridgeOutput bridge =
        prefill_bridge(part_a[0], part_a[1], part_a[2], cos, sin);
    const auto part_b = layer.attention_output(
        hidden_states, bridge.attention_result);
    const auto part_c = layer.mlp(part_b[0]);
    return {
        part_c[0],
        std::move(bridge.key_cache),
        std::move(bridge.value_cache),
    };
}

PrefillOutput run_decoder_prefill_layer(
    const std::filesystem::path& layer_dir,
    const Array& hidden_states,
    const Array& cos,
    const Array& sin) {
    return run_decoder_prefill_layer(
        DecoderLayer(layer_dir), hidden_states, cos, sin);
}

}  // namespace youtu_vl
