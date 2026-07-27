#pragma once

#include "youtu_vl/llm_decoder_layer.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace youtu_vl {

ncnn::Mat to_ncnn_mat_vision(const Array& array) {
    if (array.shape.size() == 2) {
        return ncnn::Mat(
                   array.shape[1],
                   array.shape[0],
                   const_cast<float*>(array.data.data()))
            .clone();
    }
    if (array.shape.size() == 3) {
        return ncnn::Mat(
                   array.shape[2],
                   array.shape[1],
                   array.shape[0],
                   const_cast<float*>(array.data.data()))
            .clone();
    }
    throw std::runtime_error("vision ncnn input must be rank 2 or 3");
}

Array run_vision_ncnn(
    const std::filesystem::path& param,
    const std::filesystem::path& binary,
    const std::vector<Array>& inputs,
    const std::string* param_text = nullptr) {
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 1;
    const int param_status = param_text == nullptr
        ? net.load_param(param.string().c_str())
        : net.load_param_mem(param_text->c_str());
    if (param_status != 0) {
        throw std::runtime_error("failed to load ncnn param: " + param.string());
    }
    if (net.load_model(binary.string().c_str()) != 0) {
        throw std::runtime_error("failed to load ncnn model: " + binary.string());
    }

    ncnn::Extractor extractor = net.create_extractor();
    std::vector<ncnn::Mat> input_mats;
    input_mats.reserve(inputs.size());
    for (size_t index = 0; index < inputs.size(); ++index) {
        input_mats.push_back(to_ncnn_mat_vision(inputs[index]));
        const std::string name = "in" + std::to_string(index);
        if (extractor.input(name.c_str(), input_mats.back()) != 0) {
            throw std::runtime_error("failed to set vision ncnn input " + name);
        }
    }
    ncnn::Mat output;
    if (extractor.extract("out0", output) != 0) {
        throw std::runtime_error("failed to extract vision ncnn output");
    }
    return ncnn_mat_to_array_2d(output);
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("failed to open text file: " + path.string());
    }
    return std::string(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

void replace_all(
    std::string& value,
    const std::string& source,
    const std::string& replacement) {
    size_t position = 0;
    while ((position = value.find(source, position)) != std::string::npos) {
        value.replace(position, source.size(), replacement);
        position += replacement.size();
    }
}

std::string patch_vision_embedding_param(
    const std::filesystem::path& path,
    int sequence_length) {
    std::string text = read_text_file(path);
    replace_all(text, "0=768 1=1152", "0=768 1=" + std::to_string(sequence_length));
    replace_all(text, "7=1152 8=1152 9=768", "7=" + std::to_string(sequence_length) + " 8=1152 9=768");
    return text;
}

std::string patch_vision_layer_param(
    const std::filesystem::path& path,
    int sequence_length) {
    std::string text = read_text_file(path);
    replace_all(text, "0=72 1=16 2=1152", "0=72 1=16 2=" + std::to_string(sequence_length));
    replace_all(text, "0=1152 1=1152\n", "0=1152 1=" + std::to_string(sequence_length) + "\n");
    replace_all(text, "7=1152 ", "7=" + std::to_string(sequence_length) + " ");
    return text;
}

std::string patch_merger_param(
    const std::filesystem::path& path,
    int image_token_count) {
    std::string text = read_text_file(path);
    replace_all(text, "0=4608 1=288", "0=4608 1=" + std::to_string(image_token_count));
    replace_all(text, "7=288 ", "7=" + std::to_string(image_token_count) + " ");
    return text;
}

struct VisionLayout {
    std::vector<int> window_indices;
    std::vector<int> window_ends;
};

VisionLayout build_vision_layout(int grid_h, int grid_w) {
    constexpr int merge_size = 2;
    constexpr int merge_unit = 4;
    constexpr int merger_window_size = 8;
    const int llm_grid_h = grid_h / merge_size;
    const int llm_grid_w = grid_w / merge_size;
    const int pad_h =
        (merger_window_size - llm_grid_h % merger_window_size) %
        merger_window_size;
    const int pad_w =
        (merger_window_size - llm_grid_w % merger_window_size) %
        merger_window_size;
    const int padded_h = llm_grid_h + pad_h;
    const int padded_w = llm_grid_w + pad_w;
    const int windows_h = padded_h / merger_window_size;
    const int windows_w = padded_w / merger_window_size;

    VisionLayout layout;
    layout.window_indices.reserve(
        static_cast<size_t>(llm_grid_h) * llm_grid_w);
    int cumulative_tokens = 0;
    for (int window_row = 0; window_row < windows_h; ++window_row) {
        for (int window_column = 0; window_column < windows_w; ++window_column) {
            int groups_in_window = 0;
            for (int row = 0; row < merger_window_size; ++row) {
                for (int column = 0; column < merger_window_size; ++column) {
                    const int source_row = window_row * merger_window_size + row;
                    const int source_column =
                        window_column * merger_window_size + column;
                    if (source_row < llm_grid_h && source_column < llm_grid_w) {
                        layout.window_indices.push_back(
                            source_row * llm_grid_w + source_column);
                        ++groups_in_window;
                    }
                }
            }
            cumulative_tokens += groups_in_window * merge_unit;
            if (groups_in_window != 0) {
                layout.window_ends.push_back(cumulative_tokens);
            }
        }
    }
    if (cumulative_tokens != grid_h * grid_w) {
        throw std::runtime_error("dynamic vision layout token count mismatch");
    }
    return layout;
}

std::vector<int> build_vision_window_index(int grid_h, int grid_w) {
    return build_vision_layout(grid_h, grid_w).window_indices;
}

std::pair<Array, Array> build_vision_rope(
    int grid_h,
    int grid_w,
    const std::vector<int>& window_indices) {
    constexpr int rope_half_dim = 18;
    constexpr int rope_dim = 72;
    const int sequence_length = grid_h * grid_w;
    const int group_width = grid_w / 2;
    Array cosine{{sequence_length, rope_dim}, std::vector<float>(static_cast<size_t>(sequence_length) * rope_dim)};
    Array sine = cosine;
    for (int output_group = 0; output_group < sequence_length / 4; ++output_group) {
        const int source_group = window_indices[output_group];
        const int group_row = source_group / group_width;
        const int group_column = source_group % group_width;
        for (int unit = 0; unit < 4; ++unit) {
            const int h_position = group_row * 2 + unit / 2;
            const int w_position = group_column * 2 + unit % 2;
            const int token = output_group * 4 + unit;
            for (int axis = 0; axis < 2; ++axis) {
                const int position = axis == 0 ? h_position : w_position;
                for (int index = 0; index < rope_half_dim; ++index) {
                    const float exponent =
                        static_cast<float>(index * 2) / 36.0f;
                    const float inverse_frequency =
                        1.0f / std::pow(10000.0f, exponent);
                    const float frequency = inverse_frequency * position;
                    const float cos_value = std::cos(frequency);
                    const float sin_value = std::sin(frequency);
                    const int base_dimension = axis * rope_half_dim + index;
                    for (int copy = 0; copy < 2; ++copy) {
                        const int dimension = base_dimension + copy * 36;
                        cosine.data[static_cast<size_t>(token) * rope_dim + dimension] = cos_value;
                        sine.data[static_cast<size_t>(token) * rope_dim + dimension] = sin_value;
                    }
                }
            }
        }
    }
    return {std::move(cosine), std::move(sine)};
}

Array build_vision_attention_mask(
    int sequence_length,
    const std::vector<int>& window_ends,
    bool full_attention) {
    Array mask{
        {1, sequence_length, sequence_length},
        std::vector<float>(
            static_cast<size_t>(sequence_length) * sequence_length,
            full_attention ? 0.0f : -std::numeric_limits<float>::max()),
    };
    if (full_attention) {
        return mask;
    }
    int start = 0;
    for (int end : window_ends) {
        for (int row = start; row < end; ++row) {
            std::fill(
                mask.data.begin() + static_cast<size_t>(row) * sequence_length + start,
                mask.data.begin() + static_cast<size_t>(row) * sequence_length + end,
                0.0f);
        }
        start = end;
    }
    return mask;
}

Array reorder_vision_groups(
    const Array& input,
    const std::vector<int>& group_indices) {
    if (input.shape.size() != 2 || input.shape[0] % 4 != 0) {
        throw std::runtime_error("invalid vision tensor shape for reordering");
    }
    const int rows = input.shape[0];
    const int hidden = input.shape[1];
    const int groups = rows / 4;
    if (static_cast<int>(group_indices.size()) != groups) {
        throw std::runtime_error("vision group index count mismatch");
    }

    Array output{input.shape, std::vector<float>(input.data.size())};
    for (int output_group = 0; output_group < groups; ++output_group) {
        const int source_group = group_indices[output_group];
        for (int unit = 0; unit < 4; ++unit) {
            std::copy_n(
                input.data.begin() +
                    static_cast<size_t>(source_group * 4 + unit) * hidden,
                hidden,
                output.data.begin() +
                    static_cast<size_t>(output_group * 4 + unit) * hidden);
        }
    }
    return output;
}

Array reverse_vision_reorder(
    const Array& input,
    const std::vector<int>& window_indices) {
    std::vector<int> reverse(window_indices.size());
    std::iota(reverse.begin(), reverse.end(), 0);
    std::sort(reverse.begin(), reverse.end(), [&](int left, int right) {
        return window_indices[left] < window_indices[right];
    });
    return reorder_vision_groups(input, reverse);
}

class VisionPipeline {
public:
    explicit VisionPipeline(std::filesystem::path root)
        : root_(std::move(root)) {}

    Array encode(
        const Array& pixel_values,
        int grid_h,
        int grid_w) const {
        const int sequence_length = grid_h * grid_w;
        const VisionLayout layout = build_vision_layout(grid_h, grid_w);
        const std::vector<int>& window_indices = layout.window_indices;
        // The portable model package contains only ncnn weights. Generate
        // RoPE and attention masks for every image shape, including the
        // original 36x32 export shape, instead of depending on auxiliary
        // cos.npy/sin.npy/attention_mask.npy files from the export workspace.
        const bool frozen_shape = false;
        const auto embedding_param_path =
            root_ / "artifacts/vision_embedding/vision_embedding_wrapper.ncnn.param";
        const std::string embedding_param = frozen_shape
            ? std::string()
            : patch_vision_embedding_param(embedding_param_path, sequence_length);
        Array hidden = run_vision_ncnn(
            embedding_param_path,
            root_ / "artifacts/vision_embedding/vision_embedding_wrapper.ncnn.bin",
            {pixel_values},
            frozen_shape ? nullptr : &embedding_param);
        hidden = reorder_vision_groups(hidden, window_indices);

        Array dynamic_cos;
        Array dynamic_sin;
        if (!frozen_shape) {
            auto rope = build_vision_rope(grid_h, grid_w, window_indices);
            dynamic_cos = std::move(rope.first);
            dynamic_sin = std::move(rope.second);
        }

        for (int layer = 0; layer < 27; ++layer) {
            const auto layer_dir =
                root_ /
                ("artifacts/vision_layer" + std::to_string(layer) +
                 "_masked_core");
            const std::string model =
                "vision_layer" + std::to_string(layer) +
                "_masked_core_wrapper";
            const Array cos = frozen_shape
                ? load_npy_float((layer_dir / "cos.npy").string())
                : dynamic_cos;
            const Array sin = frozen_shape
                ? load_npy_float((layer_dir / "sin.npy").string())
                : dynamic_sin;
            const bool full_attention =
                layer == 7 || layer == 15 || layer == 23 || layer == 26;
            const Array mask = frozen_shape
                ? load_npy_float((layer_dir / "attention_mask.npy").string())
                : build_vision_attention_mask(
                      sequence_length,
                      layout.window_ends,
                      full_attention);
            const auto param_path = layer_dir / (model + ".ncnn.param");
            const std::string dynamic_param = frozen_shape
                ? std::string()
                : patch_vision_layer_param(param_path, sequence_length);
            hidden = run_vision_ncnn(
                param_path,
                layer_dir / (model + ".ncnn.bin"),
                {hidden, cos, sin, mask},
                frozen_shape ? nullptr : &dynamic_param);
        }

        hidden = reverse_vision_reorder(hidden, window_indices);
        return run_vision_ncnn(
            root_ /
                "artifacts/vision_post_layernorm/"
                "vision_post_layernorm_wrapper.ncnn.param",
            root_ /
                "artifacts/vision_post_layernorm/"
                "vision_post_layernorm_wrapper.ncnn.bin",
            {hidden});
    }

    Array merge(const Array& vision_hidden) const {
        const int image_token_count = vision_hidden.shape[0] / 4;
        const auto param_path = root_ / "models/youtu_merger.ncnn.param";
        const bool frozen_shape = image_token_count == 288;
        const std::string dynamic_param = frozen_shape
            ? std::string()
            : patch_merger_param(param_path, image_token_count);
        return run_vision_ncnn(
            param_path,
            root_ / "models/youtu_merger.ncnn.bin",
            {vision_hidden},
            frozen_shape ? nullptr : &dynamic_param);
    }

private:
    std::filesystem::path root_;
};

}  // namespace youtu_vl
