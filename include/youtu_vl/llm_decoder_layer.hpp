#pragma once

#include <net.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace youtu_vl {

constexpr int kNumHeads = 32;
constexpr int kQkNopeHeadDim = 128;
constexpr int kQkRopeHeadDim = 64;
constexpr int kQkHeadDim = 192;
constexpr int kValueHeadDim = 128;
constexpr int kHiddenSize = 2560;
constexpr int kAttentionOutputSize = kNumHeads * kValueHeadDim;
constexpr float kScaling = 0.072168782353401184f;

struct Array {
    std::vector<int> shape;
    std::vector<float> data;
};

struct Int64Array {
    std::vector<int> shape;
    std::vector<int64_t> data;
};

struct BridgeOutput {
    Array attention_result;
    Array new_key;
    Array new_value;
    Array key_cache;
    Array value_cache;
};

struct DecodeOutput {
    Array layer_output;
    Array attention_result;
    Array new_key;
    Array new_value;
    Array key_cache;
    Array value_cache;
};

inline uint16_t read_u16_le(std::ifstream& stream) {
    unsigned char bytes[2];
    stream.read(reinterpret_cast<char*>(bytes), 2);
    return static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
}

inline float half_to_float(uint16_t value);

inline uint16_t float_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000;
    const uint32_t exponent = (bits >> 23) & 0xff;
    const uint32_t mantissa = bits & 0x7fffff;

    if (exponent == 0xff) {
        return static_cast<uint16_t>(
            sign | (mantissa != 0 ? 0x7e00 : 0x7c00));
    }

    const int adjusted_exponent = static_cast<int>(exponent) - 127 + 15;
    if (adjusted_exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    if (adjusted_exponent <= 0) {
        if (adjusted_exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        uint32_t normalized = mantissa | 0x800000;
        const int shift = 14 - adjusted_exponent;
        uint32_t rounded = normalized >> shift;
        const uint32_t remainder = normalized & ((1u << shift) - 1);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1))) {
            ++rounded;
        }
        return static_cast<uint16_t>(sign | rounded);
    }

    uint32_t rounded_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fff;
    if (remainder > 0x1000 ||
        (remainder == 0x1000 && (rounded_mantissa & 1))) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400) {
            rounded_mantissa = 0;
            if (adjusted_exponent + 1 >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00);
            }
            return static_cast<uint16_t>(
                sign | ((adjusted_exponent + 1) << 10));
        }
    }
    return static_cast<uint16_t>(
        sign | (adjusted_exponent << 10) | rounded_mantissa);
}

inline float round_to_half(float value) {
    return half_to_float(float_to_half(value));
}

inline uint32_t read_u32_le(std::ifstream& stream) {
    unsigned char bytes[4];
    stream.read(reinterpret_cast<char*>(bytes), 4);
    return static_cast<uint32_t>(
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24));
}

inline void write_u16_le(std::ofstream& stream, uint16_t value) {
    const unsigned char bytes[2] = {
        static_cast<unsigned char>(value & 0xff),
        static_cast<unsigned char>((value >> 8) & 0xff),
    };
    stream.write(reinterpret_cast<const char*>(bytes), 2);
}

inline float half_to_float(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value & 0x8000)) << 16;
    uint32_t exponent = (value >> 10) & 0x1f;
    uint32_t mantissa = value & 0x03ff;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ff;
            bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000 | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float output;
    std::memcpy(&output, &bits, sizeof(float));
    return output;
}

inline std::vector<int> parse_shape(const std::string& header) {
    const size_t left = header.find('(');
    const size_t right = header.find(')', left);
    if (left == std::string::npos || right == std::string::npos) {
        throw std::runtime_error("failed to parse npy shape");
    }

    std::vector<int> shape;
    size_t position = left + 1;
    while (position < right) {
        while (position < right &&
               (header[position] == ' ' || header[position] == ',')) {
            ++position;
        }
        if (position >= right) {
            break;
        }
        size_t end = position;
        while (end < right && header[end] != ',' && header[end] != ' ') {
            ++end;
        }
        if (end > position) {
            shape.push_back(std::stoi(header.substr(position, end - position)));
        }
        position = end + 1;
    }
    if (shape.empty()) {
        throw std::runtime_error("empty npy shape");
    }
    return shape;
}

inline size_t numel(const std::vector<int>& shape) {
    size_t count = 1;
    for (int dimension : shape) {
        if (dimension < 0) {
            throw std::runtime_error("npy shape contains a negative dimension");
        }
        count *= static_cast<size_t>(dimension);
    }
    return count;
}

inline Array load_npy_float(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open npy: " + path);
    }

    char magic[6];
    stream.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("invalid npy magic: " + path);
    }

    unsigned char version[2];
    stream.read(reinterpret_cast<char*>(version), 2);
    uint32_t header_length = 0;
    if (version[0] == 1) {
        header_length = read_u16_le(stream);
    } else if (version[0] == 2 || version[0] == 3) {
        header_length = read_u32_le(stream);
    } else {
        throw std::runtime_error("unsupported npy version: " + path);
    }

    std::string header(header_length, '\0');
    stream.read(header.data(), header_length);
    if (header.find("False") == std::string::npos) {
        throw std::runtime_error("fortran-order npy is not supported: " + path);
    }
    const bool float32 =
        header.find("'descr': '<f4'") != std::string::npos ||
        header.find("\"descr\": \"<f4\"") != std::string::npos;
    const bool float16 =
        header.find("'descr': '<f2'") != std::string::npos ||
        header.find("\"descr\": \"<f2\"") != std::string::npos;
    if (!float32 && !float16) {
        throw std::runtime_error(
            "only little-endian float16/float32 npy is supported: " + path);
    }

    Array output;
    output.shape = parse_shape(header);
    output.data.resize(numel(output.shape));
    if (float32) {
        stream.read(
            reinterpret_cast<char*>(output.data.data()),
            static_cast<std::streamsize>(output.data.size() * sizeof(float)));
    } else {
        std::vector<uint16_t> half_data(output.data.size());
        stream.read(
            reinterpret_cast<char*>(half_data.data()),
            static_cast<std::streamsize>(half_data.size() * sizeof(uint16_t)));
        for (size_t index = 0; index < half_data.size(); ++index) {
            output.data[index] = half_to_float(half_data[index]);
        }
    }
    if (!stream) {
        throw std::runtime_error("failed to read npy data: " + path);
    }
    return output;
}

inline Int64Array load_npy_int64(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open npy: " + path);
    }

    char magic[6];
    stream.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("invalid npy magic: " + path);
    }
    unsigned char version[2];
    stream.read(reinterpret_cast<char*>(version), 2);
    uint32_t header_length = 0;
    if (version[0] == 1) {
        header_length = read_u16_le(stream);
    } else if (version[0] == 2 || version[0] == 3) {
        header_length = read_u32_le(stream);
    } else {
        throw std::runtime_error("unsupported npy version: " + path);
    }

    std::string header(header_length, '\0');
    stream.read(header.data(), header_length);
    if (header.find("False") == std::string::npos) {
        throw std::runtime_error("fortran-order npy is not supported: " + path);
    }
    const bool int64 =
        header.find("'descr': '<i8'") != std::string::npos ||
        header.find("\"descr\": \"<i8\"") != std::string::npos;
    if (!int64) {
        throw std::runtime_error("expected little-endian int64 npy: " + path);
    }

    Int64Array output;
    output.shape = parse_shape(header);
    output.data.resize(numel(output.shape));
    stream.read(
        reinterpret_cast<char*>(output.data.data()),
        static_cast<std::streamsize>(output.data.size() * sizeof(int64_t)));
    if (!stream) {
        throw std::runtime_error("failed to read npy data: " + path);
    }
    return output;
}

class NpyFloatRowReader {
public:
    explicit NpyFloatRowReader(const std::string& path)
        : stream_(path, std::ios::binary), path_(path) {
        if (!stream_) {
            throw std::runtime_error("failed to open npy: " + path);
        }
        char magic[6];
        stream_.read(magic, 6);
        if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            throw std::runtime_error("invalid npy magic: " + path);
        }
        unsigned char version[2];
        stream_.read(reinterpret_cast<char*>(version), 2);
        uint32_t header_length = 0;
        if (version[0] == 1) {
            header_length = read_u16_le(stream_);
        } else if (version[0] == 2 || version[0] == 3) {
            header_length = read_u32_le(stream_);
        } else {
            throw std::runtime_error("unsupported npy version: " + path);
        }
        std::string header(header_length, '\0');
        stream_.read(header.data(), header_length);
        shape_ = parse_shape(header);
        if (shape_.size() != 2) {
            throw std::runtime_error("row reader requires rank-2 npy: " + path);
        }
        float32_ = header.find("'descr': '<f4'") != std::string::npos;
        float16_ = header.find("'descr': '<f2'") != std::string::npos;
        if (!float32_ && !float16_) {
            throw std::runtime_error(
                "row reader supports float16/float32 only: " + path);
        }
        data_offset_ = stream_.tellg();
    }

    Array read_row(int row) {
        if (row < 0 || row >= shape_[0]) {
            throw std::runtime_error("npy row index is out of range");
        }
        const int columns = shape_[1];
        const size_t item_size = float32_ ? sizeof(float) : sizeof(uint16_t);
        const std::streamoff offset =
            data_offset_ +
            static_cast<std::streamoff>(
                static_cast<size_t>(row) * columns * item_size);
        stream_.clear();
        stream_.seekg(offset);
        Array output{{1, columns}, std::vector<float>(columns)};
        if (float32_) {
            stream_.read(
                reinterpret_cast<char*>(output.data.data()),
                static_cast<std::streamsize>(columns * sizeof(float)));
        } else {
            std::vector<uint16_t> values(columns);
            stream_.read(
                reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(columns * sizeof(uint16_t)));
            for (int index = 0; index < columns; ++index) {
                output.data[index] = half_to_float(values[index]);
            }
        }
        if (!stream_) {
            throw std::runtime_error("failed to read npy row: " + path_);
        }
        return output;
    }

    const std::vector<int>& shape() const {
        return shape_;
    }

private:
    std::ifstream stream_;
    std::string path_;
    std::vector<int> shape_;
    std::streamoff data_offset_ = 0;
    bool float32_ = false;
    bool float16_ = false;
};

inline void save_npy_float32(
    const std::string& path,
    const std::vector<int>& shape,
    const std::vector<float>& data) {
    if (data.size() != numel(shape)) {
        throw std::runtime_error("output data size does not match shape: " + path);
    }
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open output npy: " + path);
    }

    std::string shape_text = "(";
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            shape_text += ", ";
        }
        shape_text += std::to_string(shape[index]);
    }
    if (shape.size() == 1) {
        shape_text += ",";
    }
    shape_text += ")";
    std::string header =
        "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_text + ", }";
    constexpr size_t preamble = 10;
    size_t padding = 16 - ((preamble + header.size() + 1) % 16);
    if (padding == 16) {
        padding = 0;
    }
    header.append(padding, ' ');
    header.push_back('\n');
    if (header.size() > std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("npy header is too large");
    }

    stream.write("\x93NUMPY", 6);
    const unsigned char version[2] = {1, 0};
    stream.write(reinterpret_cast<const char*>(version), 2);
    write_u16_le(stream, static_cast<uint16_t>(header.size()));
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    stream.write(
        reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(float)));
}

inline void save_npy_int64(
    const std::string& path,
    const std::vector<int>& shape,
    const std::vector<int64_t>& data) {
    if (data.size() != numel(shape)) {
        throw std::runtime_error("output data size does not match shape: " + path);
    }
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open output npy: " + path);
    }
    std::string shape_text = "(";
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            shape_text += ", ";
        }
        shape_text += std::to_string(shape[index]);
    }
    if (shape.size() == 1) {
        shape_text += ",";
    }
    shape_text += ")";
    std::string header =
        "{'descr': '<i8', 'fortran_order': False, 'shape': " + shape_text + ", }";
    constexpr size_t preamble = 10;
    size_t padding = 16 - ((preamble + header.size() + 1) % 16);
    if (padding == 16) {
        padding = 0;
    }
    header.append(padding, ' ');
    header.push_back('\n');
    stream.write("\x93NUMPY", 6);
    const unsigned char version[2] = {1, 0};
    stream.write(reinterpret_cast<const char*>(version), 2);
    write_u16_le(stream, static_cast<uint16_t>(header.size()));
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    stream.write(
        reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(int64_t)));
}

inline void flatten_leading_unit_dimensions(Array& array, int expected_last_dimension) {
    if (array.shape.empty() || array.shape.back() != expected_last_dimension) {
        throw std::runtime_error("unexpected last tensor dimension");
    }
    const size_t rows = array.data.size() / expected_last_dimension;
    array.shape = {static_cast<int>(rows), expected_last_dimension};
}

inline ncnn::Mat to_ncnn_mat_2d(const Array& array) {
    if (array.shape.size() != 2) {
        throw std::runtime_error("ncnn input must be rank 2");
    }
    return ncnn::Mat(
               array.shape[1],
               array.shape[0],
               const_cast<float*>(array.data.data()))
        .clone();
}

inline Array ncnn_mat_to_array_2d(const ncnn::Mat& mat) {
    if (mat.dims == 1) {
        Array output;
        output.shape = {1, mat.w};
        output.data.resize(static_cast<size_t>(mat.w));
        std::memcpy(output.data.data(), mat.data, output.data.size() * sizeof(float));
        return output;
    }
    if (mat.dims != 2) {
        throw std::runtime_error(
            "expected rank-1/rank-2 ncnn output, got dims=" +
            std::to_string(mat.dims));
    }
    Array output;
    output.shape = {mat.h, mat.w};
    output.data.resize(static_cast<size_t>(mat.h) * mat.w);
    std::memcpy(output.data.data(), mat.data, output.data.size() * sizeof(float));
    return output;
}

struct NcnnRuntimeOptions {
    int num_threads = 1;
    bool use_vulkan = false;
};

class NcnnModule {
public:
    NcnnModule(
        std::filesystem::path model_dir,
        std::string model_name,
        NcnnRuntimeOptions options = {})
        : model_dir_(std::move(model_dir)),
          model_name_(std::move(model_name)),
          options_(options) {
        if (options_.num_threads < 1) {
            throw std::invalid_argument("ncnn num_threads must be positive");
        }
    }

    NcnnModule(const NcnnModule&) = delete;
    NcnnModule& operator=(const NcnnModule&) = delete;
    NcnnModule(NcnnModule&&) noexcept = default;
    NcnnModule& operator=(NcnnModule&&) noexcept = default;

    void load() {
        if (!net_) {
            net_ = load_net();
        }
    }

    void unload() {
        net_.reset();
    }

    bool loaded() const {
        return static_cast<bool>(net_);
    }

    size_t model_load_count() const {
        return model_load_count_;
    }

    size_t run_count() const {
        return run_count_;
    }

    std::vector<Array> run(
        const std::vector<Array>& inputs,
        int output_count) const {
        ++run_count_;
        if (net_) {
            return run_with_net(*net_, inputs, output_count);
        }
        std::unique_ptr<ncnn::Net> temporary = load_net();
        return run_with_net(*temporary, inputs, output_count);
    }

private:
    std::unique_ptr<ncnn::Net> load_net() const {
        auto net = std::make_unique<ncnn::Net>();
        net->opt.use_vulkan_compute = options_.use_vulkan;
        net->opt.num_threads = options_.num_threads;
        const std::string param =
            (model_dir_ / (model_name_ + ".ncnn.param")).string();
        const std::string binary =
            (model_dir_ / (model_name_ + ".ncnn.bin")).string();
        if (net->load_param(param.c_str()) != 0) {
            throw std::runtime_error("failed to load ncnn param: " + param);
        }
        if (net->load_model(binary.c_str()) != 0) {
            throw std::runtime_error("failed to load ncnn model: " + binary);
        }
        ++model_load_count_;
        return net;
    }

    std::vector<Array> run_with_net(
        const ncnn::Net& net,
        const std::vector<Array>& inputs,
        int output_count) const {
        ncnn::Extractor extractor = net.create_extractor();
        std::vector<ncnn::Mat> input_mats;
        input_mats.reserve(inputs.size());
        for (size_t index = 0; index < inputs.size(); ++index) {
            input_mats.push_back(to_ncnn_mat_2d(inputs[index]));
            const std::string input_name = "in" + std::to_string(index);
            if (extractor.input(input_name.c_str(), input_mats.back()) != 0) {
                throw std::runtime_error("failed to set ncnn input " + input_name);
            }
        }

        std::vector<Array> outputs;
        outputs.reserve(output_count);
        for (int index = 0; index < output_count; ++index) {
            ncnn::Mat output_mat;
            const std::string output_name = "out" + std::to_string(index);
            if (extractor.extract(output_name.c_str(), output_mat) != 0) {
                throw std::runtime_error(
                    "failed to extract ncnn output " + output_name);
            }
            outputs.push_back(ncnn_mat_to_array_2d(output_mat));
        }
        return outputs;
    }

    std::filesystem::path model_dir_;
    std::string model_name_;
    NcnnRuntimeOptions options_;
    std::unique_ptr<ncnn::Net> net_;
    mutable size_t model_load_count_ = 0;
    mutable size_t run_count_ = 0;
};

inline std::vector<Array> run_ncnn_part(
    const std::filesystem::path& part_dir,
    const std::string& model_name,
    const std::vector<Array>& inputs,
    int output_count) {
    return NcnnModule(part_dir, model_name).run(inputs, output_count);
}

inline size_t offset3(int first, int second, int third, int dim2, int dim3) {
    return (static_cast<size_t>(first) * dim2 + second) * dim3 + third;
}

inline float interleaved_value(const float* source, int dimension) {
    if (dimension < kQkRopeHeadDim / 2) {
        return source[dimension * 2];
    }
    return source[(dimension - kQkRopeHeadDim / 2) * 2 + 1];
}

inline BridgeOutput decode_bridge(
    const Array& q_flat,
    const Array& kv_b_flat,
    const Array& k_rot_flat,
    const Array& cos,
    const Array& sin,
    const Array& past_key_input,
    const Array& past_value_input) {
    if (q_flat.shape.size() != 2 || q_flat.shape[1] != kNumHeads * kQkHeadDim) {
        throw std::runtime_error("q_flat must have shape [seq, 6144]");
    }
    const int sequence_length = q_flat.shape[0];
    if (kv_b_flat.shape !=
        std::vector<int>{sequence_length, kNumHeads * (kQkNopeHeadDim + kValueHeadDim)}) {
        throw std::runtime_error("kv_b_flat must have shape [seq, 8192]");
    }
    if (k_rot_flat.shape != std::vector<int>{sequence_length, kQkRopeHeadDim}) {
        throw std::runtime_error("k_rot must have shape [seq, 64]");
    }
    if (cos.data.size() != static_cast<size_t>(sequence_length * kQkRopeHeadDim) ||
        sin.data.size() != static_cast<size_t>(sequence_length * kQkRopeHeadDim)) {
        throw std::runtime_error("RoPE cos/sin element count does not match sequence");
    }

    Array past_key = past_key_input;
    Array past_value = past_value_input;
    if (past_key.shape.size() == 4 && past_key.shape[0] == 1) {
        past_key.shape.erase(past_key.shape.begin());
    }
    if (past_value.shape.size() == 4 && past_value.shape[0] == 1) {
        past_value.shape.erase(past_value.shape.begin());
    }
    if (past_key.shape.size() != 3 || past_key.shape[0] != kNumHeads ||
        past_key.shape[2] != kQkHeadDim) {
        throw std::runtime_error("past_key must have shape [32, past, 192]");
    }
    if (past_value.shape.size() != 3 || past_value.shape[0] != kNumHeads ||
        past_value.shape[2] != kValueHeadDim) {
        throw std::runtime_error("past_value must have shape [32, past, 128]");
    }
    const int past_length = past_key.shape[1];
    if (past_value.shape[1] != past_length) {
        throw std::runtime_error("past Key/Value lengths differ");
    }
    const int total_length = past_length + sequence_length;

    Array query_states{
        {kNumHeads, sequence_length, kQkHeadDim},
        std::vector<float>(
            static_cast<size_t>(kNumHeads) * sequence_length * kQkHeadDim)};
    Array new_key{
        {kNumHeads, sequence_length, kQkHeadDim},
        std::vector<float>(
            static_cast<size_t>(kNumHeads) * sequence_length * kQkHeadDim)};
    Array new_value{
        {kNumHeads, sequence_length, kValueHeadDim},
        std::vector<float>(
            static_cast<size_t>(kNumHeads) * sequence_length * kValueHeadDim)};

    for (int token = 0; token < sequence_length; ++token) {
        const float* key_rope_source =
            k_rot_flat.data.data() + static_cast<size_t>(token) * kQkRopeHeadDim;
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
                    head, token, dimension, sequence_length, kQkHeadDim)] =
                    q_source[dimension];
                new_key.data[offset3(
                    head, token, dimension, sequence_length, kQkHeadDim)] =
                    kv_source[dimension];
                new_value.data[offset3(
                    head, token, dimension, sequence_length, kValueHeadDim)] =
                    kv_source[kQkNopeHeadDim + dimension];
            }
            for (int dimension = 0; dimension < kQkRopeHeadDim; ++dimension) {
                const float q_interleaved =
                    interleaved_value(q_source + kQkNopeHeadDim, dimension);
                const float k_interleaved =
                    interleaved_value(key_rope_source, dimension);
                const int rotated_dimension =
                    dimension < kQkRopeHeadDim / 2
                        ? dimension + kQkRopeHeadDim / 2
                        : dimension - kQkRopeHeadDim / 2;
                const float q_rotated =
                    dimension < kQkRopeHeadDim / 2
                        ? -interleaved_value(
                              q_source + kQkNopeHeadDim, rotated_dimension)
                        : interleaved_value(
                              q_source + kQkNopeHeadDim, rotated_dimension);
                const float k_rotated =
                    dimension < kQkRopeHeadDim / 2
                        ? -interleaved_value(key_rope_source, rotated_dimension)
                        : interleaved_value(key_rope_source, rotated_dimension);
                const float rope_cos =
                    cos.data[static_cast<size_t>(token) * kQkRopeHeadDim + dimension];
                const float rope_sin =
                    sin.data[static_cast<size_t>(token) * kQkRopeHeadDim + dimension];
                query_states.data[offset3(
                    head,
                    token,
                    kQkNopeHeadDim + dimension,
                    sequence_length,
                    kQkHeadDim)] =
                    q_interleaved * rope_cos + q_rotated * rope_sin;
                new_key.data[offset3(
                    head,
                    token,
                    kQkNopeHeadDim + dimension,
                    sequence_length,
                    kQkHeadDim)] =
                    k_interleaved * rope_cos + k_rotated * rope_sin;
            }
        }
    }

    Array attention_result{
        {sequence_length, kAttentionOutputSize},
        std::vector<float>(
            static_cast<size_t>(sequence_length) * kAttentionOutputSize)};
    std::vector<float> scores(total_length);
    for (int head = 0; head < kNumHeads; ++head) {
        for (int token = 0; token < sequence_length; ++token) {
            const int allowed_length = past_length + token + 1;
            float maximum = -std::numeric_limits<float>::infinity();
            for (int key_index = 0; key_index < allowed_length; ++key_index) {
                float dot = 0.0f;
                for (int dimension = 0; dimension < kQkHeadDim; ++dimension) {
                    const float query = query_states.data[offset3(
                        head, token, dimension, sequence_length, kQkHeadDim)];
                    const float key =
                        key_index < past_length
                            ? past_key.data[offset3(
                                  head,
                                  key_index,
                                  dimension,
                                  past_length,
                                  kQkHeadDim)]
                            : new_key.data[offset3(
                                  head,
                                  key_index - past_length,
                                  dimension,
                                  sequence_length,
                                  kQkHeadDim)];
                    dot += query * key;
                }
                scores[key_index] = dot * kScaling;
                maximum = std::max(maximum, scores[key_index]);
            }

            float denominator = 0.0f;
            for (int key_index = 0; key_index < allowed_length; ++key_index) {
                scores[key_index] = std::exp(scores[key_index] - maximum);
                denominator += scores[key_index];
            }
            for (int key_index = 0; key_index < allowed_length; ++key_index) {
                scores[key_index] /= denominator;
            }

            for (int dimension = 0; dimension < kValueHeadDim; ++dimension) {
                float output = 0.0f;
                for (int key_index = 0; key_index < allowed_length; ++key_index) {
                    const float value =
                        key_index < past_length
                            ? past_value.data[offset3(
                                  head,
                                  key_index,
                                  dimension,
                                  past_length,
                                  kValueHeadDim)]
                            : new_value.data[offset3(
                                  head,
                                  key_index - past_length,
                                  dimension,
                                  sequence_length,
                                  kValueHeadDim)];
                    output += scores[key_index] * value;
                }
                attention_result.data[
                    static_cast<size_t>(token) * kAttentionOutputSize +
                    head * kValueHeadDim + dimension] = output;
            }
        }
    }

    Array key_cache{
        {kNumHeads, total_length, kQkHeadDim},
        std::vector<float>(
            static_cast<size_t>(kNumHeads) * total_length * kQkHeadDim)};
    Array value_cache{
        {kNumHeads, total_length, kValueHeadDim},
        std::vector<float>(
            static_cast<size_t>(kNumHeads) * total_length * kValueHeadDim)};
    for (int head = 0; head < kNumHeads; ++head) {
        float* key_destination =
            key_cache.data.data() +
            static_cast<size_t>(head) * total_length * kQkHeadDim;
        float* value_destination =
            value_cache.data.data() +
            static_cast<size_t>(head) * total_length * kValueHeadDim;
        const float* past_key_source =
            past_key.data.data() +
            static_cast<size_t>(head) * past_length * kQkHeadDim;
        const float* past_value_source =
            past_value.data.data() +
            static_cast<size_t>(head) * past_length * kValueHeadDim;
        const float* new_key_source =
            new_key.data.data() +
            static_cast<size_t>(head) * sequence_length * kQkHeadDim;
        const float* new_value_source =
            new_value.data.data() +
            static_cast<size_t>(head) * sequence_length * kValueHeadDim;
        std::copy(
            past_key_source,
            past_key_source + static_cast<size_t>(past_length) * kQkHeadDim,
            key_destination);
        std::copy(
            new_key_source,
            new_key_source + static_cast<size_t>(sequence_length) * kQkHeadDim,
            key_destination + static_cast<size_t>(past_length) * kQkHeadDim);
        std::copy(
            past_value_source,
            past_value_source + static_cast<size_t>(past_length) * kValueHeadDim,
            value_destination);
        std::copy(
            new_value_source,
            new_value_source + static_cast<size_t>(sequence_length) * kValueHeadDim,
            value_destination + static_cast<size_t>(past_length) * kValueHeadDim);
    }

    return {
        std::move(attention_result),
        std::move(new_key),
        std::move(new_value),
        std::move(key_cache),
        std::move(value_cache),
    };
}

inline void require_shape(
    const Array& array,
    const std::vector<int>& expected,
    const std::string& name) {
    if (array.shape != expected) {
        throw std::runtime_error(name + " has an unexpected shape");
    }
}

class DecoderLayer {
public:
    explicit DecoderLayer(
        std::filesystem::path layer_dir,
        NcnnRuntimeOptions options = {},
        bool load_immediately = false)
        : layer_dir_(std::move(layer_dir)),
          part_a_(
              layer_dir_ / "part_a_attention_input",
              "part_a_attention_input",
              options),
          part_b_(
              layer_dir_ / "part_b_attention_output",
              "part_b_attention_output",
              options),
          part_c_(layer_dir_ / "part_c_mlp", "part_c_mlp", options) {
        if (load_immediately) {
            load();
        }
    }

    void load() {
        part_a_.load();
        part_b_.load();
        part_c_.load();
    }

    bool loaded() const {
        return part_a_.loaded() && part_b_.loaded() && part_c_.loaded();
    }

    size_t model_load_count() const {
        return part_a_.model_load_count() + part_b_.model_load_count() +
            part_c_.model_load_count();
    }

    std::vector<Array> attention_input(const Array& hidden_states) const {
        return part_a_.run({hidden_states}, 3);
    }

    std::vector<Array> attention_output(
        const Array& hidden_states,
        const Array& attention_result) const {
        return part_b_.run({hidden_states, attention_result}, 1);
    }

    std::vector<Array> mlp(const Array& hidden_states) const {
        return part_c_.run({hidden_states}, 1);
    }

    DecodeOutput decode(
        const Array& hidden_states,
        const Array& cos,
        const Array& sin,
        const Array& past_key,
        const Array& past_value) const {
        const auto part_a = attention_input(hidden_states);
        require_shape(
            part_a[0],
            {hidden_states.shape[0], kNumHeads * kQkHeadDim},
            "q_flat");
        require_shape(
            part_a[1],
            {
                hidden_states.shape[0],
                kNumHeads * (kQkNopeHeadDim + kValueHeadDim),
            },
            "kv_b_flat");
        require_shape(
            part_a[2],
            {hidden_states.shape[0], kQkRopeHeadDim},
            "k_rot");

        BridgeOutput bridge = decode_bridge(
            part_a[0], part_a[1], part_a[2], cos, sin, past_key, past_value);
        const auto part_b = attention_output(hidden_states, bridge.attention_result);
        const auto part_c = mlp(part_b[0]);

        require_shape(
            part_c[0],
            {hidden_states.shape[0], kHiddenSize},
            "layer_output");
        return {
            part_c[0],
            std::move(bridge.attention_result),
            std::move(bridge.new_key),
            std::move(bridge.new_value),
            std::move(bridge.key_cache),
            std::move(bridge.value_cache),
        };
    }

private:
    std::filesystem::path layer_dir_;
    NcnnModule part_a_;
    NcnnModule part_b_;
    NcnnModule part_c_;
};

inline void save_decode_output(
    const std::filesystem::path& output_dir,
    const DecodeOutput& output) {
    std::filesystem::create_directories(output_dir);
    save_npy_float32(
        (output_dir / "layer_output.npy").string(),
        output.layer_output.shape,
        output.layer_output.data);
    save_npy_float32(
        (output_dir / "new_key.npy").string(),
        output.new_key.shape,
        output.new_key.data);
    save_npy_float32(
        (output_dir / "new_value.npy").string(),
        output.new_value.shape,
        output.new_value.data);
    save_npy_float32(
        (output_dir / "attention_result.npy").string(),
        output.attention_result.shape,
        output.attention_result.data);
    save_npy_float32(
        (output_dir / "updated_key_cache.npy").string(),
        output.key_cache.shape,
        output.key_cache.data);
    save_npy_float32(
        (output_dir / "updated_value_cache.npy").string(),
        output.value_cache.shape,
        output.value_cache.data);
}

}  // namespace youtu_vl
