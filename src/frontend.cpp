#include "youtu_vl/frontend.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_SIMD
#define STBI_FAILURE_USERMSG
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include "third_party_stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <jpeglib.h>

namespace youtu_vl {
namespace {


struct DecodedImage {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgb;
};

struct JpegErrorManager {
    jpeg_error_mgr base;
    std::jmp_buf jump;
    char message[JMSG_LENGTH_MAX]{};
};

void jpeg_error_exit(j_common_ptr common) {
    auto* error = reinterpret_cast<JpegErrorManager*>(common->err);
    (*common->err->format_message)(common, error->message);
    std::longjmp(error->jump, 1);
}

DecodedImage decode_jpeg_rgb(const std::filesystem::path& path) {
    FILE* file = std::fopen(path.string().c_str(), "rb");
    if (file == nullptr) {
        throw std::runtime_error("failed to open JPEG: " + path.string());
    }
    jpeg_decompress_struct decoder{};
    JpegErrorManager error{};
    decoder.err = jpeg_std_error(&error.base);
    error.base.error_exit = jpeg_error_exit;
    if (setjmp(error.jump) != 0) {
        jpeg_destroy_decompress(&decoder);
        std::fclose(file);
        throw std::runtime_error(
            "failed to decode JPEG " + path.string() + ": " + error.message);
    }
    jpeg_create_decompress(&decoder);
    jpeg_stdio_src(&decoder, file);
    jpeg_read_header(&decoder, TRUE);
    decoder.out_color_space = JCS_RGB;
    jpeg_start_decompress(&decoder);
    DecodedImage output;
    output.width = static_cast<int>(decoder.output_width);
    output.height = static_cast<int>(decoder.output_height);
    const int channels = static_cast<int>(decoder.output_components);
    if (channels != 3) {
        jpeg_destroy_decompress(&decoder);
        std::fclose(file);
        throw std::runtime_error("JPEG decoder did not produce RGB");
    }
    output.rgb.resize(
        static_cast<size_t>(output.width) * output.height * channels);
    while (decoder.output_scanline < decoder.output_height) {
        JSAMPROW row =
            output.rgb.data() +
            static_cast<size_t>(decoder.output_scanline) *
                output.width * channels;
        jpeg_read_scanlines(&decoder, &row, 1);
    }
    jpeg_finish_decompress(&decoder);
    jpeg_destroy_decompress(&decoder);
    std::fclose(file);
    return output;
}

DecodedImage decode_image_rgb(const std::filesystem::path& path) {
    std::ifstream probe(path, std::ios::binary);
    unsigned char signature[2]{};
    probe.read(reinterpret_cast<char*>(signature), 2);
    if (!probe) {
        throw std::runtime_error("failed to read image: " + path.string());
    }
    if (signature[0] == 0xff && signature[1] == 0xd8) {
        return decode_jpeg_rgb(path);
    }
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* decoded = stbi_load(
        path.string().c_str(), &width, &height, &channels, 3);
    if (decoded == nullptr) {
        const char* reason = stbi_failure_reason();
        throw std::runtime_error(
            "failed to decode image " + path.string() +
            (reason == nullptr ? std::string() : ": " + std::string(reason)));
    }
    DecodedImage output;
    output.width = width;
    output.height = height;
    output.rgb.assign(
        decoded,
        decoded + static_cast<size_t>(width) * height * 3);
    stbi_image_free(decoded);
    return output;
}

uint32_t read_u32(std::ifstream& stream) {
    unsigned char bytes[4]{};
    stream.read(reinterpret_cast<char*>(bytes), 4);
    if (!stream) {
        throw std::runtime_error("truncated tokenizer binary");
    }
    return static_cast<uint32_t>(
        bytes[0] |
        (static_cast<uint32_t>(bytes[1]) << 8) |
        (static_cast<uint32_t>(bytes[2]) << 16) |
        (static_cast<uint32_t>(bytes[3]) << 24));
}

std::string read_string(std::ifstream& stream) {
    const uint32_t length = read_u32(stream);
    std::string value(length, '\0');
    stream.read(value.data(), static_cast<std::streamsize>(length));
    if (!stream) {
        throw std::runtime_error("truncated tokenizer string");
    }
    return value;
}

std::string pair_key(const std::string& left, const std::string& right) {
    std::string key;
    key.reserve(sizeof(uint32_t) + left.size() + right.size());
    const uint32_t left_length = static_cast<uint32_t>(left.size());
    key.push_back(static_cast<char>(left_length & 0xff));
    key.push_back(static_cast<char>((left_length >> 8) & 0xff));
    key.push_back(static_cast<char>((left_length >> 16) & 0xff));
    key.push_back(static_cast<char>((left_length >> 24) & 0xff));
    key.append(left);
    key.append(right);
    return key;
}

int scaled_dimension(double scale, int size) {
    constexpr int merged_patch = 32;
    return std::max(
        merged_patch,
        static_cast<int>(
            std::ceil(size * scale / merged_patch) * merged_patch));
}

std::pair<int, int> target_image_size(
    int height,
    int width,
    int max_image_patches) {
    if (height <= 0 || width <= 0 || max_image_patches <= 0) {
        throw std::invalid_argument("invalid image or max_image_patches size");
    }
    double scale = 1.0;
    for (int step = 0; step <= 50; ++step) {
        const int target_height = scaled_dimension(scale, height);
        const int target_width = scaled_dimension(scale, width);
        const int patches = (target_height / 16) * (target_width / 16);
        if (patches <= max_image_patches) {
            return {target_height, target_width};
        }
        scale -= 0.02;
    }
    throw std::runtime_error("failed to fit image within max_image_patches");
}

struct AxisFilter {
    std::vector<int> starts;
    std::vector<int> sizes;
    std::vector<int16_t> weights;
    int max_filter_size = 0;
    unsigned int weights_precision = 0;
};

AxisFilter build_antialias_linear_filter(int input_size, int output_size) {
    if (input_size <= 0 || output_size <= 0) {
        throw std::invalid_argument("invalid resize axis size");
    }
    const double scale =
        static_cast<double>(input_size) / output_size;
    const double support = scale >= 1.0 ? scale : 1.0;
    const int max_filter_size =
        static_cast<int>(std::ceil(support)) * 2 + 1;

    AxisFilter filter;
    filter.starts.resize(output_size);
    filter.sizes.resize(output_size);
    filter.max_filter_size = max_filter_size;
    std::vector<double> floating_weights(
        static_cast<size_t>(output_size) * max_filter_size, 0.0);
    double maximum_weight = 0.0;
    const double inverse_scale = scale >= 1.0 ? 1.0 / scale : 1.0;

    for (int output_index = 0; output_index < output_size; ++output_index) {
        const double center = scale * (output_index + 0.5);
        int start = std::max(
            static_cast<int>(center - support + 0.5), 0);
        int size =
            std::min(
                static_cast<int>(center + support + 0.5), input_size) -
            start;
        size = std::clamp(size, 0, max_filter_size);
        filter.starts[output_index] = start;
        filter.sizes[output_index] = size;

        double total_weight = 0.0;
        double* weights =
            floating_weights.data() +
            static_cast<size_t>(output_index) * max_filter_size;
        for (int index = 0; index < size; ++index) {
            const double distance =
                std::abs(
                    (index + start - center + 0.5) * inverse_scale);
            const double weight = distance < 1.0 ? 1.0 - distance : 0.0;
            weights[index] = weight;
            total_weight += weight;
        }
        if (total_weight != 0.0) {
            for (int index = 0; index < size; ++index) {
                weights[index] /= total_weight;
                maximum_weight = std::max(maximum_weight, weights[index]);
            }
        }
    }

    unsigned int precision = 0;
    for (; precision < 22; ++precision) {
        const int next_value = static_cast<int>(
            0.5 + maximum_weight * (1 << (precision + 1)));
        if (next_value >= (1 << 15)) {
            break;
        }
    }
    if (precision == 0) {
        throw std::runtime_error("invalid resize weight precision");
    }
    filter.weights_precision = precision;
    filter.weights.resize(
        static_cast<size_t>(output_size) * max_filter_size, 0);
    const int scale_factor = 1 << precision;
    for (size_t index = 0; index < floating_weights.size(); ++index) {
        const double value = floating_weights[index] * scale_factor;
        filter.weights[index] = static_cast<int16_t>(
            value < 0.0 ? static_cast<int>(value - 0.5)
                        : static_cast<int>(value + 0.5));
    }
    return filter;
}

unsigned char apply_uint8_filter(
    const unsigned char* source,
    int source_stride,
    int start,
    int size,
    const int16_t* weights,
    unsigned int precision) {
    int value = 1 << (precision - 1);
    for (int index = 0; index < size; ++index) {
        value +=
            static_cast<int>(source[(start + index) * source_stride]) *
            weights[index];
    }
    return static_cast<unsigned char>(
        std::clamp(value >> precision, 0, 255));
}

std::vector<unsigned char> resize_bilinear_antialias_u8(
    const unsigned char* source,
    int source_width,
    int source_height,
    int target_width,
    int target_height) {
    constexpr int channels = 3;
    if (source_width <= 0 || source_height <= 0 ||
        target_width <= 0 || target_height <= 0) {
        throw std::invalid_argument("invalid image resize dimensions");
    }

    std::vector<unsigned char> horizontal;
    const unsigned char* vertical_source = source;
    if (source_width != target_width) {
        const AxisFilter filter =
            build_antialias_linear_filter(source_width, target_width);
        horizontal.resize(
            static_cast<size_t>(source_height) * target_width * channels);
        for (int y = 0; y < source_height; ++y) {
            for (int x = 0; x < target_width; ++x) {
                const int start = filter.starts[x];
                const int size = filter.sizes[x];
                const int16_t* weights =
                    filter.weights.data() +
                    static_cast<size_t>(x) * filter.max_filter_size;
                for (int channel = 0; channel < channels; ++channel) {
                    const unsigned char* row_channel =
                        source +
                        (static_cast<size_t>(y) * source_width * channels) +
                        channel;
                    horizontal[
                        (static_cast<size_t>(y) * target_width + x) *
                            channels +
                        channel] = apply_uint8_filter(
                            row_channel,
                            channels,
                            start,
                            size,
                            weights,
                            filter.weights_precision);
                }
            }
        }
        vertical_source = horizontal.data();
    }

    if (source_height == target_height) {
        if (source_width == target_width) {
            return std::vector<unsigned char>(
                source,
                source +
                    static_cast<size_t>(source_width) * source_height *
                        channels);
        }
        return horizontal;
    }

    const AxisFilter filter =
        build_antialias_linear_filter(source_height, target_height);
    std::vector<unsigned char> target(
        static_cast<size_t>(target_width) * target_height * channels);
    const int row_stride = target_width * channels;
    for (int y = 0; y < target_height; ++y) {
        const int start = filter.starts[y];
        const int size = filter.sizes[y];
        const int16_t* weights =
            filter.weights.data() +
            static_cast<size_t>(y) * filter.max_filter_size;
        for (int x = 0; x < target_width; ++x) {
            for (int channel = 0; channel < channels; ++channel) {
                const unsigned char* column =
                    vertical_source +
                    static_cast<size_t>(x) * channels + channel;
                target[
                    (static_cast<size_t>(y) * target_width + x) *
                        channels +
                    channel] = apply_uint8_filter(
                        column,
                        row_stride,
                        start,
                        size,
                        weights,
                        filter.weights_precision);
            }
        }
    }
    return target;
}

void write_npy_int32(
    const std::filesystem::path& path,
    const std::vector<int>& shape,
    const std::vector<int32_t>& data) {
    if (numel(shape) != data.size()) {
        throw std::runtime_error("int32 npy shape does not match data");
    }
    std::ostringstream shape_text;
    shape_text << "(";
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            shape_text << ", ";
        }
        shape_text << shape[index];
    }
    if (shape.size() == 1) {
        shape_text << ",";
    }
    shape_text << ")";
    std::string header =
        "{'descr': '<i4', 'fortran_order': False, 'shape': " +
        shape_text.str() + ", }";
    const size_t preamble = 10;
    const size_t padding =
        (64 - ((preamble + header.size() + 1) % 64)) % 64;
    header.append(padding, ' ');
    header.push_back('\n');
    if (header.size() > std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("npy header is too long");
    }
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to create npy: " + path.string());
    }
    stream.write("\x93NUMPY", 6);
    const unsigned char version[2] = {1, 0};
    stream.write(reinterpret_cast<const char*>(version), 2);
    const uint16_t length = static_cast<uint16_t>(header.size());
    const unsigned char length_bytes[2] = {
        static_cast<unsigned char>(length & 0xff),
        static_cast<unsigned char>((length >> 8) & 0xff),
    };
    stream.write(reinterpret_cast<const char*>(length_bytes), 2);
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    stream.write(
        reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(int32_t)));
    if (!stream) {
        throw std::runtime_error("failed to write npy: " + path.string());
    }
}

void append_encoded(
    std::vector<int64_t>& output,
    const ByteBpeTokenizer& tokenizer,
    const std::string& text) {
    const auto encoded = tokenizer.encode(text);
    output.insert(output.end(), encoded.begin(), encoded.end());
}

}  // namespace

ByteBpeTokenizer::ByteBpeTokenizer(
    const std::filesystem::path& binary_path) {
    std::ifstream stream(binary_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "failed to open tokenizer binary: " + binary_path.string());
    }
    char magic[8]{};
    stream.read(magic, 8);
    if (std::memcmp(magic, "YVLTOK2", 8) != 0) {
        throw std::runtime_error("unsupported tokenizer binary format");
    }
    const uint32_t base_count = read_u32(stream);
    const uint32_t merge_count = read_u32(stream);
    const uint32_t added_count = read_u32(stream);
    base_id_to_bytes_.reserve(base_count);
    bytes_to_base_id_.reserve(base_count * 2);
    for (uint32_t id = 0; id < base_count; ++id) {
        std::string bytes = read_string(stream);
        bytes_to_base_id_.emplace(bytes, id);
        base_id_to_bytes_.push_back(std::move(bytes));
    }
    merge_ranks_.reserve(merge_count * 2);
    for (uint32_t rank = 0; rank < merge_count; ++rank) {
        std::string left = read_string(stream);
        std::string right = read_string(stream);
        merge_ranks_.emplace(pair_key(left, right), rank);
    }
    added_tokens_.reserve(added_count * 2);
    for (uint32_t index = 0; index < added_count; ++index) {
        const int64_t id = read_u32(stream);
        const int flag = stream.get();
        if (flag == std::char_traits<char>::eof()) {
            throw std::runtime_error("truncated tokenizer added token");
        }
        added_tokens_.emplace(
            id,
            AddedToken{read_string(stream), flag != 0});
    }
}

std::vector<int64_t> ByteBpeTokenizer::encode_bpe_piece(
    const std::string& text) const {
    std::vector<std::string> pieces;
    pieces.reserve(text.size());
    for (unsigned char byte : text) {
        pieces.emplace_back(1, static_cast<char>(byte));
    }
    while (pieces.size() >= 2) {
        uint32_t best_rank = std::numeric_limits<uint32_t>::max();
        size_t best_position = pieces.size();
        for (size_t position = 0; position + 1 < pieces.size(); ++position) {
            const auto found = merge_ranks_.find(
                pair_key(pieces[position], pieces[position + 1]));
            if (found != merge_ranks_.end() && found->second < best_rank) {
                best_rank = found->second;
                best_position = position;
            }
        }
        if (best_position == pieces.size()) {
            break;
        }
        const std::string left = pieces[best_position];
        const std::string right = pieces[best_position + 1];
        for (size_t position = 0; position + 1 < pieces.size();) {
            if (pieces[position] == left && pieces[position + 1] == right) {
                pieces[position].append(pieces[position + 1]);
                pieces.erase(pieces.begin() + position + 1);
            } else {
                ++position;
            }
        }
    }
    std::vector<int64_t> ids;
    ids.reserve(pieces.size());
    for (const std::string& piece : pieces) {
        const auto found = bytes_to_base_id_.find(piece);
        if (found == bytes_to_base_id_.end()) {
            throw std::runtime_error("BPE produced an unknown token");
        }
        ids.push_back(found->second);
    }
    return ids;
}

std::vector<int64_t> ByteBpeTokenizer::encode(
    const std::string& text) const {
    const auto append_piece = [&](
        std::vector<int64_t>& output,
        const std::string& piece) {
        const auto ids = encode_bpe_piece(piece);
        output.insert(output.end(), ids.begin(), ids.end());
    };
    const auto whitespace = [](unsigned char value) {
        return value == ' ' || value == '\t' || value == '\n' ||
               value == '\r' || value == '\f' || value == '\v';
    };

    std::vector<int64_t> output;
    size_t position = 0;
    while (position < text.size()) {
        if (!whitespace(static_cast<unsigned char>(text[position]))) {
            size_t end = position + 1;
            while (end < text.size() &&
                   !whitespace(static_cast<unsigned char>(text[end]))) {
                ++end;
            }
            append_piece(output, text.substr(position, end - position));
            position = end;
            continue;
        }

        size_t end = position + 1;
        bool only_spaces = text[position] == ' ';
        while (end < text.size() &&
               whitespace(static_cast<unsigned char>(text[end]))) {
            only_spaces = only_spaces && text[end] == ' ';
            ++end;
        }
        if (!only_spaces || end == text.size()) {
            append_piece(output, text.substr(position, end - position));
            position = end;
            continue;
        }
        const size_t count = end - position;
        if (count > 1) {
            append_piece(output, text.substr(position, count - 1));
        }
        size_t word_end = end;
        while (word_end < text.size() &&
               !whitespace(static_cast<unsigned char>(text[word_end]))) {
            ++word_end;
        }
        append_piece(
            output,
            text.substr(end - 1, word_end - (end - 1)));
        position = word_end;
    }
    return output;
}

std::string ByteBpeTokenizer::decode(
    const std::vector<int64_t>& token_ids,
    bool skip_special_tokens) const {
    std::string text;
    for (int64_t id : token_ids) {
        if (id >= 0 &&
            static_cast<size_t>(id) < base_id_to_bytes_.size()) {
            text.append(base_id_to_bytes_[static_cast<size_t>(id)]);
            continue;
        }
        const auto found = added_tokens_.find(id);
        if (found == added_tokens_.end()) {
            continue;
        }
        if (!skip_special_tokens || !found->second.special) {
            text.append(found->second.content);
        }
    }
    return text;
}

std::string build_single_image_chat_text(
    const std::string& prompt,
    int image_token_count) {
    if (image_token_count < 0) {
        throw std::invalid_argument("image_token_count must be non-negative");
    }
    std::string text =
        "<|begin_of_text|>system\n"
        "You are a helpful assistant.<|end_of_text|>\n"
        "<|begin_of_text|>user\n"
        "<|vision_start|>";
    for (int index = 0; index < image_token_count; ++index) {
        text.append("<|image_pad|>");
    }
    text.append("<|vision_end|>");
    text.append(prompt);
    text.append(
        "<|end_of_text|>\n"
        "<|begin_of_text|>assistant\n");
    return text;
}

std::vector<int64_t> build_single_image_chat_ids(
    const ByteBpeTokenizer& tokenizer,
    const std::string& prompt,
    int image_token_count) {
    std::vector<int64_t> ids;
    ids.reserve(static_cast<size_t>(image_token_count) + 32);
    ids.push_back(kBeginOfTextTokenId);
    append_encoded(ids, tokenizer, "system\nYou are a helpful assistant.");
    ids.push_back(kEndOfTextTokenId);
    append_encoded(ids, tokenizer, "\n");
    ids.push_back(kBeginOfTextTokenId);
    append_encoded(ids, tokenizer, "user\n");
    ids.push_back(kVisionStartTokenId);
    ids.insert(ids.end(), image_token_count, kImagePadTokenId);
    ids.push_back(kVisionEndTokenId);
    append_encoded(ids, tokenizer, prompt);
    ids.push_back(kEndOfTextTokenId);
    append_encoded(ids, tokenizer, "\n");
    ids.push_back(kBeginOfTextTokenId);
    append_encoded(ids, tokenizer, "assistant\n");
    return ids;
}

ProcessorOutput process_single_image_prompt(
    const std::filesystem::path& image_path,
    const std::string& prompt,
    const ByteBpeTokenizer& tokenizer,
    int max_image_patches) {
    const DecodedImage decoded = decode_image_rgb(image_path);
    const int source_width = decoded.width;
    const int source_height = decoded.height;
    const auto target =
        target_image_size(source_height, source_width, max_image_patches);
    const int target_height = target.first;
    const int target_width = target.second;
    const std::vector<unsigned char> resized = resize_bilinear_antialias_u8(
        decoded.rgb.data(),
        source_width,
        source_height,
        target_width,
        target_height);

    constexpr int patch_size = 16;
    constexpr int merge_size = 2;
    constexpr int channels = 3;
    const int grid_h = target_height / patch_size;
    const int grid_w = target_width / patch_size;
    const int patch_count = grid_h * grid_w;
    Array pixel_values{
        {1, patch_count, patch_size * patch_size * channels},
        std::vector<float>(
            static_cast<size_t>(patch_count) *
            patch_size * patch_size * channels),
    };
    size_t output_index = 0;
    for (int group_h = 0; group_h < grid_h / merge_size; ++group_h) {
        for (int group_w = 0; group_w < grid_w / merge_size; ++group_w) {
            for (int merge_h = 0; merge_h < merge_size; ++merge_h) {
                for (int merge_w = 0; merge_w < merge_size; ++merge_w) {
                    const int patch_y =
                        (group_h * merge_size + merge_h) * patch_size;
                    const int patch_x =
                        (group_w * merge_size + merge_w) * patch_size;
                    for (int patch_h = 0; patch_h < patch_size; ++patch_h) {
                        for (int patch_w = 0; patch_w < patch_size; ++patch_w) {
                            const size_t source_index =
                                (static_cast<size_t>(patch_y + patch_h) *
                                     target_width +
                                 patch_x + patch_w) *
                                channels;
                            for (int channel = 0; channel < channels; ++channel) {
                                const double rescaled =
                                    static_cast<double>(
                                        resized[source_index + channel]) /
                                    255.0;
                                pixel_values.data[output_index++] =
                                    static_cast<float>(
                                        (rescaled - 0.5) / 0.5);
                            }
                        }
                    }
                }
            }
        }
    }
    if (output_index != pixel_values.data.size()) {
        throw std::runtime_error("image patchification size mismatch");
    }

    const int image_token_count = patch_count / 4;
    const std::vector<int64_t> input_ids = build_single_image_chat_ids(
        tokenizer, prompt, image_token_count);
    ProcessorOutput output;
    output.pixel_values = std::move(pixel_values);
    output.pixel_attention_mask = {
        {1, patch_count},
        std::vector<int64_t>(patch_count, 1),
    };
    output.spatial_shapes = {{1, 2}, {grid_h, grid_w}};
    output.input_ids = {
        {1, static_cast<int>(input_ids.size())},
        input_ids,
    };
    output.attention_mask = {
        output.input_ids.shape,
        std::vector<int64_t>(input_ids.size(), 1),
    };
    output.original_height = source_height;
    output.original_width = source_width;
    output.resized_height = target_height;
    output.resized_width = target_width;
    output.chat_text =
        build_single_image_chat_text(prompt, image_token_count);
    return output;
}

void save_processor_output(
    const std::filesystem::path& output_dir,
    const ProcessorOutput& output) {
    std::filesystem::create_directories(output_dir);
    save_npy_float32(
        (output_dir / "pixel_values.npy").string(),
        output.pixel_values.shape,
        output.pixel_values.data);
    std::vector<int32_t> pixel_mask(
        output.pixel_attention_mask.data.begin(),
        output.pixel_attention_mask.data.end());
    write_npy_int32(
        output_dir / "pixel_attention_mask.npy",
        output.pixel_attention_mask.shape,
        pixel_mask);
    save_npy_int64(
        (output_dir / "spatial_shapes.npy").string(),
        output.spatial_shapes.shape,
        output.spatial_shapes.data);
    save_npy_int64(
        (output_dir / "input_ids.npy").string(),
        output.input_ids.shape,
        output.input_ids.data);
    save_npy_int64(
        (output_dir / "attention_mask.npy").string(),
        output.attention_mask.shape,
        output.attention_mask.data);
    std::ofstream chat(output_dir / "chat_template.txt");
    if (!chat) {
        throw std::runtime_error("failed to create chat_template.txt");
    }
    chat << output.chat_text;
}

std::vector<int64_t> convert_coordinate_token_ids(
    const std::vector<int64_t>& token_ids,
    int original_width,
    int original_height,
    int resized_width,
    int resized_height) {
    if (original_width <= 0 || original_height <= 0 ||
        resized_width <= 0 || resized_height <= 0) {
        throw std::invalid_argument("invalid coordinate scaling dimensions");
    }
    constexpr int max_coordinate = 2047;
    const double scale_x =
        static_cast<double>(original_width) / resized_width;
    const double scale_y =
        static_cast<double>(original_height) / resized_height;
    std::vector<int64_t> converted;
    converted.reserve(token_ids.size());
    for (int64_t token_id : token_ids) {
        if (token_id >= kX0TokenId && token_id <= kY2047TokenId) {
            const int64_t offset = token_id - kX0TokenId;
            const bool is_y = (offset & 1) != 0;
            const int coordinate = static_cast<int>(offset >> 1);
            const double scale = is_y ? scale_y : scale_x;
            int scaled = static_cast<int>(std::nearbyint(coordinate * scale));
            scaled = std::clamp(scaled, 0, max_coordinate);
            token_id =
                kX0TokenId + (static_cast<int64_t>(scaled) << 1) +
                (is_y ? 1 : 0);
        }
        converted.push_back(token_id);
    }
    return converted;
}

}  // namespace youtu_vl
