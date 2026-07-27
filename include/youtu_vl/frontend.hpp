#pragma once

#include "youtu_vl/llm_decoder_layer.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace youtu_vl {

constexpr int64_t kBeginOfTextTokenId = 128000;
constexpr int64_t kEndOfTextTokenId = 128001;
constexpr int64_t kVisionStartTokenId = 128262;
constexpr int64_t kVisionEndTokenId = 128263;
constexpr int64_t kImagePadTokenId = 128264;
constexpr int64_t kX0TokenId = 278267;
constexpr int64_t kY2047TokenId = 282362;

class ByteBpeTokenizer {
public:
    explicit ByteBpeTokenizer(const std::filesystem::path& binary_path);

    std::vector<int64_t> encode(const std::string& text) const;
    std::string decode(
        const std::vector<int64_t>& token_ids,
        bool skip_special_tokens = true) const;

private:
    std::vector<int64_t> encode_bpe_piece(const std::string& text) const;

    struct AddedToken {
        std::string content;
        bool special = false;
    };

    std::vector<std::string> base_id_to_bytes_;
    std::unordered_map<std::string, int64_t> bytes_to_base_id_;
    std::unordered_map<std::string, uint32_t> merge_ranks_;
    std::unordered_map<int64_t, AddedToken> added_tokens_;
};

struct ProcessorOutput {
    Array pixel_values;
    Int64Array pixel_attention_mask;
    Int64Array spatial_shapes;
    Int64Array input_ids;
    Int64Array attention_mask;
    int original_height = 0;
    int original_width = 0;
    int resized_height = 0;
    int resized_width = 0;
    std::string chat_text;
};

std::string build_single_image_chat_text(
    const std::string& prompt,
    int image_token_count);

std::vector<int64_t> build_single_image_chat_ids(
    const ByteBpeTokenizer& tokenizer,
    const std::string& prompt,
    int image_token_count);

ProcessorOutput process_single_image_prompt(
    const std::filesystem::path& image_path,
    const std::string& prompt,
    const ByteBpeTokenizer& tokenizer,
    int max_image_patches = 36864);

void save_processor_output(
    const std::filesystem::path& output_dir,
    const ProcessorOutput& output);

std::vector<int64_t> convert_coordinate_token_ids(
    const std::vector<int64_t>& token_ids,
    int original_width,
    int original_height,
    int resized_width,
    int resized_height);

}  // namespace youtu_vl
