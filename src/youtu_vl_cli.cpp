#include "youtu_vl/frontend.hpp"
#include "youtu_vl/llm_decoder_layer.hpp"
#include "youtu_vl/llm_prefill.hpp"
#include "youtu_vl/vision_pipeline.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <shellapi.h>
#include <windows.h>
#endif

namespace {

std::vector<std::string> command_line_arguments(
    int argc,
    char** argv) {
#ifdef _WIN32
    int wide_argc = 0;
    wchar_t** wide_argv =
        CommandLineToArgvW(GetCommandLineW(), &wide_argc);
    if (wide_argv != nullptr) {
        std::vector<std::string> arguments;
        arguments.reserve(static_cast<size_t>(wide_argc));
        for (int index = 0; index < wide_argc; ++index) {
            const int wide_length = lstrlenW(wide_argv[index]);
            if (wide_length == 0) {
                arguments.emplace_back();
                continue;
            }
            const int utf8_length = WideCharToMultiByte(
                CP_UTF8,
                0,
                wide_argv[index],
                wide_length,
                nullptr,
                0,
                nullptr,
                nullptr);
            if (utf8_length <= 0) {
                arguments.clear();
                break;
            }
            std::string argument(
                static_cast<size_t>(utf8_length),
                '\0');
            if (WideCharToMultiByte(
                    CP_UTF8,
                    0,
                    wide_argv[index],
                    wide_length,
                    argument.data(),
                    utf8_length,
                    nullptr,
                    nullptr) != utf8_length) {
                arguments.clear();
                break;
            }
            arguments.push_back(std::move(argument));
        }
        LocalFree(wide_argv);
        if (arguments.size() == static_cast<size_t>(wide_argc)) {
            return arguments;
        }
    }
#endif
    return std::vector<std::string>(argv, argv + argc);
}

struct CachePair {
    youtu_vl::Array key;
    youtu_vl::Array value;
};

struct HeadShard {
    std::string name;
    int vocabulary_start;
    int vocabulary_end;
};

struct TokenResult {
    int token_id;
    float logit;
    float raw_logit;
    int second_token_id;
    float second_logit;
    float second_raw_logit;
    bool raw_tie_break_applied;
};

std::string layer_name(int layer_index) {
    std::ostringstream stream;
    stream << "layer_" << std::setfill('0') << std::setw(2) << layer_index;
    return stream.str();
}

std::filesystem::path layer_model_path(
    const std::filesystem::path& root,
    int layer_index) {
    return root /
           ("artifacts/llm_layer" + std::to_string(layer_index) +
            "_three_part");
}

youtu_vl::Array rope_for_positions(
    int start_position,
    int length,
    bool sine,
    const youtu_vl::Array& inverse_frequency) {
    if (inverse_frequency.data.size() !=
        youtu_vl::kQkRopeHeadDim / 2) {
        throw std::runtime_error("RoPE inv_freq must contain 32 values");
    }
    youtu_vl::Array output{
        {length, youtu_vl::kQkRopeHeadDim},
        std::vector<float>(
            static_cast<size_t>(length) * youtu_vl::kQkRopeHeadDim),
    };
    for (int token = 0; token < length; ++token) {
        const int position = start_position + token;
        for (int index = 0;
             index < youtu_vl::kQkRopeHeadDim / 2;
             ++index) {
            const float frequency =
                inverse_frequency.data[index] * position;
            const float value = youtu_vl::round_to_half(
                sine ? std::sin(frequency) : std::cos(frequency));
            output.data[
                static_cast<size_t>(token) * youtu_vl::kQkRopeHeadDim +
                index] = value;
            output.data[
                static_cast<size_t>(token) * youtu_vl::kQkRopeHeadDim +
                index + youtu_vl::kQkRopeHeadDim / 2] = value;
        }
    }
    return output;
}

std::vector<HeadShard> head_shards() {
    return {
        {"lm_head_shard_00", 0, 32768},
        {"lm_head_shard_01", 32768, 65536},
        {"lm_head_shard_02", 65536, 98304},
        {"lm_head_shard_03", 98304, 131072},
        {"lm_head_shard_04", 131072, 163840},
        {"lm_head_shard_05", 163840, 196608},
        {"lm_head_shard_06", 196608, 204800},
        {"lm_head_shard_07", 204800, 212992},
        {"lm_head_shard_08", 212992, 221184},
        {"lm_head_shard_09", 221184, 229376},
        {"lm_head_shard_10", 229376, 237568},
        {"lm_head_shard_11", 237568, 245760},
        {"lm_head_shard_12", 245760, 253952},
        {"lm_head_shard_13", 253952, 262144},
        {"lm_head_shard_14", 262144, 270336},
        {"lm_head_shard_15", 270336, 278528},
        {"lm_head_shard_16", 278528, 283386},
    };
}

void update_top2(int token_id, float raw_logit, TokenResult& result) {
    const float logit = youtu_vl::round_to_half(raw_logit);
    if (logit > result.logit) {
        result.second_token_id = result.token_id;
        result.second_logit = result.logit;
        result.second_raw_logit = result.raw_logit;
        result.token_id = token_id;
        result.logit = logit;
        result.raw_logit = raw_logit;
    } else if (logit > result.second_logit) {
        result.second_token_id = token_id;
        result.second_logit = logit;
        result.second_raw_logit = raw_logit;
    }
}

void recover_fp16_tie_from_raw_logits(TokenResult& result) {
    if (result.logit != result.second_logit ||
        result.second_raw_logit <= result.raw_logit) {
        return;
    }
    const uint16_t half_value = youtu_vl::float_to_half(result.logit);
    if ((half_value & 0x8000) != 0 || half_value >= 0x7bff) {
        return;
    }
    const float next_half = youtu_vl::half_to_float(half_value + 1);
    const float half_ulp = next_half - result.logit;
    const float raw_gap = result.second_raw_logit - result.raw_logit;
    if (raw_gap <= half_ulp * 0.5f) {
        return;
    }
    std::swap(result.token_id, result.second_token_id);
    std::swap(result.logit, result.second_logit);
    std::swap(result.raw_logit, result.second_raw_logit);
    result.raw_tie_break_applied = true;
}

std::pair<int, int> torch_top2_token_ids(const TokenResult& result) {
    int first = result.token_id;
    int second = result.second_token_id;
    if (result.logit == result.second_logit && first < second) {
        std::swap(first, second);
    }
    return {first, second};
}

class FinalHeadRuntime {
public:
    FinalHeadRuntime(
        const std::filesystem::path& head_dir,
        youtu_vl::NcnnRuntimeOptions options,
        bool load_immediately)
        : final_norm_(head_dir / "final_norm", "final_norm", options),
          shards_(head_shards()) {
        shard_modules_.reserve(shards_.size());
        for (const HeadShard& shard : shards_) {
            shard_modules_.push_back(std::make_unique<youtu_vl::NcnnModule>(
                head_dir / shard.name,
                shard.name,
                options));
        }
        if (load_immediately) {
            load();
        }
    }

    void load() {
        final_norm_.load();
        for (const auto& module : shard_modules_) {
            module->load();
        }
    }

    bool loaded() const {
        if (!final_norm_.loaded()) {
            return false;
        }
        return std::all_of(
            shard_modules_.begin(),
            shard_modules_.end(),
            [](const auto& module) { return module->loaded(); });
    }

    size_t model_load_count() const {
        size_t count = final_norm_.model_load_count();
        for (const auto& module : shard_modules_) {
            count += module->model_load_count();
        }
        return count;
    }

    TokenResult run(const youtu_vl::Array& hidden_states) const {
        using namespace youtu_vl;
        if (hidden_states.shape.size() != 2 ||
            hidden_states.shape[1] != kHiddenSize) {
            throw std::runtime_error(
                "final head input must be [sequence, 2560]");
        }
        Array last_hidden{
            {1, kHiddenSize},
            std::vector<float>(kHiddenSize),
        };
        std::copy_n(
            hidden_states.data.end() - kHiddenSize,
            kHiddenSize,
            last_hidden.data.begin());
        for (float& value : last_hidden.data) {
            value = round_to_half(value);
        }
        Array normalized = final_norm_.run({last_hidden}, 1)[0];
        for (float& value : normalized.data) {
            value = round_to_half(value);
        }

        TokenResult result{
            -1,
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -1,
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            false,
        };
        for (size_t shard_index = 0;
             shard_index < shards_.size();
             ++shard_index) {
            const HeadShard& shard = shards_[shard_index];
            const Array logits =
                shard_modules_[shard_index]->run({normalized}, 1)[0];
            for (size_t index = 0; index < logits.data.size(); ++index) {
                update_top2(
                    shard.vocabulary_start + static_cast<int>(index),
                    logits.data[index],
                    result);
            }
        }
        recover_fp16_tie_from_raw_logits(result);
        return result;
    }

private:
    youtu_vl::NcnnModule final_norm_;
    std::vector<HeadShard> shards_;
    std::vector<std::unique_ptr<youtu_vl::NcnnModule>> shard_modules_;
};

youtu_vl::Array build_fused_embeddings(
    const youtu_vl::Int64Array& input_ids,
    youtu_vl::NpyFloatRowReader& embeddings,
    const youtu_vl::Array& image_embeddings,
    int64_t image_token_id) {
    using namespace youtu_vl;
    if (input_ids.shape.size() != 2 || input_ids.shape[0] != 1) {
        throw std::runtime_error("input_ids must have shape [1, sequence]");
    }
    const int sequence_length = input_ids.shape[1];
    Array fused{
        {sequence_length, kHiddenSize},
        std::vector<float>(
            static_cast<size_t>(sequence_length) * kHiddenSize),
    };
    std::vector<int> image_positions;
    for (int token = 0; token < sequence_length; ++token) {
        Array row = embeddings.read_row(
            static_cast<int>(input_ids.data[token]));
        std::copy(
            row.data.begin(),
            row.data.end(),
            fused.data.begin() +
                static_cast<size_t>(token) * kHiddenSize);
        if (input_ids.data[token] == image_token_id) {
            image_positions.push_back(token);
        }
    }
    if (image_embeddings.shape !=
        std::vector<int>{
            static_cast<int>(image_positions.size()),
            kHiddenSize,
        }) {
        throw std::runtime_error(
            "image token count does not match image embeddings");
    }
    for (size_t feature = 0; feature < image_positions.size(); ++feature) {
        std::copy_n(
            image_embeddings.data.begin() +
                feature * kHiddenSize,
            kHiddenSize,
            fused.data.begin() +
                static_cast<size_t>(image_positions[feature]) * kHiddenSize);
    }
    return fused;
}

void save_cache(
    const std::filesystem::path& output_dir,
    const std::vector<CachePair>& caches) {
    const auto cache_dir = output_dir / "final_cache";
    std::filesystem::create_directories(cache_dir);
    for (size_t index = 0; index < caches.size(); ++index) {
        const std::string name = layer_name(static_cast<int>(index));
        youtu_vl::save_npy_float32(
            (cache_dir / (name + "_key.npy")).string(),
            caches[index].key.shape,
            caches[index].key.data);
        youtu_vl::save_npy_float32(
            (cache_dir / (name + "_value.npy")).string(),
            caches[index].value.shape,
            caches[index].value.data);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> argument_storage =
        command_line_arguments(argc, argv);
    std::vector<char*> argument_pointers;
    argument_pointers.reserve(argument_storage.size());
    for (std::string& argument : argument_storage) {
        argument_pointers.push_back(argument.data());
    }
    argc = static_cast<int>(argument_pointers.size());
    argv = argument_pointers.data();

    const bool cpp_frontend = argc >= 2 && std::string(argv[1]) == "--model-root";
    if ((!cpp_frontend && (argc < 10 || argc > 13)) || argc == 1) {
        std::cerr
            << "Image + prompt mode:\n  " << argv[0]
            << " --model-root MODEL --image IMAGE --prompt TEXT"
            << " --output-dir DIR [--tokenizer FILE]"
            << " [--max-new-tokens N] [--weight-mode persistent|streaming]"
            << " [--threads N] [--max-image-patches N]"
            << " [--save-final-cache] [--frontend-only]\n\n"
            << "Legacy tensor mode:\n  " << argv[0]
            << " workspace_root pixel_values.npy spatial_shapes.npy"
            << " input_ids.npy embedding_weight.npy output_dir"
            << " decode_steps image_token_id save_final_cache"
            << " [eos_token_id] [streaming|persistent] [num_threads]\n";
        return 2;
    }

    try {
        using namespace youtu_vl;
        std::filesystem::path root;
        std::string pixel_values_path;
        std::string spatial_shapes_path;
        std::string input_ids_path;
        std::string embedding_path;
        std::filesystem::path output_dir;
        std::filesystem::path image_path;
        std::filesystem::path tokenizer_path;
        std::string prompt;
        int decode_steps = 0;
        int64_t image_token_id = kImagePadTokenId;
        bool should_save_cache = false;
        int eos_token_id = static_cast<int>(kEndOfTextTokenId);
        std::string weight_mode = "persistent";
        int num_threads = 1;
        int max_image_patches = 36864;
        bool frontend_only = false;

        if (cpp_frontend) {
            for (int index = 1; index < argc; ++index) {
                const std::string option = argv[index];
                if (option == "--save-final-cache") {
                    should_save_cache = true;
                    continue;
                }
                if (option == "--frontend-only") {
                    frontend_only = true;
                    continue;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "missing value after option " + option);
                }
                const std::string value = argv[++index];
                if (option == "--model-root") {
                    root = value;
                } else if (option == "--image") {
                    image_path = value;
                } else if (option == "--prompt") {
                    prompt = value;
                } else if (option == "--output-dir") {
                    output_dir = value;
                } else if (option == "--tokenizer") {
                    tokenizer_path = value;
                } else if (option == "--embedding") {
                    embedding_path = value;
                } else if (option == "--max-new-tokens") {
                    const int count = std::stoi(value);
                    if (count < 1) {
                        throw std::invalid_argument(
                            "max-new-tokens must be positive");
                    }
                    decode_steps = count - 1;
                } else if (option == "--image-token-id") {
                    image_token_id = std::stoll(value);
                } else if (option == "--eos-token-id") {
                    eos_token_id = std::stoi(value);
                } else if (option == "--weight-mode") {
                    weight_mode = value;
                } else if (option == "--threads") {
                    num_threads = std::stoi(value);
                } else if (option == "--max-image-patches") {
                    max_image_patches = std::stoi(value);
                } else {
                    throw std::invalid_argument("unknown option: " + option);
                }
            }
            if (root.empty() || image_path.empty() || output_dir.empty()) {
                throw std::invalid_argument(
                    "--model-root, --image and --output-dir are required");
            }
            if (tokenizer_path.empty()) {
                tokenizer_path = root / "tokenizer/tokenizer.bin";
            }
            if (embedding_path.empty()) {
                embedding_path =
                    (root / "artifacts/text_embedding/embed_tokens_weight.npy")
                        .string();
            }
        } else {
            root = argv[1];
            pixel_values_path = argv[2];
            spatial_shapes_path = argv[3];
            input_ids_path = argv[4];
            embedding_path = argv[5];
            output_dir = argv[6];
            decode_steps = std::stoi(argv[7]);
            image_token_id = std::stoll(argv[8]);
            should_save_cache = std::stoi(argv[9]) != 0;
            eos_token_id = argc >= 11 ? std::stoi(argv[10]) : 128001;
            weight_mode = argc >= 12 ? argv[11] : "streaming";
            num_threads = argc >= 13 ? std::stoi(argv[12]) : 1;
        }
        if (weight_mode != "streaming" && weight_mode != "persistent") {
            throw std::invalid_argument(
                "weight mode must be streaming or persistent");
        }
        if (num_threads < 1) {
            throw std::invalid_argument("num_threads must be positive");
        }
        if (decode_steps < 0) {
            throw std::invalid_argument("decode_steps must be non-negative");
        }
        const bool persistent_weights = weight_mode == "persistent";
        const NcnnRuntimeOptions runtime_options{num_threads, false};
        omp_set_num_threads(num_threads);
        std::filesystem::create_directories(output_dir);

        const auto total_started = std::chrono::steady_clock::now();
        Array pixel_values;
        Int64Array spatial_shapes;
        Int64Array input_ids;
        std::unique_ptr<ByteBpeTokenizer> tokenizer;
        int original_width = 0;
        int original_height = 0;
        int resized_width = 0;
        int resized_height = 0;
        double frontend_seconds = 0.0;
        if (cpp_frontend) {
            const auto frontend_started = std::chrono::steady_clock::now();
            tokenizer = std::make_unique<ByteBpeTokenizer>(tokenizer_path);
            ProcessorOutput processed = process_single_image_prompt(
                image_path, prompt, *tokenizer, max_image_patches);
            save_processor_output(output_dir / "processor_inputs", processed);
            original_width = processed.original_width;
            original_height = processed.original_height;
            resized_width = processed.resized_width;
            resized_height = processed.resized_height;
            pixel_values = std::move(processed.pixel_values);
            spatial_shapes = std::move(processed.spatial_shapes);
            input_ids = std::move(processed.input_ids);
            frontend_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - frontend_started)
                    .count();
            std::cout
                << "C++ frontend complete: source=["
                << processed.original_height << ","
                << processed.original_width << "] resized=["
                << processed.resized_height << ","
                << processed.resized_width << "] grid=["
                << spatial_shapes.data[0] << ","
                << spatial_shapes.data[1] << "] input_tokens="
                << input_ids.data.size() << " elapsed="
                << frontend_seconds << "s\n";
            if (frontend_only) {
                std::ofstream report(output_dir / "frontend_report.json");
                report << std::fixed << std::setprecision(6)
                       << "{\n"
                       << "  \"frontend\": \"cpp\",\n"
                       << "  \"python_runtime_dependency\": false,\n"
                       << "  \"frontend_seconds\": " << frontend_seconds << ",\n"
                       << "  \"input_tokens\": " << input_ids.data.size() << ",\n"
                       << "  \"image_patches\": "
                       << pixel_values.shape[1] << "\n"
                       << "}\n";
                return 0;
            }
        } else {
            pixel_values = load_npy_float(pixel_values_path);
            spatial_shapes = load_npy_int64(spatial_shapes_path);
            input_ids = load_npy_int64(input_ids_path);
        }
        const Array inverse_frequency =
            load_npy_float(
                (root / "artifacts/llm_rope_inv_freq.npy").string());
        if (spatial_shapes.data.size() != 2) {
            throw std::runtime_error("spatial_shapes must contain grid h,w");
        }

        const auto vision_started = std::chrono::steady_clock::now();
        const VisionPipeline vision(root);
        Array vision_hidden = vision.encode(
            pixel_values,
            static_cast<int>(spatial_shapes.data[0]),
            static_cast<int>(spatial_shapes.data[1]));
        Array image_embeddings = vision.merge(vision_hidden);
        const double vision_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - vision_started)
                .count();
        save_npy_float32(
            (output_dir / "vision_hidden.npy").string(),
            vision_hidden.shape,
            vision_hidden.data);
        save_npy_float32(
            (output_dir / "image_embeddings.npy").string(),
            image_embeddings.shape,
            image_embeddings.data);
        std::cout
            << "vision complete: hidden=[" << vision_hidden.shape[0]
            << "," << vision_hidden.shape[1] << "] image_embeddings=["
            << image_embeddings.shape[0] << ","
            << image_embeddings.shape[1] << "] elapsed="
            << vision_seconds
            << "s\n";

        NpyFloatRowReader embedding_reader(embedding_path);
        Array hidden_states = build_fused_embeddings(
            input_ids,
            embedding_reader,
            image_embeddings,
            image_token_id);
        save_npy_float32(
            (output_dir / "fused_embeddings.npy").string(),
            {1, hidden_states.shape[0], hidden_states.shape[1]},
            hidden_states.data);

        const int prefill_length = hidden_states.shape[0];
        const Array prefill_cos =
            rope_for_positions(
                0,
                prefill_length,
                false,
                inverse_frequency);
        const Array prefill_sin =
            rope_for_positions(
                0,
                prefill_length,
                true,
                inverse_frequency);
        save_npy_float32(
            (output_dir / "prefill_rope_cos.npy").string(),
            prefill_cos.shape,
            prefill_cos.data);
        save_npy_float32(
            (output_dir / "prefill_rope_sin.npy").string(),
            prefill_sin.shape,
            prefill_sin.data);

        const auto model_load_started = std::chrono::steady_clock::now();
        std::vector<std::unique_ptr<DecoderLayer>> decoder_layers;
        decoder_layers.reserve(40);
        for (int layer_index = 0; layer_index < 40; ++layer_index) {
            decoder_layers.push_back(std::make_unique<DecoderLayer>(
                layer_model_path(root, layer_index),
                runtime_options,
                persistent_weights));
        }
        FinalHeadRuntime final_head(
            root / "artifacts/llm_final_head_ncnn",
            runtime_options,
            persistent_weights);
        const double model_load_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - model_load_started)
                .count();
        auto model_load_count = [&]() {
            size_t count = final_head.model_load_count();
            for (const auto& layer : decoder_layers) {
                count += layer->model_load_count();
            }
            return count;
        };
        std::cout
            << "runtime initialized: weight_mode=" << weight_mode
            << " threads=" << num_threads
            << " resident_models=" << (persistent_weights ? 138 : 0)
            << " model_load_count=" << model_load_count()
            << " elapsed=" << model_load_seconds << "s\n";

        std::vector<CachePair> caches;
        caches.reserve(40);
        const auto prefill_started = std::chrono::steady_clock::now();
        for (int layer_index = 0; layer_index < 40; ++layer_index) {
            PrefillOutput output = run_decoder_prefill_layer(
                *decoder_layers[layer_index],
                hidden_states,
                prefill_cos,
                prefill_sin);
            hidden_states = std::move(output.layer_output);
            caches.push_back({
                std::move(output.key_cache),
                std::move(output.value_cache),
            });
            if (layer_index == 0 || layer_index == 20 || layer_index == 39) {
                save_npy_float32(
                    (output_dir /
                     ("prefill_layer_" +
                      (layer_index < 10 ? std::string("0") : std::string()) +
                      std::to_string(layer_index) + "_output.npy"))
                        .string(),
                    hidden_states.shape,
                    hidden_states.data);
            }
            std::cout << "prefill layer " << layer_index
                      << " complete\n";
        }
        const TokenResult prefill_token =
            final_head.run(hidden_states);
        const double prefill_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - prefill_started)
                .count();
        std::cout
            << "prefill complete: token=" << prefill_token.token_id
            << " top1=" << prefill_token.logit
            << " top2=" << prefill_token.second_logit
            << " elapsed=" << prefill_seconds << "s\n";

        const auto prefill_top2_ids = torch_top2_token_ids(prefill_token);
        std::vector<int64_t> generated_ids{prefill_token.token_id};
        std::vector<int64_t> top1_token_ids{prefill_top2_ids.first};
        std::vector<int64_t> top2_token_ids{prefill_top2_ids.second};
        std::vector<int64_t> positions;
        std::vector<float> top1_logits{prefill_token.logit};
        std::vector<float> top2_logits{prefill_token.second_logit};
        std::vector<float> top1_raw_logits{prefill_token.raw_logit};
        std::vector<float> top2_raw_logits{prefill_token.second_raw_logit};
        std::vector<int64_t> raw_tie_breaks{
            prefill_token.raw_tie_break_applied ? 1 : 0,
        };
        int current_token = prefill_token.token_id;
        bool reached_eos = current_token == eos_token_id;
        const auto decode_started = std::chrono::steady_clock::now();
        for (int step = 0; step < decode_steps && !reached_eos; ++step) {
            const int position = prefill_length + step;
            hidden_states = embedding_reader.read_row(current_token);
            const Array cos = rope_for_positions(
                position,
                1,
                false,
                inverse_frequency);
            const Array sin = rope_for_positions(
                position,
                1,
                true,
                inverse_frequency);
            for (int layer_index = 0; layer_index < 40; ++layer_index) {
                DecodeOutput output = decoder_layers[layer_index]->decode(
                    hidden_states,
                    cos,
                    sin,
                    caches[layer_index].key,
                    caches[layer_index].value);
                hidden_states = std::move(output.layer_output);
                caches[layer_index].key = std::move(output.key_cache);
                caches[layer_index].value = std::move(output.value_cache);
            }
            const TokenResult token = final_head.run(hidden_states);
            const auto diagnostic_top2_ids = torch_top2_token_ids(token);
            generated_ids.push_back(token.token_id);
            top1_token_ids.push_back(diagnostic_top2_ids.first);
            top2_token_ids.push_back(diagnostic_top2_ids.second);
            positions.push_back(position);
            top1_logits.push_back(token.logit);
            top2_logits.push_back(token.second_logit);
            top1_raw_logits.push_back(token.raw_logit);
            top2_raw_logits.push_back(token.second_raw_logit);
            raw_tie_breaks.push_back(token.raw_tie_break_applied ? 1 : 0);
            std::cout
                << "decode step " << (step + 1)
                << ": input=" << current_token
                << " output=" << token.token_id
                << " margin=" << (token.logit - token.second_logit)
                << " raw_margin="
                << (token.raw_logit - token.second_raw_logit)
                << " raw_tie_break="
                << (token.raw_tie_break_applied ? "true" : "false")
                << " cache_length=" << caches[0].key.shape[1]
                << "\n";
            current_token = token.token_id;
            reached_eos = current_token == eos_token_id;
        }
        const double decode_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - decode_started)
                .count();

        const std::vector<int64_t> raw_generated_ids = generated_ids;
        int coordinate_token_count = 0;
        if (cpp_frontend) {
            coordinate_token_count = static_cast<int>(std::count_if(
                raw_generated_ids.begin(),
                raw_generated_ids.end(),
                [](int64_t token_id) {
                    return token_id >= kX0TokenId && token_id <= kY2047TokenId;
                }));
            generated_ids = convert_coordinate_token_ids(
                raw_generated_ids,
                original_width,
                original_height,
                resized_width,
                resized_height);
        }
        save_npy_int64(
            (output_dir / "raw_generated_token_ids.npy").string(),
            {static_cast<int>(raw_generated_ids.size())},
            raw_generated_ids);
        save_npy_int64(
            (output_dir / "generated_token_ids.npy").string(),
            {static_cast<int>(generated_ids.size())},
            generated_ids);
        save_npy_int64(
            (output_dir / "top1_token_ids.npy").string(),
            {static_cast<int>(top1_token_ids.size())},
            top1_token_ids);
        save_npy_int64(
            (output_dir / "decode_positions.npy").string(),
            {static_cast<int>(positions.size())},
            positions);
        save_npy_int64(
            (output_dir / "top2_token_ids.npy").string(),
            {static_cast<int>(top2_token_ids.size())},
            top2_token_ids);
        save_npy_int64(
            (output_dir / "prefill_length.npy").string(),
            {1},
            {prefill_length});
        save_npy_int64(
            (output_dir / "final_cache_length.npy").string(),
            {1},
            {caches[0].key.shape[1]});
        save_npy_float32(
            (output_dir / "top1_logits.npy").string(),
            {static_cast<int>(top1_logits.size())},
            top1_logits);
        save_npy_float32(
            (output_dir / "top2_logits.npy").string(),
            {static_cast<int>(top2_logits.size())},
            top2_logits);
        save_npy_float32(
            (output_dir / "top1_raw_logits.npy").string(),
            {static_cast<int>(top1_raw_logits.size())},
            top1_raw_logits);
        save_npy_float32(
            (output_dir / "top2_raw_logits.npy").string(),
            {static_cast<int>(top2_raw_logits.size())},
            top2_raw_logits);
        save_npy_int64(
            (output_dir / "raw_tie_breaks.npy").string(),
            {static_cast<int>(raw_tie_breaks.size())},
            raw_tie_breaks);
        if (should_save_cache) {
            save_cache(output_dir, caches);
        }
        if (tokenizer) {
            std::ofstream text_output(output_dir / "output.txt");
            if (!text_output) {
                throw std::runtime_error("failed to create output.txt");
            }
            text_output << tokenizer->decode(generated_ids, true) << '\n';
        }
        const size_t final_model_load_count = model_load_count();
        const double total_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - total_started)
                .count();
        std::ofstream runtime_report(output_dir / "runtime_report.json");
        if (!runtime_report) {
            throw std::runtime_error("failed to create runtime_report.json");
        }
        runtime_report
            << std::fixed << std::setprecision(6)
            << "{\n"
            << "  \"weight_mode\": \"" << weight_mode << "\",\n"
            << "  \"frontend\": \""
            << (cpp_frontend ? "cpp" : "precomputed_npy")
            << "\",\n"
            << "  \"python_runtime_dependency\": false,\n"
            << "  \"frontend_seconds\": " << frontend_seconds << ",\n"
            << "  \"num_threads\": " << num_threads << ",\n"
            << "  \"resident_models\": "
            << (persistent_weights ? 138 : 0) << ",\n"
            << "  \"model_load_count\": " << final_model_load_count
            << ",\n"
            << "  \"model_load_seconds\": " << model_load_seconds << ",\n"
            << "  \"vision_seconds\": " << vision_seconds << ",\n"
            << "  \"prefill_seconds\": " << prefill_seconds << ",\n"
            << "  \"decode_seconds\": " << decode_seconds << ",\n"
            << "  \"total_seconds\": " << total_seconds << ",\n"
            << "  \"generated_tokens\": " << generated_ids.size() << ",\n"
            << "  \"coordinate_tokens_postprocessed\": "
            << coordinate_token_count << ",\n"
            << "  \"final_cache_length\": " << caches[0].key.shape[1]
            << ",\n"
            << "  \"reached_eos\": "
            << (reached_eos ? "true" : "false") << "\n"
            << "}\n";
        std::cout
            << "end-to-end complete: generated_tokens="
            << generated_ids.size()
            << " final_cache_length=" << caches[0].key.shape[1]
            << " reached_eos=" << (reached_eos ? "true" : "false")
            << " weight_mode=" << weight_mode
            << " model_load_count=" << final_model_load_count
            << " total_elapsed=" << total_seconds << "s\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
