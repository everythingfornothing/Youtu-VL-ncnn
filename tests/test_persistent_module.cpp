#include "youtu_vl/llm_decoder_layer.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr
            << "Usage: " << argv[0]
            << " merger_model_dir vision_hidden.npy\n";
        return 2;
    }

    try {
        const std::filesystem::path model_dir = argv[1];
        const youtu_vl::Array input = youtu_vl::load_npy_float(argv[2]);
        youtu_vl::NcnnModule module(
            model_dir,
            "youtu_merger",
            youtu_vl::NcnnRuntimeOptions{1, false});
        if (module.loaded() || module.model_load_count() != 0) {
            throw std::runtime_error("module must start unloaded");
        }

        module.load();
        if (!module.loaded() || module.model_load_count() != 1) {
            throw std::runtime_error("persistent module did not load exactly once");
        }

        const youtu_vl::Array first = module.run({input}, 1)[0];
        const youtu_vl::Array second = module.run({input}, 1)[0];
        if (first.shape != second.shape || first.data != second.data) {
            throw std::runtime_error("repeated persistent outputs differ");
        }
        if (module.model_load_count() != 1 || module.run_count() != 2) {
            throw std::runtime_error("persistent module reloaded during execution");
        }
        if (!std::all_of(
                second.data.begin(),
                second.data.end(),
                [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("persistent output contains NaN or Inf");
        }

        std::cout
            << "status=PASS shape=[" << second.shape[0] << ","
            << second.shape[1] << "] model_load_count="
            << module.model_load_count() << " run_count="
            << module.run_count() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
