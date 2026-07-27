#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
package_root="$(cd -- "${project_root}/.." && pwd)"

if [[ -z "${NCNN_DIR:-}" ]]; then
    echo "NCNN_DIR must point to ncnn/lib/cmake/ncnn" >&2
    exit 2
fi

build_dir="${LINUX_BUILD_DIR:-${project_root}/linux-test/build/retest}"
result_root="${LINUX_RESULT_DIR:-${project_root}/linux-test/results/retest}"
model_root="${MODEL_ROOT:-${package_root}/model}"
test_data_root="${TEST_DATA_ROOT:-${package_root}/test_data}"
threads="${THREADS:-8}"

cmake -S "${project_root}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -Dncnn_DIR="${NCNN_DIR}" \
    -DYOUTU_VL_BUILD_TESTS=ON
cmake --build "${build_dir}" --parallel "${threads}"
ctest --test-dir "${build_dir}" --output-on-failure

if [[ "${FULL_REGRESSION:-0}" != "1" ]]; then
    exit 0
fi

mkdir -p "${result_root}"
for tokens in 32 64 128; do
    output_dir="${result_root}/inference-${tokens}"
    "${build_dir}/youtu_vl_ncnn" \
        --model-root "${model_root}" \
        --image "${test_data_root}/input_image.jpg" \
        --prompt "这是什么类型的垃圾" \
        --output-dir "${output_dir}" \
        --max-new-tokens "${tokens}" \
        --weight-mode persistent \
        --threads "${threads}"

    reference="${test_data_root}/reference_${tokens}/generated_token_ids.npy"
    cmp --silent "${output_dir}/generated_token_ids.npy" "${reference}"
    echo "${tokens}/${tokens} generated token IDs: PASS"
done
