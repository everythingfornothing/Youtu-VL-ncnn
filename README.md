# Youtu-VL-ncnn

本项目为腾讯犀牛鸟课题成果。

Youtu-VL-ncnn 是
[`tencent/Youtu-VL-4B-Instruct`](https://huggingface.co/tencent/Youtu-VL-4B-Instruct)
的 pnnx/ncnn C++ CPU 推理实现。程序直接接收图片和 Prompt，由 C++ 完成
图片 Processor、Chat Template、ByteLevel-BPE 编解码和生成循环，由 ncnn
执行模型计算。

```text
image + prompt
  -> C++ image decode / resize / normalize / patchify
  -> C++ Chat Template + ByteLevel-BPE
  -> 27-layer Vision -> VLPatchMerger -> token fusion
  -> 40-layer LLM Prefill -> KV-cache Decode
  -> Final RMSNorm -> 17 LM-head shards
  -> coordinate postprocess + UTF-8 text
```

推理阶段不依赖 Python、PyTorch、Transformers、BLAS、MKL 或 OpenBLAS。
构建依赖为 C++17、CMake、ncnn、OpenMP 和 libjpeg；JPEG 以外的图片格式
通过仓库内的 `stb_image` 解码。

## 模型下载

模型权重与源码分开发布：

- 模型仓库：
  [Coderdw/Youtu-VL-4B-Instruct-ncnn](https://huggingface.co/Coderdw/Youtu-VL-4B-Instruct-ncnn/tree/main)
- 原始模型：
  [tencent/Youtu-VL-4B-Instruct](https://huggingface.co/tencent/Youtu-VL-4B-Instruct)

使用 Hugging Face CLI 下载模型包：

```bash
python3 -m pip install -U huggingface_hub
hf download Coderdw/Youtu-VL-4B-Instruct-ncnn \
  youtu-vl-ncnn-model.tar.gz MODEL_PACKAGE_SHA256.txt \
  --local-dir model-download

cd model-download
sha256sum -c MODEL_PACKAGE_SHA256.txt
tar -xzf youtu-vl-ncnn-model.tar.gz
```

压缩包解开后会生成 `model/`。完整运行包包含 339 个资产，解压后约
13.43 GiB，可在源码根目录检查文件数量和布局：

```bash
python3 tools/check_model_layout.py \
  --model-root /path/to/model-download/model

(cd /path/to/model-download/model && sha256sum -c checksums.sha256)
```

核心目录结构如下：

```text
model/
├── artifacts/
│   ├── vision_embedding/
│   ├── vision_layer0_masked_core/ ... vision_layer26_masked_core/
│   ├── vision_post_layernorm/
│   ├── llm_layer0_three_part/ ... llm_layer39_three_part/
│   ├── llm_final_head_ncnn/
│   ├── text_embedding/
│   └── llm_rope_inv_freq.npy
├── models/
│   └── youtu_merger.ncnn.{param,bin}
├── tokenizer/
│   └── tokenizer.bin
└── checksums.sha256
```

## 构建

### Linux

已验证环境为 Ubuntu 22.04、GNU 11.4、ncnn `20260526`、OpenMP 4.5，
CPU Release 构建且 Vulkan 关闭。

```bash
cmake -S . -B build/linux-release \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn

cmake --build build/linux-release --parallel
ctest --test-dir build/linux-release --output-on-failure
```

也可以使用快速复测入口：

```bash
NCNN_DIR=/path/to/ncnn/lib/cmake/ncnn \
  bash linux-test/code/run_linux_tests.sh
```

### Windows

已验证环境为 Windows Server 2022、MSYS2 UCRT64/GCC 16.1、CMake
4.4、Ninja 1.13、ncnn `20260526`、OpenMP 5.2 和 libjpeg-turbo 3.2。
当前验收范围是 UCRT64/GCC，不包含 MSVC。

```powershell
cmake -S . -B build/windows-release -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -Dncnn_DIR=C:\path\to\ncnn\lib\cmake\ncnn

cmake --build build/windows-release --parallel
ctest --test-dir build/windows-release --output-on-failure
```

Windows 动态发布包需要在可执行文件旁提供
`libgcc_s_seh-1.dll`、`libstdc++-6.dll`、`libgomp-1.dll`、
`libjpeg-8.dll` 和 `libwinpthread-1.dll`。ncnn 使用静态链接。

## 运行

Linux：

```bash
./build/linux-release/youtu_vl_ncnn \
  --model-root /path/to/model-download/model \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --output-dir out/run \
  --max-new-tokens 128 \
  --weight-mode persistent \
  --threads 8
```

Windows：

```powershell
.\build\windows-release\youtu_vl_ncnn.exe `
  --model-root C:\path\to\model-download\model `
  --image C:\path\to\image.jpg `
  --prompt "Describe this image." `
  --output-dir out\run `
  --max-new-tokens 128 `
  --weight-mode persistent `
  --threads 8
```

`persistent` 会在 Prefill 前加载 138 个 ncnn 图并在生成阶段复用，是推荐
模式；`streaming` 会降低常驻内存，但会重复读取模型。Persistent 模式实测
最大 RSS 约 24 GiB，完整推理至少需要 32 GiB 内存，推荐 64 GiB。

主要输出包括：

```text
processor_inputs/*.npy
raw_generated_token_ids.npy
generated_token_ids.npy
top1/top2 token IDs and logits
output.txt
runtime_report.json
```

## 验收结果

冻结条件为单图片、单 Prompt、batch size 1、greedy decode、8 threads、
CPU Release 和 Vulkan OFF。

| 验收范围 | 结果 |
| --- | --- |
| Linux 32/64/128 token | 最终 token ID 与 PyTorch 逐元素一致，最终文本一致 |
| Windows 1/32/64/128 token | 最终 token ID 与 PyTorch 逐元素一致，最终文本一致 |
| 英文 VQA、物体检测用例 | 64-token 回归均通过 |
| C++ Processor、Chat Template、Tokenizer | 通过 |
| Persistent 生命周期 | `resident_models=138`，`model_load_count=138` |
| 推理时 Python/Transformers 依赖 | 无 |
| BLAS/MKL/OpenBLAS/Vulkan 依赖 | 无 |

Linux Persistent 最终版相对重复加载基线，在 32/64/128 token 下的总耗时
分别获得约 1.64×、1.46× 和 1.36× 加速。

涉及模型导出、Tokenizer、图片缩放、坐标转换、数值内核、权重生命周期或
多线程策略的修改，都需要重新比较最终 token、文本、Processor Tensor、
Top-1/Top-2、耗时、RSS 和 swap。

## 源码结构

```text
.
├── include/                 C++ 前端、Tokenizer、ncnn 运行时和数值内核
├── src/                     CLI 与端到端推理实现
├── models/                  模型布局说明，不保存权重
├── tools/                   布局检查、打包和辅助工具
├── tests/                   平台无关测试源码
├── linux-test/              Linux 复测入口
├── windows-test/            Windows 复测入口
├── CMakeLists.txt
└── README.md
```

实验记录和技术文章不属于源码仓库。平台历史构建、工具链和原始结果仅在本地
归档，并已通过 `.gitignore` 排除。

## License

项目沿用 [Youtu-VL License](LICENSE)。第三方组件及其许可证说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
