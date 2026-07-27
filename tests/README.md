# Tests

本目录保存平台无关的测试源码：

| 文件 | 用途 |
| --- | --- |
| `test_portable_gemm.cpp` | OpenMP FP32 GEMM 固定数值测试 |
| `test_persistent_module.cpp` | 真实 ncnn Module 单次加载、重复运行测试 |
| `compare_tokens.py` | generated IDs、Top-1/Top-2 和 logits 比较 |
| `check_persistent_run.py` | persistent 运行报告验收 |

构建和运行 CTest：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

完整模型回归条件和通过标准见项目根目录
[`README.md`](../README.md)。
