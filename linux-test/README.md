# Linux 测试归档

本目录只保存 Linux 复测入口、历史构建和机器可读结果：

```text
linux-test/
├── code/run_linux_tests.sh
├── build/                      历史构建，不在新机器复用 CMake Cache
└── results/                    JSON、NPY、日志和输出文本
```

构建环境、模型下载和验收结论见项目根目录
[`README.md`](../README.md)。

快速复测：

```bash
NCNN_DIR=/path/to/ncnn/lib/cmake/ncnn \
  bash linux-test/code/run_linux_tests.sh
```

完整 32/64/128 回归：

```bash
NCNN_DIR=/path/to/ncnn/lib/cmake/ncnn \
MODEL_ROOT=/path/to/model \
FULL_REGRESSION=1 \
  bash linux-test/code/run_linux_tests.sh
```
