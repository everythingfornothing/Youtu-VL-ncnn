# Windows 测试归档

本目录保存 Windows/UCRT64 复测入口、历史工具链、构建和机器可读结果：

```text
windows-test/
├── run_tests.ps1
├── compare_npy.js
├── build/
├── results/
├── persistent-fixture/
├── ncnn-20260526/
├── ncnn-build/
├── ncnn-install/
└── toolchain/
```

大型工具链和构建目录是历史验收归档，不应提交到最终 GitHub 源码仓库。
构建环境、模型下载和验收结论见项目根目录
[`README.md`](../README.md)。

快速复测：

```powershell
powershell -ExecutionPolicy Bypass -File .\windows-test\run_tests.ps1
```

完整回归：

```powershell
powershell -ExecutionPolicy Bypass -File .\windows-test\run_tests.ps1 `
  -FullRegression -Threads 8
```
